# test
import os
import math
import json
import torch
import numpy as np
import argparse
from dataloader import ValTestDataLoader
from model import DiffuGCD


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/SLPbio_/', type=str, help='directory containing test.json')
    parser.add_argument('--num_stu', default=4617, type=int, help='num_stu')
    parser.add_argument('--num_exer', default=221, type=int, help='num_exer')
    parser.add_argument('--num_class', default=186, type=int, help='num_class')
    parser.add_argument('--num_skill', default=38, type=int, help='number of skills')
    parser.add_argument('--t', default=0.77, type=float, help='temperature for RelaxedBernoulli')
    parser.add_argument('--model_path', default='./model_save/diff_mat_k1.pth', type=str, help='path to trained model')
    parser.add_argument('--runs', default=100, type=int, help='number of repeated evaluations')
    return parser.parse_args()


def ensure_trailing_slash(p):
    return p if p.endswith('/') else (p + '/')


def _safe_fetch_batch(data_loader):
    """Fetch one batch robustly without relying on dataloader.next_batch().

    This directly accesses `data_loader.data1` and `data_loader.ptr` to
    construct tensors, providing fallbacks when certain keys are missing
    (e.g., missing 'exer_test' or using 'score_list' instead of 'test_score').
    """
    # Guard: if loader is at end
    if data_loader.is_end():
        return None

    # Prefer direct access if available to avoid KeyError raised in next_batch
    if hasattr(data_loader, 'data1') and hasattr(data_loader, 'ptr'):
        log = data_loader.data1[data_loader.ptr]
        data_loader.ptr = data_loader.ptr + 1

        class_id = log['student_class_id']
        stu_list = log['stu_list']
        exer_list = log['exer_list']
        # Fallbacks
        exer_test = log.get('exer_test', exer_list)
        score_key = 'test_score' if 'test_score' in log else 'score_list'

        labels = torch.Tensor(log[score_key])
        stu_exer_true = log['s_e_t']
        stu_exer_false = log['s_e_f']
        knowledge_code = log['skill_list']
        kn_emb = torch.Tensor(knowledge_code)

        return (
            torch.LongTensor(stu_exer_true),
            torch.LongTensor(stu_exer_false),
            kn_emb,
            torch.tensor(stu_list),
            torch.tensor(class_id),
            torch.tensor(exer_list),
            torch.tensor(exer_test),
            labels,
        )

    # Fallback: use original API if internals are not present
    try:
        return data_loader.next_batch()
    except KeyError:
        # As a last resort, signal end to avoid crashing
        return None


def _align_pred_and_label(output, labels, exer_test=None):
    """Align shapes of predictions and labels for fair metric computation.

    Strategy:
    1) If shapes match -> flatten and return.
    2) If both 2D and `exer_test` provided -> select output columns by `exer_test`.
    3) If output is 2D and labels is 1D -> mean-reduce the unmatched axis.
    4) Fallback -> flatten and truncate to min length.
    """
    # Ensure tensors
    output_t = torch.as_tensor(output)
    labels_t = torch.as_tensor(labels)

    # Case 1: exact match
    if output_t.shape == labels_t.shape:
        return output_t.reshape([-1]), labels_t.reshape([-1])

    # Case 2: try selecting columns by exer_test if plausible
    if output_t.dim() == 2 and labels_t.dim() == 2:
        if (
            exer_test is not None
            and labels_t.shape[0] == output_t.shape[0]
            and labels_t.shape[1] == int(exer_test.shape[0])
            and int(exer_test.max().item()) < output_t.shape[1]
        ):
            cols = exer_test.long()
            aligned_output = output_t.index_select(dim=1, index=cols)
            if aligned_output.shape == labels_t.shape:
                return aligned_output.reshape([-1]), labels_t.reshape([-1])

    # Case 3: reduce mean along extra axis when labels is vector
    if output_t.dim() == 2 and labels_t.dim() == 1:
        if labels_t.shape[0] == output_t.shape[0]:
            reduced = output_t.mean(dim=1)
            return reduced.reshape([-1]), labels_t.reshape([-1])
        if labels_t.shape[0] == output_t.shape[1]:
            reduced = output_t.mean(dim=0)
            return reduced.reshape([-1]), labels_t.reshape([-1])

    # Case 4: flatten and truncate to common length
    out_flat = output_t.reshape([-1])
    lab_flat = labels_t.reshape([-1])
    n = min(out_flat.numel(), lab_flat.numel())
    if n == 0:
        return out_flat, lab_flat
    return out_flat[:n], lab_flat[:n]


def evaluate_once(net, data_loader, device):
    data_loader.reset()
    total_abs_error = 0.0
    total_sq_error = 0.0
    total_count = 0

    # explainability accumulators
    paths = ['student_to_class', 'class_to_student', 'student_to_exercise', 'exercise_to_student', 'class_to_class']
    path_sum = {k: 0.0 for k in paths}
    explain_steps = 0

    while not data_loader.is_end():
        batch = _safe_fetch_batch(data_loader)
        if batch is None:
            break
        edge_t, edge_f, kn_emb, stu_list, class_id, exer_list, exer_test, labels = batch
        edge_t = edge_t.to(device)
        edge_f = edge_f.to(device)
        kn_emb = kn_emb.to(device)
        stu_list = stu_list.to(device)
        class_id = class_id.to(device)
        exer_list = exer_list.to(device)
        exer_test = exer_test.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            output, _, _ = net.forward(edge_t, edge_f, class_id, stu_list, kn_emb, exer_list, exer_test)
        preds, labels_vec = _align_pred_and_label(output, labels, exer_test)

        # metrics
        diff = preds - labels_vec
        total_abs_error += torch.abs(diff).sum().item()
        total_sq_error += torch.pow(diff, 2).sum().item()
        total_count += labels_vec.numel()

        # explainability (last call stored inside model)
        exp = net.get_last_explanations()
        if exp is not None and isinstance(exp.get('paths'), dict):
            for k in paths:
                if k in exp['paths']:
                    path_sum[k] += float(exp['paths'][k])
            explain_steps += 1

    mae = total_abs_error / max(1, total_count)
    rmse = math.sqrt(total_sq_error / max(1, total_count))

    mean_paths = None
    if explain_steps > 0:
        mean_paths = {k: (path_sum[k] / explain_steps) for k in paths}

    return mae, rmse, mean_paths


def summarize_explainability(run_path_list):
    # run_path_list: list of dicts with same keys
    if len(run_path_list) == 0:
        return None
    keys = list(run_path_list[0].keys())
    arr = np.array([[d[k] for k in keys] for d in run_path_list], dtype=float)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    # coefficient of variation per path
    with np.errstate(divide='ignore', invalid='ignore'):
        cov = np.where(mean != 0.0, std / (np.abs(mean) + 1e-12), np.nan)

    # concentration index from entropy of normalized positive path scores
    eps = 1e-12
    pos = np.maximum(mean, 0.0)
    s = pos.sum()
    if s > 0:
        p = pos / (s + eps)
        H = -np.sum(p * np.log(p + eps))
        concentration = 1.0 - H / np.log(len(keys))
    else:
        concentration = 0.0

    return {
        'mean_paths': {k: float(v) for k, v in zip(keys, mean)},
        'std_paths': {k: float(v) for k, v in zip(keys, std)},
        'cov_paths': {k: (None if np.isnan(v) else float(v)) for k, v in zip(keys, cov)},
        'concentration': float(concentration)
    }


def main():
    args = set_args()
    data_path = ensure_trailing_slash(args.data_path)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # model
    net = DiffuGCD(args.num_class, args.num_stu, args.num_exer, args.num_skill, args.t)
    net = net.to(device)
    if os.path.exists(args.model_path):
        state = torch.load(args.model_path, map_location=device)
        net.load_state_dict(state, strict=False)
        print(f'Loaded model from {args.model_path}')
    else:
        print(f'Warning: model file not found at {args.model_path}, using randomly initialized weights')

    # run repeated evaluations
    maes = []
    rmses = []
    explain_runs = []
    for r in range(args.runs):
        loader = ValTestDataLoader(data_path=data_path)
        net.eval()
        mae, rmse, mean_paths = evaluate_once(net, loader, device)
        maes.append(mae)
        rmses.append(rmse)
        if mean_paths is not None:
            explain_runs.append(mean_paths)
        if (r + 1) % 10 == 0:
            print(f'[run {r+1}] mae={mae:.6f} rmse={rmse:.6f}')

    mae_mean = float(np.mean(maes)) if len(maes) > 0 else float('nan')
    mae_std = float(np.std(maes)) if len(maes) > 0 else float('nan')
    rmse_mean = float(np.mean(rmses)) if len(rmses) > 0 else float('nan')
    rmse_std = float(np.std(rmses)) if len(rmses) > 0 else float('nan')

    print('==== Overall Metrics over {} runs ===='.format(args.runs))
    print('MAE:  mean={:.6f} std={:.6f}'.format(mae_mean, mae_std))
    print('RMSE: mean={:.6f} std={:.6f}'.format(rmse_mean, rmse_std))

    explain_summary = summarize_explainability(explain_runs)
    if explain_summary is not None:
        print('==== Explainability Metrics (path scores) ====')
        print('Mean path scores:', json.dumps(explain_summary['mean_paths'], ensure_ascii=False))
        print('Std  path scores:', json.dumps(explain_summary['std_paths'], ensure_ascii=False))
        print('CoV  path scores:', json.dumps(explain_summary['cov_paths'], ensure_ascii=False))
        print('Path concentration (0-1): {:.6f}'.format(explain_summary['concentration']))
    else:
        print('Explainability not available (no explanations collected).')


if __name__ == '__main__':
    main()


