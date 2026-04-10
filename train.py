#train
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from dataloader import TrainDataLoader
from model import DiffuGCD
import argparse

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/SLPbio_', type=str, help='')
    parser.add_argument('--num_stu', default=4617, type=int, help='num_stu')
    parser.add_argument('--num_exer', default=221, type=int, help='num_exer')
    parser.add_argument('--num_class', default=186, type=int, help='num_class')
    parser.add_argument('--num_train_epochs', default=100, type=int, help='number of training epochs')
    parser.add_argument('--num_skill', default=38, type=int, help='number of skills')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size number')
    parser.add_argument('--t', default=0.77, type=float, help='')
    parser.add_argument('--reg_r', default=1e-4, type=float, help='')
    parser.add_argument('--kl_r', default=1e-6, type=float, help='')
    parser.add_argument('--diff_r', default=1e-4, type=float, help='diffusion loss weight')
    parser.add_argument('--model_save_path', default='./model_save/', type=str, help='model save path')
    parser.add_argument('--explain_every', default=0, type=int, help='print explainability every N steps; 0 to disable')
    return parser.parse_args()


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

result_r_m = [[[1.0],[1.0]] for i in range(5)]
def train():
    data_loader = TrainDataLoader(data_path)
    net = DiffuGCD(class_n,student_n,exer_n,skill_n,t)

    net = net.to(device)
    optimizer = optim.Adam(net.parameters(),lr=learn,weight_decay=reg_r)
    print('training model...')
    loss_function = nn.MSELoss()
    rmsem = 1.0
    maem = 1.0
    for epoch in range(epochs):
        data_loader.reset()
        running_loss = 0.0
        count = 1
        while not data_loader.is_end():
            edge_t,edge_f,kn_emb,stu_list,class_id,exer_list,labels= data_loader.next_batch()
            batch_size = bsize
            start = 0
            edge_t,edge_f,kn_emb, stu_list, class_id, exer_list,labels = edge_t.to(device),edge_f.to(device),kn_emb.to(device),stu_list.to(device),class_id.to(device),exer_list.to(device),\
                                                                            labels.to(device)
            while (start + batch_size) <= len(exer_list):
                optimizer.zero_grad()
                output0_1, kl_loss, diffusion_loss = net.forward(edge_t,edge_f,class_id, stu_list, kn_emb, exer_list,None)
                output1_1 = output0_1.reshape([-1])
                output_1 = output1_1[start:(start+batch_size)]
                loss = loss_function(output_1, labels[start:(start+batch_size)]) + args.kl_r * kl_loss + args.diff_r * diffusion_loss
                loss.backward()
                optimizer.step()
                net.apply_clipper()
                running_loss += loss.item()
                start = start + batch_size
                count = count + 1
                if count%100 == 0 :
                    print('[%d,%5d] loss: %.4f, kl_loss: %.4f, diffusion_loss: %.4f'%(
                        epoch+1, count, running_loss/batch_size, kl_loss.item(), diffusion_loss.item()))
                    running_loss = 0.0
                if args.explain_every > 0 and count % args.explain_every == 0:
                    exp = net.get_last_explanations()
                    if exp is not None:
                        print('[explain] step=%d paths=%s flow_weight_mean=%.6f consistency=%s' % (
                            count,
                            str(exp.get('paths')),
                            exp.get('flow_weight_mean', float('nan')),
                            str(exp.get('consistency_weights'))
                        ))
                   
            if start < len(exer_list):
                optimizer.zero_grad()
                output0_1, kl_loss, diffusion_loss = net.forward(edge_t,edge_f,class_id, stu_list, kn_emb, exer_list,None)
                output1_1 = output0_1.reshape([-1])
                output_1 = output1_1[start:len(exer_list)]
                loss = loss_function(output_1, labels[start:len(exer_list)]) + args.kl_r * kl_loss + args.diff_r * diffusion_loss
                loss.backward()
                optimizer.step()
                net.apply_clipper()
                running_loss += loss.item()
                count = count + 1
                if count%100 == 0 :
                    print('[%d,%5d] loss: %.4f, kl_loss: %.4f, diffusion_loss: %.4f'%(
                        epoch+1, count, running_loss/batch_size, kl_loss.item(), diffusion_loss.item()))
                    running_loss = 0.0
                if args.explain_every > 0 and count % args.explain_every == 0:
                    exp = net.get_last_explanations()
                    if exp is not None:
                        print('[explain] step=%d paths=%s flow_weight_mean=%.6f consistency=%s' % (
                            count,
                            str(exp.get('paths')),
                            exp.get('flow_weight_mean', float('nan')),
                            str(exp.get('consistency_weights'))
                        ))
    os.makedirs("model_save", exist_ok=True)
    model_path = os.path.join("model_save", f"diff_mat_k1.pth")
    torch.save(net.state_dict(), model_path)
    print(f"Saved PyTorch Model State to {model_path}")

if __name__ == '__main__':
    args = set_args()
    knowledge_n = args.num_skill
    student_n = args.num_stu
    class_n = args.num_class
    exer_n = args.num_exer
    epochs = args.num_train_epochs
    skill_n = args.num_skill
    learn = args.lr
    data_path = args.data_path
    bsize = args.batch_size
    t = args.t
    reg_r = args.reg_r
    train()