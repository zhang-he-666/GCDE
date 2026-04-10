import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

class FlowDiffusionModule(nn.Module):
    def __init__(self, embedding_dim, num_timesteps=1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_timesteps = num_timesteps
        
        self.noise_pred_nets = nn.ModuleDict({
            'class': nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim*2),
                nn.ReLU(),
                nn.Linear(embedding_dim*2, embedding_dim)
            ),
            'student': nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim*2),
                nn.ReLU(),
                nn.Linear(embedding_dim*2, embedding_dim)
            ),
            'exercise': nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim*2),
                nn.ReLU(),
                nn.Linear(embedding_dim*2, embedding_dim)
            )
        })
        
        self.flow_attention = nn.ModuleDict({
            'student_to_class': nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=2),
            'class_to_student': nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=2),
            'student_to_exercise': nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=2),
            'exercise_to_student': nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=2),
            'class_to_class': nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=2)
        })
        
        self.flow_gates = nn.ModuleDict({
            'student_to_class': nn.Sequential(
                nn.Linear(embedding_dim*2, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 1),
                nn.Sigmoid()
            ),
            'class_to_student': nn.Sequential(
                nn.Linear(embedding_dim*2, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 1),
                nn.Sigmoid()
            ),
            'student_to_exercise': nn.Sequential(
                nn.Linear(embedding_dim*2, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 1),
                nn.Sigmoid()
            ),
            'exercise_to_student': nn.Sequential(
                nn.Linear(embedding_dim*2, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 1),
                nn.Sigmoid()
            )
        })
        
        self.consistency_module = nn.Sequential(
            nn.Linear(embedding_dim*3, embedding_dim*2),
            nn.ReLU(),
            nn.Linear(embedding_dim*2, 3),
            nn.Softmax(dim=-1)
        )
        
        self.beta_start = 0.0001
        self.beta_end = 0.02
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def forward(self, class_emb, stu_embs, exer_embs, timestep=0):
        t = min(max(timestep, 0), self.num_timesteps-1)
        
        noise_class = torch.randn_like(class_emb)
        noise_stu = torch.randn_like(stu_embs)
        noise_exer = torch.randn_like(exer_embs)
        
        sqrt_alphas = self.sqrt_alphas_cumprod[t].to(class_emb.device)
        sqrt_one_minus_alphas = self.sqrt_one_minus_alphas_cumprod[t].to(class_emb.device)
        
        noisy_class = sqrt_alphas * class_emb + sqrt_one_minus_alphas * noise_class
        noisy_stu = sqrt_alphas * stu_embs + sqrt_one_minus_alphas * noise_stu
        noisy_exer = sqrt_alphas * exer_embs + sqrt_one_minus_alphas * noise_exer
        
        # Flow updates
        stu_mean = torch.mean(noisy_stu, dim=0, keepdim=True)
        gate_stu_to_class = self.flow_gates['student_to_class'](torch.cat([noisy_class, stu_mean], dim=-1))
        attn_stu_to_class, attn_stu_to_class_w = self.flow_attention['student_to_class'](noisy_class, noisy_stu, noisy_stu)
        updated_class = noisy_class + gate_stu_to_class * attn_stu_to_class
        
        gate_class_to_stu = self.flow_gates['class_to_student'](torch.cat([updated_class, stu_mean], dim=-1))
        attn_class_to_stu, attn_class_to_stu_w = self.flow_attention['class_to_student'](noisy_stu, updated_class, updated_class)
        updated_stu = noisy_stu + gate_class_to_stu * attn_class_to_stu
        
        exer_mean = torch.mean(noisy_exer, dim=0, keepdim=True)
        gate_stu_to_exer = self.flow_gates['student_to_exercise'](torch.cat([stu_mean, exer_mean], dim=-1))
        attn_stu_to_exer, attn_stu_to_exer_w = self.flow_attention['student_to_exercise'](noisy_exer, updated_stu, updated_stu)
        updated_exer = noisy_exer + gate_stu_to_exer * attn_stu_to_exer
        
        gate_exer_to_stu = self.flow_gates['exercise_to_student'](torch.cat([stu_mean, exer_mean], dim=-1))
        attn_exer_to_stu, attn_exer_to_stu_w = self.flow_attention['exercise_to_student'](updated_stu, updated_exer, updated_exer)
        updated_stu = updated_stu + gate_exer_to_stu * attn_exer_to_stu
        
        attn_class_to_class, attn_class_to_class_w = self.flow_attention['class_to_class'](updated_class, updated_class, updated_class)
        updated_class = updated_class + attn_class_to_class
        
        pred_noise_class = self.noise_pred_nets['class'](updated_class)
        pred_noise_stu = self.noise_pred_nets['student'](updated_stu)
        pred_noise_exer = self.noise_pred_nets['exercise'](updated_exer)
        
        diffusion_loss_class = F.mse_loss(pred_noise_class, noise_class)
        diffusion_loss_stu = F.mse_loss(pred_noise_stu, noise_stu)
        diffusion_loss_exer = F.mse_loss(pred_noise_exer, noise_exer)
        
        consistency_input = torch.cat([torch.mean(updated_class, dim=0, keepdim=True), torch.mean(updated_stu, dim=0, keepdim=True), torch.mean(updated_exer, dim=0, keepdim=True)], dim=-1)
        consistency_weights = self.consistency_module(consistency_input).squeeze(0)
        
        diffusion_loss = consistency_weights[0] * diffusion_loss_class + consistency_weights[1] * diffusion_loss_stu + consistency_weights[2] * diffusion_loss_exer
        
        # collect explainability artifacts
        try:
            g_sc = gate_stu_to_class.mean().detach()
            g_cs = gate_class_to_stu.mean().detach()
            g_se = gate_stu_to_exer.mean().detach()
            g_es = gate_exer_to_stu.mean().detach()
            a_sc = attn_stu_to_class_w.abs().mean().detach()
            a_cs = attn_class_to_stu_w.abs().mean().detach()
            a_se = attn_stu_to_exer_w.abs().mean().detach()
            a_es = attn_exer_to_stu_w.abs().mean().detach()
            a_cc = attn_class_to_class_w.abs().mean().detach()
            path_scores = {
                'student_to_class': (g_sc * a_sc).cpu(),
                'class_to_student': (g_cs * a_cs).cpu(),
                'student_to_exercise': (g_se * a_se).cpu(),
                'exercise_to_student': (g_es * a_es).cpu(),
                'class_to_class': a_cc.cpu()
            }
            explain = {
                'gates': {
                    'student_to_class': g_sc.cpu(),
                    'class_to_student': g_cs.cpu(),
                    'student_to_exercise': g_se.cpu(),
                    'exercise_to_student': g_es.cpu()
                },
                'attentions': {
                    'student_to_class': a_sc.cpu(),
                    'class_to_student': a_cs.cpu(),
                    'student_to_exercise': a_se.cpu(),
                    'exercise_to_student': a_es.cpu(),
                    'class_to_class': a_cc.cpu()
                },
                'path_scores': path_scores
            }
        except Exception:
            explain = None
        
        return diffusion_loss, updated_class, updated_stu, updated_exer, consistency_weights, explain

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(in_features, out_features)
        self.linear3 = nn.Linear(in_features, out_features)
        self.linear4 = nn.Linear(in_features, out_features)
        self.linear5 = nn.Linear(in_features, out_features)
        self.linear6 = nn.Linear(in_features, out_features)
 
    def forward(self, sparse_adj_t, sparse_adj_f, class_emb, stu_embs, exer_embs):
        s2c_emb = torch.mean(self.linear3(stu_embs), dim=0, keepdim=True)
        c2s_emb = self.linear3(class_emb) / stu_embs.size(0)
        s2s_emb = self.linear4(stu_embs)
        c2c_emb = self.linear5(class_emb)
        e2e_emb = self.linear6(exer_embs)
        
        if sparse_adj_f is None:
            if sparse_adj_t.is_sparse:
                stu_emb_t = self.linear1(torch.sparse.mm(sparse_adj_t, exer_embs))
                exer_embs_t = self.linear1(torch.sparse.mm(sparse_adj_t.t(), stu_embs))
            else:
                stu_emb_t = self.linear1(torch.matmul(sparse_adj_t, exer_embs))
                exer_embs_t = self.linear1(torch.matmul(sparse_adj_t.t(), stu_embs))

            stu_embs_new = (stu_emb_t + c2s_emb) / 2 + s2s_emb
            exer_embs_new = exer_embs_t + e2e_emb
            class_emb_new = s2c_emb + c2c_emb

        elif sparse_adj_t is None:
            if sparse_adj_f.is_sparse:
                stu_emb_f = self.linear2(torch.sparse.mm(sparse_adj_f, exer_embs))
                exer_embs_f = self.linear2(torch.sparse.mm(sparse_adj_f.t(), stu_embs))
            else:
                stu_emb_f = self.linear2(torch.matmul(sparse_adj_f, exer_embs))
                exer_embs_f = self.linear2(torch.matmul(sparse_adj_f.t(), stu_embs))

            stu_embs_new = (stu_emb_f + c2s_emb) / 2 + s2s_emb
            exer_embs_new = exer_embs_f + e2e_emb
            class_emb_new = s2c_emb + c2c_emb
            
        else:
            if sparse_adj_t.is_sparse:
                stu_emb_t = self.linear1(torch.sparse.mm(sparse_adj_t, exer_embs))
                exer_embs_t = self.linear1(torch.sparse.mm(sparse_adj_t.t(), stu_embs))
            else:
                stu_emb_t = self.linear1(torch.matmul(sparse_adj_t, exer_embs))
                exer_embs_t = self.linear1(torch.matmul(sparse_adj_t.t(), stu_embs))
                
            if sparse_adj_f.is_sparse:
                stu_emb_f = self.linear2(torch.sparse.mm(sparse_adj_f, exer_embs))
                exer_embs_f = self.linear2(torch.sparse.mm(sparse_adj_f.t(), stu_embs))
            else:
                stu_emb_f = self.linear2(torch.matmul(sparse_adj_f, exer_embs))
                exer_embs_f = self.linear2(torch.matmul(sparse_adj_f.t(), stu_embs))

            stu_embs_new = (stu_emb_t + stu_emb_f + c2s_emb) / 3 + s2s_emb
            exer_embs_new = (exer_embs_t + exer_embs_f) / 2 + e2e_emb
            class_emb_new = s2c_emb + c2c_emb

        return class_emb_new, stu_embs_new, exer_embs_new

class NoneNegClipper(object):
    def __init__(self):
        super().__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)

class DiffuGCD(nn.Module):
    def __init__(self,class_n,stu_n,exer_n,skill_n,t):
        super().__init__()
        self.class_n = class_n
        self.stu_n = stu_n
        self.exer_n = exer_n
        self.skill_n = skill_n
        self.prednet_input_len = skill_n
        self.prednet_len1, self.prednet_len2, self.prednet_len3 = 256,128,1
        self.pi = t 

        self.class_emb = nn.Embedding(self.class_n, self.skill_n)
        self.stu_emb = nn.Embedding(self.stu_n, self.skill_n)
        self.exer_diff = nn.Embedding(self.exer_n, self.skill_n)
        self.exer_dis = nn.Embedding(self.exer_n, 1)
        self.mu = nn.Linear(2*self.skill_n, 1)
        self.logvar = nn.Linear(2*self.skill_n, 1)
        self.conv1 = GCNLayer(self.skill_n, self.skill_n)
        self.conv2 = GCNLayer(self.skill_n, self.skill_n)
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)
        
        self.flow_diffusion = FlowDiffusionModule(self.skill_n)
        
        self.flow_weight_layer = nn.Linear(self.skill_n, 1)
        self.last_explanations = None

        for name,param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.normal_(param, mean=0.0, std=0.01)
    
    def forward(self,edge_t,edge_f,class_id,stu_list,kn_emb,exer_list,exer_test=None):
        if exer_test is not None:
            return self.testforward(edge_t,edge_f,class_id,stu_list,exer_list,kn_emb,exer_test)
        else:
            return self.trainforward(edge_t,edge_f,class_id,stu_list,exer_list,kn_emb)
    
    def trainforward(self,edge_t,edge_f,class_id,stu_list,exer_list,kn_emb):
        stu_embeddings = self.stu_emb(stu_list)
        exer_embeddings = self.exer_diff(exer_list)
        class_emb = self.class_emb(class_id)
        exer_dis = torch.sigmoid(self.exer_dis(exer_list))*10

        timestep = torch.randint(0, self.flow_diffusion.num_timesteps, (1,)).item()
        diffusion_loss, diffused_class_emb, diffused_stu_embs, diffused_exer_embs, consistency_weights, explain = self.flow_diffusion(
            class_emb.unsqueeze(0), stu_embeddings, exer_embeddings, timestep
        )
        diffused_class_emb = diffused_class_emb.squeeze(0) if diffused_class_emb.dim() > 1 else diffused_class_emb
        
        flow_weights = torch.sigmoid(self.flow_weight_layer(diffused_class_emb))
        # store explanations for external access
        if explain is not None:
            try:
                self.last_explanations = {
                    'consistency_weights': consistency_weights.detach().cpu().tolist(),
                    'flow_weight_mean': float(flow_weights.detach().mean().cpu()),
                    'paths': {k: float(v) for k, v in explain['path_scores'].items()}
                }
            except Exception:
                self.last_explanations = None
        
        if edge_f.numel() == 0:
            combined_embeddings_t = torch.cat([diffused_stu_embs[edge_t[0]], diffused_exer_embs[edge_t[1]]], dim=1)
            edge_mu_t = self.mu(combined_embeddings_t)
            edge_logvar_t = self.logvar(combined_embeddings_t)
            edge_std_t = torch.exp(0.5 * edge_logvar_t)
            kl_loss = self.kl_loss(edge_mu_t, edge_logvar_t)
            edge_t_1 = edge_mu_t + edge_std_t * torch.randn_like(edge_std_t)
            edge_t_1 = torch.sigmoid(edge_t_1).squeeze()
            edge_t_1 = RelaxedBernoulli(self.pi,probs=edge_t_1).rsample()
            adj_t_1 = torch.sparse.FloatTensor(edge_t, edge_t_1, (stu_list.size(0), exer_list.size(0)))
            
            class_emb_1,stu_embs_1,exer_embs_1 = self.conv1(adj_t_1, None, diffused_class_emb, diffused_stu_embs, diffused_exer_embs)
            class_emb_2,stu_embs_2,exer_embs_2 = self.conv2(adj_t_1, None, class_emb_1, stu_embs_1, exer_embs_1)
        elif edge_t.numel() == 0:
            combined_embeddings_f = torch.cat([diffused_stu_embs[edge_f[0]], diffused_exer_embs[edge_f[1]]], dim=1)
            edge_mu_f = self.mu(combined_embeddings_f)
            edge_logvar_f = self.logvar(combined_embeddings_f)
            edge_std_f = torch.exp(0.5 * edge_logvar_f)
            kl_loss = self.kl_loss(edge_mu_f, edge_logvar_f)
            edge_f_1 = edge_mu_f + edge_std_f * torch.randn_like(edge_std_f)
            edge_f_1 = torch.sigmoid(edge_f_1).squeeze() 
            edge_f_1 = RelaxedBernoulli(self.pi,probs=edge_f_1).rsample()
            adj_f_1 = torch.sparse.FloatTensor(edge_f, edge_f_1, (stu_list.size(0), exer_list.size(0)))
            class_emb_1,stu_embs_1,exer_embs_1 = self.conv1(None, adj_f_1, diffused_class_emb, diffused_stu_embs, diffused_exer_embs)
            class_emb_2,stu_embs_2,exer_embs_2 = self.conv2(None, adj_f_1, class_emb_1, stu_embs_1, exer_embs_1)
        else:
            combined_embeddings_t = torch.cat([diffused_stu_embs[edge_t[0]], diffused_exer_embs[edge_t[1]]], dim=1)
            edge_mu_t = self.mu(combined_embeddings_t)
            edge_logvar_t = self.logvar(combined_embeddings_t)
            edge_std_t = torch.exp(0.5 * edge_logvar_t)

            combined_embeddings_f = torch.cat([diffused_stu_embs[edge_f[0]], diffused_exer_embs[edge_f[1]]], dim=1)
            edge_mu_f = self.mu(combined_embeddings_f)
            edge_logvar_f = self.logvar(combined_embeddings_f)
            edge_std_f = torch.exp(0.5 * edge_logvar_f)

            kl_loss = self.kl_loss(edge_mu_t, edge_logvar_t) + self.kl_loss(edge_mu_f, edge_logvar_f)

            edge_t_1 = edge_mu_t + edge_std_t * torch.randn_like(edge_std_t)
            edge_f_1 = edge_mu_f + edge_std_f * torch.randn_like(edge_std_f)
            edge_t_1 = torch.sigmoid(edge_t_1).squeeze() 
            edge_f_1 = torch.sigmoid(edge_f_1).squeeze() 
            edge_t_1 = RelaxedBernoulli(self.pi,probs=edge_t_1).rsample()
            edge_f_1 = RelaxedBernoulli(self.pi,probs=edge_f_1).rsample()

            adj_t_1 = torch.sparse.FloatTensor(edge_t, edge_t_1, (stu_list.size(0), exer_list.size(0)))
            adj_f_1 = torch.sparse.FloatTensor(edge_f, edge_f_1, (stu_list.size(0), exer_list.size(0)))

            class_emb_1,stu_embs_1,exer_embs_1 = self.conv1(adj_t_1, adj_f_1, diffused_class_emb, diffused_stu_embs, diffused_exer_embs)
            class_emb_2,stu_embs_2,exer_embs_2 = self.conv2(adj_t_1, adj_f_1, class_emb_1, stu_embs_1, exer_embs_1)
        
        class_k = torch.sigmoid(class_emb_2)
        exer_k = torch.sigmoid(exer_embeddings)
        
        if class_k.dim() == 1:
            class_k = class_k.unsqueeze(0)
        if kn_emb.dim() >1:
            kn_emb = kn_emb.mean(dim=0)
        if kn_emb.size(0) != self.skill_n:
            kn_emb = torch.ones(self.skill_n, device=class_k.device)
        
        class_k_exp = class_k.unsqueeze(1) 
        diff = class_k_exp - exer_k.unsqueeze(0) 
        diff_weighted = diff * kn_emb.unsqueeze(0).unsqueeze(1) 
        exer_dis_exp = exer_dis.unsqueeze(0).unsqueeze(2)
        input_x = (diff_weighted * exer_dis_exp).squeeze(0)
        
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x)) 
        
        return output, kl_loss, diffusion_loss
    
    def testforward(self,edge_t,edge_f,class_id,stu_list,exer_list,kn_emb,exer_test):
        class_emb = self.class_emb(class_id)
        stu_embeddings = self.stu_emb(stu_list)
        exer_embeddings = self.exer_diff(exer_list)
        
        _, diffused_class_emb, diffused_stu_embs, diffused_exer_embs, _, explain = self.flow_diffusion(
            class_emb.unsqueeze(0), stu_embeddings, exer_embeddings, timestep=0
        )
        diffused_class_emb = diffused_class_emb.squeeze(0) if diffused_class_emb.dim() > 1 else diffused_class_emb
        
        flow_weights = torch.sigmoid(self.flow_weight_layer(diffused_class_emb))
        if explain is not None:
            try:
                self.last_explanations = {
                    'consistency_weights': None,
                    'flow_weight_mean': float(flow_weights.detach().mean().cpu()),
                    'paths': {k: float(v) for k, v in explain['path_scores'].items()}
                }
            except Exception:
                self.last_explanations = None
        
        if edge_f.numel() == 0:
            combined_embeddings_t = torch.cat([diffused_stu_embs[edge_t[0]], diffused_exer_embs[edge_t[1]]], dim=1)
            edge_mu_t = self.mu(combined_embeddings_t)
            edge_t_1 = torch.sigmoid(edge_mu_t).squeeze()
            edge_t_1 = RelaxedBernoulli(self.pi,probs=edge_t_1).rsample()
            adj_t_1 = torch.sparse.FloatTensor(edge_t, edge_t_1, (stu_list.size(0), exer_list.size(0)))
            
            class_emb_1,stu_embs_1,exer_embs_1 = self.conv1(adj_t_1, None, diffused_class_emb, diffused_stu_embs, diffused_exer_embs)
            class_emb_2,stu_embs_2,exer_embs_2 = self.conv2(adj_t_1, None, class_emb_1, stu_embs_1, exer_embs_1)
        elif edge_t.numel() == 0:
            combined_embeddings_f = torch.cat([diffused_stu_embs[edge_f[0]], diffused_exer_embs[edge_f[1]]], dim=1)
            edge_mu_f = self.mu(combined_embeddings_f)
            edge_f_1 = torch.sigmoid(edge_mu_f).squeeze()
            edge_f_1 = RelaxedBernoulli(self.pi,probs=edge_f_1).rsample()
            adj_f_1 = torch.sparse.FloatTensor(edge_f, edge_f_1, (stu_list.size(0), exer_list.size(0)))
            class_emb_1,stu_embs_1,exer_embs_1 = self.conv1(None, adj_f_1, diffused_class_emb, diffused_stu_embs, diffused_exer_embs)
            class_emb_2,stu_embs_2,exer_embs_2 = self.conv2(None, adj_f_1, class_emb_1, stu_embs_1, exer_embs_1)
        else:
            combined_embeddings_t = torch.cat([diffused_stu_embs[edge_t[0]], diffused_exer_embs[edge_t[1]]], dim=1)
            edge_mu_t = self.mu(combined_embeddings_t)
            edge_t_1 = torch.sigmoid(edge_mu_t).squeeze()
            edge_t_1 = RelaxedBernoulli(self.pi,probs=edge_t_1).rsample()

            combined_embeddings_f = torch.cat([diffused_stu_embs[edge_f[0]], diffused_exer_embs[edge_f[1]]], dim=1)
            edge_mu_f = self.mu(combined_embeddings_f)
            edge_f_1 = torch.sigmoid(edge_mu_f).squeeze()
            edge_f_1 = RelaxedBernoulli(self.pi,probs=edge_f_1).rsample()

            adj_t_1 = torch.sparse.FloatTensor(edge_t, edge_t_1, (stu_list.size(0), exer_list.size(0)))
            adj_f_1 = torch.sparse.FloatTensor(edge_f, edge_f_1, (stu_list.size(0), exer_list.size(0)))

            class_emb_1,stu_embs_1,exer_embs_1 = self.conv1(adj_t_1, adj_f_1, diffused_class_emb, diffused_stu_embs, diffused_exer_embs)
            class_emb_2,stu_embs_2,exer_embs_2 = self.conv2(adj_t_1, adj_f_1, class_emb_1, stu_embs_1, exer_embs_1)
        
        class_k = torch.sigmoid(class_emb_2)
        exer_dis = torch.sigmoid(self.exer_dis(exer_test))*10
        exer_k = torch.sigmoid(self.exer_diff(exer_test))
        
        if class_k.dim() == 1:
            class_k = class_k.unsqueeze(0)
        if kn_emb.dim() >1:
            kn_emb = kn_emb.mean(dim=0)
        if kn_emb.size(0) != self.skill_n:
            kn_emb = torch.ones(self.skill_n, device=class_k.device)
        
        class_k_exp = class_k.unsqueeze(1) 
        diff = class_k_exp - exer_k.unsqueeze(0) 
        diff_weighted = diff * kn_emb.unsqueeze(0).unsqueeze(1)
        exer_dis_exp = exer_dis.unsqueeze(0).unsqueeze(2)
        input_x = (diff_weighted * exer_dis_exp).squeeze(0) 
        
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x)) 
        
        kl_loss = torch.tensor(0.0).to(output.device)
        diffusion_loss = torch.tensor(0.0).to(output.device)
        return output, kl_loss, diffusion_loss
    
    def kl_loss(self,mu,logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp() + 1e-6)

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def get_last_explanations(self):
        return self.last_explanations