import torch
import numpy as np
import math
import os
import csv

def normalize(R):
    mean = R.mean(dim=-2,keepdim=True)
    R = R - mean
    norms = torch.norm(R, p=2, dim=-1, keepdim=True)
    R = R/norms
    return R

# 三维(单个batch)
def cal_cov_3(R):
    Z = torch.nn.functional.normalize(R, dim=-1)
    A = torch.einsum('bji,bjk->bik',Z,Z) / Z.shape[-2]
    return A

def cal_entropy(A):
    A = A.contiguous()
    A_np = A.cpu().numpy()
    traces_np = np.trace(A_np, axis1=-2, axis2=-1)
    traces_np = traces_np[np.newaxis, np.newaxis]
    epsilon = 1e-10  
    eig_val_np = np.linalg.svd(A_np / traces_np + epsilon * np.eye(A_np.shape[-1]), compute_uv=False)
    entropy_np = -np.nansum(eig_val_np * np.log(eig_val_np + epsilon), axis=-1)
    normalized_entropy = entropy_np/math.log(A.shape[-1])
        
    return normalized_entropy

def get_entropy_3_dimensions_single_batch(key):  # 1*seq_len*model_dim                 
    R = normalize(key)
    A = cal_cov_3(R)
    A.squeeze_(0)
    Entropy1 = cal_entropy(A)
    Entropy1=Entropy1.sum(axis=0)
    return Entropy1


# 四维(单个batch) 分head
def cal_cov_4(R):
    # R: bsz, num_heads, seq, head_dim
    Z = torch.nn.functional.normalize(R, dim=-1)
    A = torch.einsum('bhji,bhjk->bhik',Z,Z) / Z.shape[-2]
    return A

def cal_entropy_no_group(A):
    # np 方法
    A = A.contiguous()
    A_np = A.cpu().numpy()
    traces_np = np.trace(A_np, axis1=-2, axis2=-1)
    traces_np = traces_np[:, np.newaxis, np.newaxis]
    
    epsilon = 1e-10  # 或更小的值，取决于您的数据精度需求
    # 对归一化的矩阵求特征值
    eig_val_np = np.linalg.svd(A_np / traces_np + epsilon * np.eye(A_np.shape[-1]), compute_uv=False)
    # 计算每个特征值的熵然后求和再除以缩放因子
    entropy_np = -np.nansum(eig_val_np * np.log(eig_val_np), axis=-1)
    normalized_entropy = entropy_np/math.log(A.shape[-1])
        
    return normalized_entropy

def get_entropy_4_dimensions_single_batch(key):                    
    R = normalize(key)
    A = cal_cov_4(R)
    A.squeeze_(0)
    Entropy1 = cal_entropy_no_group(A)
    Entropy1=Entropy1.sum(axis=0)
    # 
    return Entropy1

def get_entropy_head_4_dimensions_single_batch(key):                    
    R = normalize(key)
    A = cal_cov_4(R)
    A.squeeze_(0)
    Entropy1 = cal_entropy_no_group(A)
    return Entropy1

# 五维 
def cal_cov_5(R):
    Z = torch.nn.functional.normalize(R, dim=-1)
    A = torch.einsum('bhgji,bhgjk->bhgik',Z,Z) / Z.shape[-2]
    return A

def cal_entropy_5(A):
    # np 方法
    A = A.contiguous()
    A_np = A.cpu().numpy()
    # print(A_np.dtype)
    traces_np = np.trace(A_np, axis1=-2, axis2=-1)
    traces_np = traces_np[:,:, np.newaxis, np.newaxis]
    
    epsilon = 1e-10  # 或更小的值，取决于您的数据精度需求
    # 对归一化的矩阵求特征值
    eig_val_np = np.linalg.svd(A_np / traces_np + epsilon * np.eye(A_np.shape[-1]), compute_uv=False)
    # 计算每个特征值的熵然后求和再除以缩放因子
    entropy_np = -np.nansum(eig_val_np * np.log(eig_val_np + epsilon), axis=-1)
    normalized_entropy = entropy_np/math.log(A.shape[-1])
        
    return normalized_entropy

def cal_entropy_5_svd(A,k):
    A = A.contiguous()
    A_np = A.cpu().numpy()
    # print(A_np.dtype)
    traces_np = np.trace(A_np, axis1=-2, axis2=-1)
    traces_np = traces_np[:,:, np.newaxis, np.newaxis]
    
    epsilon = 1e-10  # 或更小的值，取决于您的数据精度需求
    # 对归一化的矩阵求特征值
    eig_val_np = np.linalg.svd(A_np / traces_np + epsilon * np.eye(A_np.shape[-1]), compute_uv=False)
    # print("eig_val_np.shape",eig_val_np.shape)
    # 计算每个特征值的熵然后求和再除以缩放因子
    entropy_np = -np.nansum(eig_val_np[:,:,:k] * np.log(eig_val_np[:,:,:k] + epsilon), axis=-1)
    normalized_entropy = entropy_np/math.log(A.shape[-1])
        
    return normalized_entropy

def get_entropy_5_dimensions_single_batch(key):                    
    R = normalize(key)
    A = cal_cov_5(R)
    A.squeeze_(0)
    Entropy1 = cal_entropy_5(A)
    Entropy1=Entropy1.sum(axis=0)
    return Entropy1

def get_entropy_head_5_dimensions_single_batch(key):                    
    R = normalize(key)
    A = cal_cov_5(R)
    A.squeeze_(0)
    Entropy1 = cal_entropy_5(A)
    return Entropy1

def get_entropy_head_5_svd_dimensions_single_batch(key,svd_num):
    R = normalize(key)
    A = cal_cov_5(R)
    A.squeeze_(0)
    Entropy1 = cal_entropy_5_svd(A,svd_num)
    return Entropy1


# new svd
def get_head_pattern_attn_entropy(attn_weights,key_states,aerfa=0.5,beta=0.65,relative=0,reletive_entropy=0):
    attn_weights = attn_weights.squeeze()[:,-1,:]
    attn_weights_std = attn_weights.std(dim=-1)

    if relative == 1:
        attn_max = attn_weights_std.max(dim=-1)[0]
        attn_min = attn_weights_std.min(dim=-1)[0]
        mid = attn_min + (attn_max - attn_min)*aerfa 
        mask = attn_weights_std >= mid 
    elif relative == 2:
        top16 = attn_weights_std.topk(16, dim=-1, largest=True)[0][-1]
        mask = attn_weights_std >= top16
    elif relative == 0:
        mask = attn_weights_std >= aerfa 
    if mask.dim() != 1:
        raise ValueError("mask.dim() != 1")
    true_positions = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if true_positions.shape[0] == 0:
        return mask

    key_new_states = key_states[:,mask,:,:].view(key_states.shape[0],true_positions.shape[0],-1,key_states.shape[-1])
    if key_new_states.shape[1] != true_positions.shape[0]:
        raise ValueError("key_new_states.shape[1] != true_positions.shape[0]")

    if reletive_entropy[0] == 0:
        entropy = get_entropy_svd_4_dimensions_single_batch(key_new_states,beta)
        entropy = torch.from_numpy(entropy).to(key_states.device)
        topn = entropy.topk(reletive_entropy[1], dim=-1, largest=True)[0][-1]
        mask_new = entropy >= topn
    elif reletive_entropy[0] == 1:
        entropy = get_entropy_head_svd_4_dimensions_single_batch(key_new_states,beta)
        # print(entropy)
        entropy = torch.from_numpy(entropy).to(key_states.device)
        return entropy
    elif reletive_entropy[0] == 2:
        entropy = get_entropy_no_group_head_svd_draw_picture(key_new_states,beta,reletive_entropy[1],reletive_entropy[2],reletive_entropy[3],reletive_entropy[4])
        entropy = torch.from_numpy(entropy).to(key_states.device)
        top16 = entropy.topk(16, dim=-1, largest=True)[0][-1]
        mask_new = entropy >= top16
    
    mask[true_positions] = mask_new
    return mask

def get_head_pattern_attn_variance(attn_weights):
    attn_weights = attn_weights.squeeze()[:,-1,:]
    attn_weights_std = attn_weights.std(dim=-1)
    
    top16 = attn_weights_std.topk(16, dim=-1, largest=True)[0][-1]
    mask = attn_weights_std >= top16
    
    if mask.dim() != 1:
        raise ValueError("mask.dim() != 1")
    
    return mask

def cal_entropy_no_group_svd(A,topk):
    A = A.contiguous()
    A_np = A.cpu().numpy().astype(np.float64)
    traces_np = np.trace(A_np, axis1=-2, axis2=-1)
    # print("traces_np.shape",traces_np.shape)
    traces_np = traces_np[:, np.newaxis, np.newaxis]
    
    epsilon = 1e-10  
    eig_val_np = np.linalg.svd(A_np / traces_np + epsilon * np.eye(A_np.shape[-1]), compute_uv=False)
    entropy_np = -np.nansum(eig_val_np[:,:topk] * np.log(eig_val_np[:,:topk]), axis=-1)
    normalized_entropy = entropy_np/math.log(A.shape[-1])
    return normalized_entropy

def cal_entropy_no_group_svd_draw_picture(A,svd_n,draw_model,layer_idx,dataname,my_type):
    A = A.contiguous()
    A_np = A.cpu().numpy().astype(np.float64)
    traces_np = np.trace(A_np, axis1=-2, axis2=-1)
    traces_np = traces_np[:, np.newaxis, np.newaxis]
    
    epsilon = 1e-10  
    eig_val_np = np.linalg.svd(A_np / traces_np + epsilon * np.eye(A_np.shape[-1]), compute_uv=False)
    entropy_np = -np.nansum(eig_val_np[:,:svd_n] * np.log(eig_val_np[:,:svd_n]), axis=-1)
    normalized_entropy = entropy_np / math.log(A.shape[-1])
    
    if draw_model == "draw_svd_trend":    
        datapath = os.path.join(os.getcwd(), 'draw_picture', 'svdn_trend')
        datapath = datapath + f"/llama2-13B/{my_type}/csv/" + dataname + "_eigenvalues.csv"
        with open(datapath, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            
            for i in range(40):
                row = [f"{x:.20f}" for x in eig_val_np[i, :]]
                csv_writer.writerow(row)
    elif draw_model == "draw_thermodynamic_chart":
        entropy_np = -np.nansum(eig_val_np[:,:svd_n] * np.log(eig_val_np[:,:svd_n]), axis=-1)
        normalized_entropy = entropy_np/math.log(A.shape[-1])
        
        datapath = os.path.join(os.getcwd(), 'draw_picture', 'thermodynamic_chart')
        datapath = datapath + f"/llama2-13B/{my_type}/csv/" 
        os.makedirs(datapath, exist_ok=True)
        datapath = datapath+ dataname + "_eigenvalues.csv"
        mode = 'a' if layer_idx > 0 else 'w'
        with open(datapath, mode, newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            
            row = [f"{x:.20f}" for x in normalized_entropy]
            csv_writer.writerow(row)
    elif draw_model == "draw_accumulated_energy":
        if layer_idx == 0:
            dim = (-2,-1)
            raw_matrix = A_np / traces_np + epsilon * np.eye(A_np.shape[-1])
            datapath = os.path.join(os.getcwd(), 'draw_picture', 'accumulated_energy')
            datapath = datapath + "/" + dataname + "_eigenvalues.csv"
            with open(datapath, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                
                for i in range(raw_matrix.shape[0]):
                    U, S, V = np.linalg.svd(raw_matrix[i],full_matrices=False)
                    cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
                    csv_writer.writerow(cumulative_energy)        
    
    return normalized_entropy

def get_entropy_svd_4_dimensions_single_batch(key,topk):   
    R = normalize(key)
    A = cal_cov_4(R)
    A.squeeze_(0)
    Entropy1 = cal_entropy_no_group_svd(A,topk)
    # Entropy1=Entropy1.sum(axis=0)
    return Entropy1

def get_entropy_head_svd_4_dimensions_single_batch(key,topk):    
    R = normalize(key)
    A = cal_cov_4(R)
    A.squeeze_(0)
    Entropy1 = cal_entropy_no_group_svd(A,topk)
    return Entropy1

def get_entropy_no_group_head_svd_draw_picture(key,svd_n,draw_model,layer_idx,dataname,my_type):    
    R = normalize(key)
    A = cal_cov_4(R)
    A.squeeze_(0)
    Entropy1 = cal_entropy_no_group_svd_draw_picture(A,svd_n,draw_model,layer_idx,dataname,my_type)
    return Entropy1

