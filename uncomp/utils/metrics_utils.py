import torch
from scipy.stats import pearsonr

def min_max_pearsonr(x_list, y_list):
    # 将输入的列表转换为 PyTorch tensors
    x = torch.tensor(x_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32)
    
    # 使用 Min-Max 标准化
    x_normalized = (x - x.min()) / (x.max() - x.min())
    y_normalized = (y - y.min()) / (y.max() - y.min())

    # 将 PyTorch tensors 转换为 NumPy 数组
    x_normalized_np = x_normalized.numpy()
    y_normalized_np = y_normalized.numpy()

    # 使用 SciPy 计算皮尔逊相关系数
    correlation, p_value = pearsonr(x_normalized_np, y_normalized_np)
    
    # 返回相关系数和 p 值
    return correlation, p_value

def z_score_pearsonr(x_list, y_list):
    # 将输入的列表转换为 PyTorch tensors
    x = torch.tensor(x_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32)
    
    # 使用 Z-score 标准化
    x_normalized = (x - x.mean()) / x.std()
    y_normalized = (y - y.mean()) / y.std()

    # 将 PyTorch tensors 转换为 NumPy 数组
    x_normalized_np = x_normalized.numpy()
    y_normalized_np = y_normalized.numpy()

    # 使用 SciPy 计算皮尔逊相关系数
    correlation, p_value = pearsonr(x_normalized_np, y_normalized_np)
    
    # 返回相关系数和 p 值
    return correlation, p_value