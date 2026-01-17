# basicts/metrics/kge.py
import torch
import numpy as np
def kge(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """计算Kling-Gupta Efficiency (KGE)指标
    
    Args:
        prediction: 预测值张量
        target: 真实值张量
        
    Returns:
        KGE值
    """
    # 计算相关系数(r)
    mean_pred = torch.mean(prediction)
    mean_target = torch.mean(target)
    
    numerator = torch.sum((prediction - mean_pred) * (target - mean_target))
    denominator = torch.sqrt(torch.sum((prediction - mean_pred) ** 2) * torch.sum((target - mean_target) ** 2))
    r = numerator / denominator
    
    # 计算相对偏差(α)
    alpha = torch.std(prediction) / torch.std(target)
    
    # 计算相对变异(β)
    beta = mean_pred / mean_target
    
    # 计算KGE
    kge_value = 1 - torch.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return kge_value

def masked_kge(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None, null_val: float = np.nan) -> torch.Tensor:
    if mask is None:
        return kge(prediction, target)
    # 应用掩码
    prediction = prediction * mask
    target = target * mask
    # KGE计算
    r = torch.corrcoef(torch.stack([prediction.flatten(), target.flatten()]))[0, 1]
    alpha = torch.mean(prediction) / torch.mean(target)
    beta = torch.std(prediction) / torch.std(target)
    kge_value = 1 - torch.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return kge_value


