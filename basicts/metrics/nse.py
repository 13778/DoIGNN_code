# basicts/metrics/nse.py
import torch
import numpy as np
def nse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """计算Nash-Sutcliffe Efficiency (NSE)指标
    
    Args:
        prediction: 预测值张量
        target: 真实值张量
        
    Returns:
        NSE值
    """
    numerator = torch.sum((target - prediction) ** 2)
    denominator = torch.sum((target - torch.mean(target)) ** 2)
    return 1 - numerator / denominator

def masked_nse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None, null_val: float = np.nan) -> torch.Tensor:
    if mask is None:
        return nse(prediction, target)
    # 应用掩码
    prediction = prediction * mask
    target = target * mask
    numerator = torch.sum((target - prediction) ** 2)
    denominator = torch.sum((target - torch.mean(target)) ** 2)
    return 1 - numerator / denominator

