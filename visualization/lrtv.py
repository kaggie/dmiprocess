import torch
import torch.nn.functional # For avg_pool3d
from config.settings import get_device # Ensure device configuration is used

def lrtv_denoise(tensor, rank=10, tv_weight=0.1):
    """
    Low-rank + TV approximation.
    tensor: torch.Tensor of shape (X, Y, Z, F) already on the target device.
    """
    if tensor.ndim != 4:
        raise ValueError(f"Input tensor must be 4D (X, Y, Z, F), got {tensor.ndim}D")
    
    X, Y, Z, F_dim = tensor.shape # Renamed F to F_dim to avoid conflict
    # Device is determined by the input tensor's device property
    # device = tensor.device 
    
    reshaped = tensor.view(-1, F_dim)  # (N_voxels, F)

    # Low-rank approximation via SVD
    try:
        U, S_val, Vh = torch.linalg.svd(reshaped, full_matrices=False)
    except Exception as e: 
        # Fallback to CPU if SVD on current device fails (e.g. some CUDA SVD issues)
        print(f"Warning: SVD failed on device {tensor.device} ({e}). Trying SVD on CPU.")
        try:
            U, S_val, Vh = torch.linalg.svd(reshaped.cpu(), full_matrices=False)
            U, S_val, Vh = U.to(tensor.device), S_val.to(tensor.device), Vh.to(tensor.device) # Move back to original device
        except Exception as e_cpu:
            print(f"SVD on CPU also failed ({e_cpu}). Returning original tensor.")
            return tensor 
    
    actual_rank = min(rank, S_val.shape[0])
    if rank > S_val.shape[0]:
        print(f"Warning: Requested rank {rank} > available singular values {S_val.shape[0]}. Using rank {actual_rank}.")

    S_low_rank = S_val.clone()
    S_low_rank[actual_rank:] = 0
    
    if S_low_rank.numel() == 0 or U.shape[1] == 0 or Vh.shape[0] == 0 or \
       U.shape[1] < actual_rank or S_low_rank.shape[0] < actual_rank or Vh.shape[0] < actual_rank:
         print(f"Warning: Low rank component cannot be formed due to SVD output dimensions or rank. U:{U.shape}, S:{S_low_rank.shape}, Vh:{Vh.shape}, actual_rank:{actual_rank}. Returning original tensor.")
         if actual_rank == 0:
             low_rank_reshaped = torch.zeros_like(reshaped)
         else:
            low_rank_reshaped = reshaped.clone() 
    else:
         diag_S_low_rank = torch.diag(S_low_rank[:actual_rank])
         low_rank_reshaped = (U[:, :actual_rank] @ diag_S_low_rank @ Vh[:actual_rank, :])
            
    low_rank = low_rank_reshaped.view(X, Y, Z, F_dim)

    if X > 0 and Y > 0 and Z > 0 and tv_weight > 0: 
        smoothed = torch.nn.functional.avg_pool3d(
            low_rank.permute(3, 0, 1, 2).unsqueeze(0), # (1, F_dim, X, Y, Z)
            kernel_size=3, stride=1, padding=1
        ).squeeze(0).permute(1, 2, 3, 0) # (X, Y, Z, F_dim)
        final_result = (1 - tv_weight) * low_rank + tv_weight * smoothed
    else: 
        if tv_weight > 0: 
            print("Warning: One or more spatial dimensions are zero or tv_weight is zero. TV smoothing cannot be applied. Returning low-rank component only.")
        final_result = low_rank 
    return final_result
