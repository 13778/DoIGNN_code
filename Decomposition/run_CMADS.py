import os
import pickle
import numpy as np

from GLDTR import GLDTR3D   

def forecast_X_ar1(X_train: np.ndarray, steps: int, eps: float = 1e-8):
    """
    X_train: [R, T_train]
    steps: number of future steps to forecast
    Return:
      X_future: [R, steps]
      phi: [R]
    """
    R, T_train = X_train.shape
    if T_train < 2:
        raise ValueError("X_train too short for AR(1).")

    x_prev = X_train[:, :-1]  
    x_next = X_train[:, 1:]   

    denom = (x_prev * x_prev).sum(axis=1) + eps
    phi = (x_next * x_prev).sum(axis=1) / denom  

    X_future = np.zeros((R, steps), dtype=np.float32)
    cur = X_train[:, -1].astype(np.float32)  

    for i in range(steps):
        cur = (phi * cur).astype(np.float32)
        X_future[:, i] = cur

    return X_future, phi

def make_AX_no_leak(
    raw_data_path: str,
    index_pkl_path: str,
    ax_save_path: str,
    target_channel: int,
    history_seq_len: int,
    future_seq_len: int,
    rank: int = 16,
    hor_batch_size: int = 256,
    ver_batch_size: int = 10,
    lr: float = 1e-3,
):
    
    raw_data = np.load(raw_data_path)  
    L, N, _ = raw_data.shape

    
    with open(index_pkl_path, "rb") as f:
        index = pickle.load(f)
    train_index = index["train"]

    train_cut = train_index[-1][1]          
    end_index = int(train_cut - 1)          
    if end_index < 1:
        raise ValueError("end_index too small.")

    print(f"[no-leak] L={L}, N={N}, train_cut(t)={train_cut}, end_index={end_index}")

    
    
    
    all_channels = list(range(raw_data.shape[2]))
    other_channels = [c for c in all_channels if c != target_channel]
    Y_tnc = raw_data[:, :, other_channels].astype(np.float32)  

    
    DG = GLDTR3D(
        Y_tnc,
        ver_batch_size=ver_batch_size,
        hor_batch_size=hor_batch_size,
        rank=rank,
        lr=lr,
        end_index=end_index,
        val_len=0,            
        normalize=True,       
    )

    X, A, B = DG.train_GLDTR(num_epochs=50)   

    
    
    
    X_train = X[:, :end_index+1]             
    future_steps = L - (end_index + 1)
    if future_steps < 0:
        raise ValueError("future_steps < 0, check L/end_index.")

    X_future, phi = forecast_X_ar1(X_train, steps=future_steps)
    X_full = np.concatenate([X_train, X_future], axis=1)  
    assert X_full.shape[1] == L

    
    AX = A @ X_full
    assert AX.shape == (N, L)

    os.makedirs(os.path.dirname(ax_save_path), exist_ok=True)
    np.save(ax_save_path, AX.astype(np.float32))
    np.save(ax_save_path.replace(".npy", "_phi.npy"), phi.astype(np.float32))

    print(f"[no-leak] saved AX: {ax_save_path}, shape={AX.shape}")
    print(f"[no-leak] saved phi: {ax_save_path.replace('.npy','_phi.npy')}, shape={phi.shape}")

if __name__ == "__main__":
    make_AX_no_leak(
        raw_data_path="CMADS_1979_2018_(14610_130_9).npy",
        index_pkl_path="datasets/CMADS_AVE_TEM/index_in12_out12.pkl",
        ax_save_path="Decomposition/AX_noleak.npy",
        target_channel=1,
        history_seq_len=12,
        future_seq_len=12,
        rank=16,
        hor_batch_size=256,
        ver_batch_size=10,
        lr=1e-3,
    )
