from __future__ import print_function
import torch
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from TCN import *
import pickle
from data_loader import data_loader_3d

use_cuda = True
"""
    Refference Code: https://github.com/rajatsen91/deepglo/blob/master/DeepGLO/DeepGLO.py
"""
class GLDTR(object):
    def __init__(
        self,
        MTS,
        ver_batch_size=1,
        hor_batch_size=256,
        TCN_channels=[32, 32, 32, 32, 1],
        kernel_size=7,
        dropout=0.1,
        rank=64,
        lr=0.0005,
        val_len=1000,
        end_index=4999,
        normalize=True,
    ):
        # Initialize class variables
        self.dropout = dropout
        self.TCN = TemporalConvNet(
            num_inputs=1,
            num_channels=TCN_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            init=True,
        )

        # Normalize the input data if required
        if normalize:
            self.s = np.std(MTS[:, 0:end_index], axis=1) + 1.0
            self.m = np.mean(MTS[:, 0:end_index], axis=1)
            self.MTS = (MTS - self.m[:, None]) / self.s[:, None]
            self.mini = np.abs(np.min(self.MTS))
            self.MTS = self.MTS + self.mini 

        else:
            self.MTS = MTS

        self.normalize = normalize
        n, T = self.MTS.shape
        t0 = end_index + 1

        if t0 > T:
            self.MTS = np.hstack([self.MTS, self.MTS[:, -1].reshape(-1, 1)])

        # Initialize factor matrices

        self.global_matrix = torch.normal(torch.zeros(rank, t0).float(), 0.1).float()
        self.scores = torch.normal(torch.zeros(n, rank).float(), 0.1).float()
                
        self.ver_batch_size = ver_batch_size
        self.hor_batch_size = hor_batch_size
        self.TCN_channels = TCN_channels
        self.rank = rank
        self.kernel_size = kernel_size
        self.lr = lr
        self.val_len = val_len
        self.end_index = end_index
        self.Dataloader = data_loader(
            MTS=self.MTS,
            ver_batch_size=ver_batch_size,
            hor_batch_size=hor_batch_size,
            end_index=end_index,
            val_len=val_len,
            shuffle=False,
        )

    def tensor2d_to_temporal(self, T):
        """
        Convert a 2D tensor to a temporal tensor
        """
        T = T.view(1, T.size(0), T.size(1))
        T = T.transpose(0, 1)
        return T

    def temporal_to_tensor2d(self, T):
        """
        Convert a temporal tensor to a 2D tensor
        """
        T = T.view(T.size(0), T.size(2))
        return T

    def step_global_matrix_loss(self, inp, out, last_vindex, last_hindex, reg=0.2):
        """
        Compute and update the loss for factor X
        """
        global_matrix_out = self.global_matrix[:, last_hindex + 1 : last_hindex + 1 + out.size(2)]
        scores_out = self.scores[self.Dataloader.I[last_vindex : last_vindex + out.size(0)], :]
        
        if use_cuda:
            global_matrix_out = global_matrix_out.cuda()
            scores_out = scores_out.cuda()

        global_matrix_out = Variable(global_matrix_out, requires_grad=True)
        out = self.temporal_to_tensor2d(out)
        optim_X = optim.Adam(params=[global_matrix_out], lr=self.lr)
        Hout = torch.matmul(scores_out, global_matrix_out)
        optim_X.zero_grad()
        loss = torch.mean(torch.pow(Hout - out.detach(), 2))
        l2 = torch.mean(torch.pow(global_matrix_out, 2))
        r = loss.detach() / l2.detach()
        loss = loss + r * reg * l2
        loss.backward()
        optim_X.step()
        self.global_matrix[:, last_hindex + 1 : last_hindex + 1 + inp.size(2)] = global_matrix_out.cpu().detach()
        return loss

    def step_scores_loss(self, inp, out, last_vindex, last_hindex, reg=0.2):
        """
        Compute and update the loss for factor F
        """
        global_matrix_out = self.global_matrix[:, last_hindex + 1 : last_hindex + 1 + out.size(2)]
        scores_out = self.scores[self.Dataloader.I[last_vindex : last_vindex + out.size(0)], :]
        
        if use_cuda:
            global_matrix_out = global_matrix_out.cuda()
            scores_out = scores_out.cuda()

        scores_out = Variable(scores_out, requires_grad=True)
        optim_F = optim.Adam(params=[scores_out], lr=self.lr)
        out = self.temporal_to_tensor2d(out)
        Hout = torch.matmul(scores_out, global_matrix_out)
        optim_F.zero_grad()
        loss = torch.mean(torch.pow(Hout - out.detach(), 2))
        l2 = torch.mean(torch.pow(scores_out, 2))
        r = loss.detach() / l2.detach()
        loss = loss + r * reg * l2
        loss.backward()
        optim_F.step()
        self.scores[self.Dataloader.I[last_vindex : last_vindex + inp.size(0)], :] = scores_out.cpu().detach()
        return loss

    def step_temporal_loss_global_matrix(self, inp, last_vindex, last_hindex, lam=0.2):
        """
        Compute and update the temporal loss for factor X
        """
        Xin = self.global_matrix[:, last_hindex : last_hindex + inp.size(2)]
        global_matrix_out = self.global_matrix[:, last_hindex + 1 : last_hindex + 1 + inp.size(2)]
        
        for p in self.TCN.parameters():
            p.requires_grad = False
        
        if use_cuda:
            Xin = Xin.cuda()
            global_matrix_out = global_matrix_out.cuda()

        Xin = Variable(Xin, requires_grad=True)
        global_matrix_out = Variable(global_matrix_out, requires_grad=True)
        optim_out = optim.Adam(params=[global_matrix_out], lr=self.lr)
        Xin = self.tensor2d_to_temporal(Xin)
        global_matrix_out = self.tensor2d_to_temporal(global_matrix_out)
        hatX = self.TCN(Xin)
        optim_out.zero_grad()
        loss = lam * torch.mean(torch.pow(global_matrix_out - hatX.detach(), 2))
        loss.backward()
        optim_out.step()
        temp = self.temporal_to_tensor2d(global_matrix_out.detach())
        self.global_matrix[:, last_hindex + 1 : last_hindex + 1 + inp.size(2)] = temp
        return loss

    def predict_future_batch(self, model, inp, future=10, cpu=True):
        """
        Predict future values for a batch of input sequences
        """
        if cpu:
            model = model.cpu()
            inp = inp.cpu()
        else:
            inp = inp.cuda()

        out = model(inp)
        output = out[:, :, out.size(2) - 1].view(out.size(0), out.size(1), 1)
        out = torch.cat((inp, output), dim=2)
        torch.cuda.empty_cache()
        
        for i in range(future - 1):
            inp = out
            out = model(inp)
            output = out[:, :, out.size(2) - 1].view(out.size(0), out.size(1), 1)
            out = torch.cat((inp, output), dim=2)
            torch.cuda.empty_cache()

        out = self.temporal_to_tensor2d(out)
        out = np.array(out.cpu().detach())
        return out

    def predict_future(self, model, inp, future=10, cpu=True, bsize=90):
        """
        Predict future values for the entire input sequence
        """
        n = inp.size(0)
        inp = inp.cpu()
        ids = np.arange(0, n, bsize)
        ids = list(ids) + [n]
        out = self.predict_future_batch(model, inp[ids[0] : ids[1], :, :], future, cpu)
        torch.cuda.empty_cache()

        for i in range(1, len(ids) - 1):
            temp = self.predict_future_batch(
                model, inp[ids[i] : ids[i + 1], :, :], future, cpu
            )
            torch.cuda.empty_cache()
            out = np.vstack([out, temp])

        out = torch.from_numpy(out).float()
        return self.tensor2d_to_temporal(out)

    def predict_global(
        self, ind, last_step=100, future=10, cpu=False, normalize=False, bsize=90
    ):
        """
        Predict global future values for given indices
        """
        if ind is None:
            ind = np.arange(self.MTS.shape[0])
        
        if cpu:
            self.TCN = self.TCN.cpu()
        
        self.TCN = self.TCN.eval()
        rg = 1 + 2 * (self.kernel_size - 1) * 2 ** (len(self.TCN_channels) - 1)
        X = self.global_matrix[:, last_step - rg : last_step]
        X = self.tensor2d_to_temporal(X)
        
        outX = self.predict_future(
            model=self.TCN, inp=X, future=future, cpu=cpu, bsize=bsize
        )
        outX = self.temporal_to_tensor2d(outX)
        
        F = self.scores
        Y = torch.matmul(F, outX)
        Y = np.array(Y[ind, :].cpu().detach())
        
        self.TCN = self.TCN.cuda()
        del F
        torch.cuda.empty_cache()
        
        for p in self.TCN.parameters():
            p.requires_grad = True
        
        if normalize:
            Y = Y - self.mini
            Y = Y * self.s[ind, None] + self.m[ind, None]
        
        return Y

    def train_TCN(self, MTS, num_epochs=20, early_stop=False, tenacity=3):
        """
        Train the TCN model
        """
        seq = self.TCN
        num_channels = self.TCN_channels
        kernel_size = self.kernel_size
        ver_batch_size = min(self.ver_batch_size, MTS.shape[0] / 2)

        for p in seq.parameters():
            p.requires_grad = True

        TC = TCN(
            MTS=MTS,
            num_inputs=1,
            num_channels=num_channels,
            kernel_size=kernel_size,
            ver_batch_size=ver_batch_size,
            hor_batch_size=self.hor_batch_size,
            normalize=False,
            end_index=self.end_index - self.val_len,
            val_len=self.val_len,
            lr=self.lr,
            num_epochs=num_epochs,
        )

        TC.train_model(early_stop=early_stop, tenacity=tenacity)
        self.TCN = TC.seq

    def train_factors(
        self,
        reg_X=0.2,
        reg_F=0.2,
        mod=5,
        early_stop=False,
        tenacity=3,
        ind=None,
        seed=False,
    ):
        """
        Train the factor matrices X and F
        """
        self.Dataloader.epoch = 0
        self.Dataloader.vindex = 0
        self.Dataloader.hindex = 0
        
        if use_cuda:
            self.TCN = self.TCN.cuda()
        
        for p in self.TCN.parameters():
            p.requires_grad = True

        l_F = [0.0]
        l_X = [0.0]
        l_X_temporal = [0.0]
        iter_count = 0
        vae = float("inf")
        scount = 0
        global_matrix_best = self.global_matrix.clone()
        scores_best = self.scores.clone()
        
        while self.Dataloader.epoch < self.num_epochs:
            last_epoch = self.Dataloader.epoch
            last_vindex = self.Dataloader.vindex
            last_hindex = self.Dataloader.hindex
            
            inp, out, _, _ = self.Dataloader.next_batch(option=1)
            
            if use_cuda:
                inp = inp.float().cuda()
                out = out.float().cuda()
            
            if iter_count % mod >= 0:
                l1 = self.step_scores_loss(inp, out, last_vindex, last_hindex, reg=reg_F)
                l_F = l_F + [l1.cpu().item()]
            
            if iter_count % mod >= 0:
                l1 = self.step_global_matrix_loss(inp, out, last_vindex, last_hindex, reg=reg_X)
                l_X = l_X + [l1.cpu().item()]
            
            if not seed and iter_count % mod == 1:
                l2 = self.step_temporal_loss_global_matrix(inp, last_vindex, last_hindex)
                l_X_temporal = l_X_temporal + [l2.cpu().item()]
            
            iter_count += 1

            if self.Dataloader.epoch > last_epoch:
                print("Entering Epoch# ", self.Dataloader.epoch)
                print("Factorization Loss F: ", np.mean(l_F))
                print("Factorization Loss X: ", np.mean(l_X))
                print("Temporal Loss X: ", np.mean(l_X_temporal))
                
                if ind is None:
                    ind = np.arange(self.MTS.shape[0])
                
                inp = self.predict_global(
                    ind,
                    last_step=self.end_index - self.val_len,
                    future=self.val_len,
                    cpu=False,
                )
                
                R = self.MTS[ind, self.end_index - self.val_len : self.end_index]
                S = inp[:, -self.val_len : :]
                ve = np.abs(R - S).mean() / np.abs(R).mean()
                print("Validation Loss (Global): ", ve)
                
                if ve <= vae:
                    vae = ve
                    scount = 0
                    global_matrix_best = self.global_matrix.clone()
                    scores_best = self.scores.clone()
                    TCNbest = pickle.loads(pickle.dumps(self.TCN))
                else:
                    scount += 1
                    if scount > tenacity and early_stop:
                        print("Early Stopped")
                        self.global_matrix = global_matrix_best
                        self.scores = scores_best
                        self.TCN = TCNbest
                        if use_cuda:
                            self.TCN = self.TCN.cuda()
                        break
        
        return self.global_matrix, self.scores

    def train_GLDTR(
        self, init_epochs=100, alt_iters=10, tenacity=7, mod=5
    ):
        """
        Train all models using alternating training strategy
        """
        print("Initializing Factors.....")
        self.num_epochs = init_epochs
        self.train_factors()

        if alt_iters % 2 == 1:
            alt_iters += 1

        print("Starting Alternate Training.....")

        for i in range(1, alt_iters):
            if i % 2 == 0:
                print(
                    "--------------------------------------------Training global martix and scores. Iter#: "
                    + str(i)
                    + "-------------------------------------------------------"
                )
                self.num_epochs = 100
                X, F = self.train_factors(
                    seed=False, early_stop=True, tenacity=tenacity, mod=mod
                )
            else:
                print(
                    "--------------------------------------------Training TCN. Iter#: "
                    + str(i)
                    + "-------------------------------------------------------"
                )
                self.num_epochs = 100
                T = np.array(self.global_matrix.cpu().detach())
                self.train_TCN(
                    MTS=T,
                    num_epochs=self.num_epochs,
                    early_stop=True,
                    tenacity=tenacity,
                )

        return X, F


class GLDTR3D(object):
    """
    Tensor (N,C,T) decomposition:
      Y[n,c,t] ≈ Σ_r A[n,r] * B[c,r] * X[r,t]
    A: [N,R], B:[C,R], X:[R,T]
    """

    def __init__(
        self,
        Y_tnc,                          # [T,N,C]
        ver_batch_size=8,
        hor_batch_size=256,
        TCN_channels=[32, 32, 32, 1],
        kernel_size=7,
        dropout=0.1,
        rank=32,
        lr=5e-4,
        val_len=1461,
        end_index=11687,
        normalize=True,
    ):
        self.dropout = dropout
        self.rank = rank
        self.lr = lr
        self.val_len = val_len
        self.end_index = end_index
        self.normalize = normalize

        # ---- reshape to [N,C,T]
        assert Y_tnc.ndim == 3, "Input must be [T,N,C]"
        T, N, C = Y_tnc.shape
        Y_nct = np.transpose(Y_tnc, (1, 2, 0)).astype(np.float32)  # [N,C,T]

        # ---- normalize per (N,C) series along time
        if normalize:
            s = np.std(Y_nct[:, :, 0:end_index], axis=2) + 1.0      # [N,C]
            m = np.mean(Y_nct[:, :, 0:end_index], axis=2)           # [N,C]
            Y_nct = (Y_nct - m[:, :, None]) / s[:, :, None]
            mini = 0.0


            self.s = s
            self.m = m
            self.mini = 0.0
        else:
            self.s = None
            self.m = None
            self.mini = 0.0

        # ---- ensure X can be indexed at end_index+1
        self.Y = Y_nct  # [N,C,T]
        self.N, self.C, self.T = self.Y.shape
        if end_index + 1 >= self.T:
            pad = self.Y[:, :, -1:]
            self.Y = np.concatenate([self.Y, pad], axis=2)
            self.T = self.Y.shape[2]

        # ---- dataloader
        self.Dataloader = data_loader_3d(
            Y_nct=self.Y,
            ver_batch_size=ver_batch_size,
            hor_batch_size=hor_batch_size,
            end_index=end_index,
            val_len=val_len,
            shuffle=False,
        )

        # ---- TCN (for X temporal regularization)
        self.TCN = TemporalConvNet(
            num_inputs=1,
            num_channels=TCN_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            init=True,
        )

        # ---- init factors (store on CPU like your original code)
        self.A = torch.randn(self.N, self.rank) * 0.01   # [N,R]
        self.B = torch.randn(self.C, self.rank) * 0.01   # [C,R]
        self.global_matrix = torch.randn(self.rank, self.T) * 0.01  # X: [R,T]

    # ---------- utils (same spirit as your original code)
    def tensor2d_to_temporal(self, T2d):
        T = T2d.view(1, T2d.size(0), T2d.size(1))
        T = T.transpose(0, 1)   # [R,1,T]
        return T

    def temporal_to_tensor2d(self, T3d):
        return T3d.view(T3d.size(0), T3d.size(2))

    # ---------- core steps
    def step_X_loss(self, out, last_vindex, last_hindex, reg=0.2):
        """
        Update X slice: X[:, t+1:t+1+Tw]
        out: [nb,C,Tw]
        """
        Tw = out.size(2)
        v0 = last_vindex
        v1 = min(self.N, v0 + out.size(0))
        t0 = last_hindex + 1
        t1 = t0 + Tw

        A_batch = self.A[v0:v1, :]                    # [nb,R]
        B = self.B                                   # [C,R]
        X_out = self.global_matrix[:, t0:t1]          # [R,Tw]

        if use_cuda:
            A_batch = A_batch.cuda()
            B = B.cuda()
            X_out = X_out.cuda()
            out = out.cuda()

        X_out = Variable(X_out, requires_grad=True)
        opt = optim.Adam(params=[X_out], lr=self.lr)
        opt.zero_grad()

        Y_hat = torch.einsum('nr,cr,rt->nct', A_batch, B, X_out)  # [nb,C,Tw]
        loss = torch.mean((Y_hat - out.detach()) ** 2)

        l2 = torch.mean(X_out ** 2)
        r = loss.detach() / (l2.detach() + 1e-12)
        loss = loss + r * reg * l2

        loss.backward()
        opt.step()

        # write back to CPU tensor
        self.global_matrix[:, t0:t1] = X_out.detach().cpu()
        return loss.detach().cpu()

    def step_AB_loss(self, out, last_vindex, last_hindex, reg=0.2):
        """
        Update A slice and B (global) with X fixed.
        out: [nb,C,Tw]
        """
        Tw = out.size(2)
        v0 = last_vindex
        v1 = min(self.N, v0 + out.size(0))
        t0 = last_hindex + 1
        t1 = t0 + Tw

        A_out = self.A[v0:v1, :]                      # [nb,R]
        B_out = self.B                                # [C,R]
        X_fixed = self.global_matrix[:, t0:t1]         # [R,Tw]

        if use_cuda:
            A_out = A_out.cuda()
            B_out = B_out.cuda()
            X_fixed = X_fixed.cuda()
            out = out.cuda()

        A_out = Variable(A_out, requires_grad=True)
        B_out = Variable(B_out, requires_grad=True)

        opt = optim.Adam(params=[A_out, B_out], lr=self.lr)
        opt.zero_grad()

        Y_hat = torch.einsum('nr,cr,rt->nct', A_out, B_out, X_fixed.detach())
        loss = torch.mean((Y_hat - out.detach()) ** 2)

        l2 = torch.mean(A_out ** 2) + torch.mean(B_out ** 2)
        r = loss.detach() / (l2.detach() + 1e-12)
        loss = loss + r * reg * l2

        loss.backward()
        opt.step()

        # write back
        self.A[v0:v1, :] = A_out.detach().cpu()
        self.B = B_out.detach().cpu()
        return loss.detach().cpu()

    def step_temporal_loss_X(self, Tw, last_hindex, lam=0.2):
        """
        Temporal regularization on X (same as your original idea):
          minimize || X_{t+1:t+Tw} - TCN(X_{t:t+Tw-1}) ||^2
        """
        Xin = self.global_matrix[:, last_hindex:last_hindex + Tw]          # [R,Tw]
        Xout = self.global_matrix[:, last_hindex + 1:last_hindex + 1 + Tw] # [R,Tw]

        for p in self.TCN.parameters():
            p.requires_grad = False

        if use_cuda:
            self.TCN = self.TCN.cuda()
            Xin = Xin.cuda()
            Xout = Xout.cuda()

        Xin = Variable(Xin, requires_grad=True)
        Xout = Variable(Xout, requires_grad=True)

        opt = optim.Adam(params=[Xout], lr=self.lr)
        opt.zero_grad()

        Xin_t = self.tensor2d_to_temporal(Xin)
        Xout_t = self.tensor2d_to_temporal(Xout)
        hat = self.TCN(Xin_t)

        loss = lam * torch.mean((Xout_t - hat.detach()) ** 2)
        loss.backward()
        opt.step()

        self.global_matrix[:, last_hindex + 1:last_hindex + 1 + Tw] = self.temporal_to_tensor2d(Xout_t.detach()).cpu()
        return loss.detach().cpu()

    # ---------- helper: reconstruct known window (no future)
    def reconstruct_window(self, n_idx, t_start, length, denorm=True):
        """
        Return: [length, nb, C] in original scale if denorm=True else normalized scale
        """
        nb = len(n_idx)
        t0 = t_start
        t1 = t_start + length
        A = self.A[n_idx, :]                     # [nb,R]
        B = self.B                               # [C,R]
        X = self.global_matrix[:, t0:t1]         # [R,length]
        Y = torch.einsum('nr,cr,rt->nct', A, B, X)  # [nb,C,length]
        Y = Y.permute(2, 0, 1).detach().cpu().numpy()  # [length,nb,C]

        if self.normalize and denorm:
            # inverse normalize per (N,C)
            Y = Y - self.mini
            s = self.s[n_idx, :]                  # [nb,C]
            m = self.m[n_idx, :]                  # [nb,C]
            Y = Y * s[None, :, :] + m[None, :, :]

        return Y

    def train_factors(
        self,
        num_epochs=50,
        reg_X=0.05,
        reg_AB=0.05,
        lam_temporal=0.1,
        mod=5,
        print_every=10,
        ema_beta=0.9,
        log_sanity=True,
    ):
        """
        GLDTR3D training loop with stable loss logging.

        Updates per iteration:
        1) Update A,B (with X fixed) via step_AB_loss
        2) Update X     (with A,B fixed) via step_X_loss
        3) (Every mod iters) temporal regularization on X via step_temporal_loss_X

        Notes:
        - loss_TCN printing shows the *most recent* computed TCN loss (not -1 due to print cadence).
        - EMA smoothing is used for readability.
        - Optional sanity stats print out/out_hat mean/std to diagnose scaling issues.
        """
        import time
        import numpy as np
        import torch

        # reset loader state
        self.Dataloader.epoch = 0
        self.Dataloader.vindex = 0
        self.Dataloader.hindex = 0

        # put TCN on GPU if needed
        if use_cuda:
            self.TCN = self.TCN.cuda()

        it = 0
        t0 = time.time()

        # EMA for smoother logs
        ema_ab, ema_x, ema_tcn = None, None, None

        # show latest computed TCN loss (independent from print cadence)
        last_tcn_print = None

        # for sanity: keep last stats
        last_out_mean = last_out_std = None
        last_hat_mean = last_hat_std = None

        while self.Dataloader.epoch < num_epochs:
            last_v = self.Dataloader.vindex
            last_h = self.Dataloader.hindex

            # fetch batch
            inp, out, _, _ = self.Dataloader.next_batch(option=1)  # out: [nb, C, Tw]
            Tw = int(out.size(2))

            # ---------- (1) update A,B
            loss_ab = self.step_AB_loss(out, last_v, last_h, reg=reg_AB)
            loss_ab_val = float(loss_ab.item() if hasattr(loss_ab, "item") else loss_ab)

            # ---------- (2) update X
            loss_x = self.step_X_loss(out, last_v, last_h, reg=reg_X)
            loss_x_val = float(loss_x.item() if hasattr(loss_x, "item") else loss_x)

            # ---------- (3) temporal regularization on X (optional)
            loss_tcn_val = None
            if mod is not None and mod > 0 and (it % mod) == 1:
                loss_tcn = self.step_temporal_loss_X(Tw=Tw, last_hindex=last_h, lam=lam_temporal)
                loss_tcn_val = float(loss_tcn.item() if hasattr(loss_tcn, "item") else loss_tcn)
                last_tcn_print = loss_tcn_val

            # ---------- EMA updates
            if ema_ab is None:
                ema_ab = loss_ab_val
                ema_x = loss_x_val
                # init ema_tcn with first observed or 0.0
                ema_tcn = (loss_tcn_val if loss_tcn_val is not None else 0.0)
            else:
                ema_ab = ema_beta * ema_ab + (1 - ema_beta) * loss_ab_val
                ema_x = ema_beta * ema_x + (1 - ema_beta) * loss_x_val
                if loss_tcn_val is not None:
                    ema_tcn = ema_beta * ema_tcn + (1 - ema_beta) * loss_tcn_val

            # ---------- Optional sanity stats (diagnose scaling/mismatch)
            if log_sanity and ((it % print_every) == 0):
                with torch.no_grad():
                    # build Y_hat for the same batch/window using CURRENT factors
                    nb = out.size(0)
                    v0 = last_v
                    v1 = min(self.N, v0 + nb)
                    t0w = last_h + 1
                    t1w = t0w + Tw

                    A_batch = self.A[v0:v1, :]            # [nb,R] on CPU
                    B = self.B                            # [C,R]  on CPU
                    X_win = self.global_matrix[:, t0w:t1w] # [R,Tw] on CPU

                    # move to GPU only for computing stats (cheap)
                    if use_cuda:
                        A_batch = A_batch.cuda()
                        B = B.cuda()
                        X_win = X_win.cuda()
                        out_ = out.cuda()
                    else:
                        out_ = out

                    Y_hat = torch.einsum('nr,cr,rt->nct', A_batch, B, X_win)

                    last_out_mean = float(out_.mean().item())
                    last_out_std  = float(out_.std(unbiased=False).item())
                    last_hat_mean = float(Y_hat.mean().item())
                    last_hat_std  = float(Y_hat.std(unbiased=False).item())

            # ---------- Print
            if (it % print_every) == 0:
                elapsed = (time.time() - t0) / 60.0
                tcn_show = last_tcn_print if last_tcn_print is not None else -1.0

                msg = (
                    f"[it={it:6d}] epoch={self.Dataloader.epoch:3d}/{num_epochs} "
                    f"v={last_v:3d} h={last_h:5d} Tw={Tw:4d} | "
                    f"loss_AB={loss_ab_val:.6f} (ema {ema_ab:.6f}) | "
                    f"loss_X={loss_x_val:.6f} (ema {ema_x:.6f}) | "
                    f"loss_TCN={tcn_show:.6f} (ema {ema_tcn:.6f}) | "
                    f"elapsed={elapsed:.1f}m"
                )
                if log_sanity and (last_out_mean is not None):
                    msg += (
                        f" || out(mean/std)={last_out_mean:.3f}/{last_out_std:.3f} "
                        f"hat(mean/std)={last_hat_mean:.3f}/{last_hat_std:.3f}"
                    )
                print(msg)

            it += 1

        return (
            self.global_matrix.detach().cpu().numpy(),  # X: [R,T]
            self.A.detach().cpu().numpy(),              # A: [N,R]
            self.B.detach().cpu().numpy(),              # B: [C,R]
        )

    def train_GLDTR(
        self,
        num_epochs=30,
        reg_X=0.05,               # 正则先减小
        reg_AB=0.05,
        lam_temporal=0.1,         # TCN正则先别太猛
        mod=5,
        print_every=10):
        """
        为了兼容你原来接口，这里先提供一个简单版本：
        - 先跑一次 train_factors 得到 A,B,X
        - 返回 (X, A, B)
        """
        X, A, B = self.train_factors(num_epochs=num_epochs,
        reg_X=reg_X,
        reg_AB=reg_AB,
        lam_temporal=lam_temporal,
        mod=mod,
        print_every=print_every)
        return X, A, B

    def get_global_local(self, denorm=True):
        """
        Return:
          Y_global: [T,N,C]
          Y_local : [T,N,C]
        """
        A = self.A
        B = self.B
        X = self.global_matrix
        Yg = torch.einsum('nr,cr,rt->nct', A, B, X)  # [N,C,T]
        Yg = Yg.permute(2,0,1).detach().cpu().numpy()  # [T,N,C]

        Y = np.transpose(self.Y, (2,0,1))  # stored normalized+mini [T,N,C]
        Yl = Y - Yg

        if self.normalize and denorm:
            # inverse for both
            Yg = Yg - self.mini
            Yl = Yl - 0.0   # local 是差分，不需要减 mini；但为了统一 scale，先不动
            # 反归一化需要按 (N,C)
            s = self.s[None, :, :]   # [1,N,C]
            m = self.m[None, :, :]
            Yg = Yg * s + m
            Yl = Yl * s             # local 不加 m（因为 (Y - m)/s - (Yg - m)/s = (Y - Yg)/s）
        return Yg, Yl
    def get_AX(self):
        """
        Return AX = A @ X
        A: [N,R], X: [R,T]
        -> AX: [N,T]
        """
        with torch.no_grad():
            AX = torch.matmul(self.A, self.global_matrix)  # [N,T]
        return AX.detach().cpu().numpy()
