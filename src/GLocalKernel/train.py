from typing import Optional

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class GLKLossFunction(nn.Module):
    def forward(
        self,
        pred_p: torch.Tensor,
        reg_loss: torch.Tensor,
        train_r: torch.Tensor,
        train_m: torch.Tensor,
    ) -> torch.Tensor:
        # Compute the elementwise difference between train ratings and predictions,
        # weighted by the mask.
        diff = train_m * (train_r - pred_p)
        # Using MSE loss with respect to zeros is equivalent to computing the mean of squared differences.
        mse = F.mse_loss(diff, torch.zeros_like(diff))
        loss = mse + reg_loss
        return loss


class GLKTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer_p: torch.optim.Optimizer,
        optimizer_f: torch.optim.Optimizer,
        epoch_p: int,
        epoch_f: int,
        loss_fn: Optional[nn.Module] = GLKLossFunction(),
    ) -> None:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.double().to(device)
        self.optimizer_p = optimizer_p
        self.optimizer_f = optimizer_f
        self.epoch_p = epoch_p
        self.epoch_f = epoch_f
        self.loss_fn = loss_fn
        self.device = device

    def __closure_p(self, train_r: torch.Tensor, train_m: torch.Tensor) -> torch.Tensor:
        self.optimizer_p.zero_grad()
        # In pre-training, we use the local kernel network only.
        self.model.local_kernel_net.train()
        pred, reg = self.model.local_kernel_net(train_r)
        loss = self.loss_fn(pred, reg, train_r, train_m)
        loss.backward()
        return loss

    def pre_train_model(
        self,
        train_r: torch.Tensor,
        train_m: torch.Tensor,
        test_r: torch.Tensor,
        test_m: torch.Tensor,
        tol_p: float,
        patience_p: int,
        show_logs_per_epoch: Optional[int] = 5,
    ) -> None:
        # Ensure data is on the proper device and in double precision.
        train_r = train_r.to(self.device).double()
        train_m = train_m.to(self.device).double()
        test_r = test_r.to(self.device).double()
        test_m = test_m.to(self.device).double()

        last_rmse = np.inf
        counter = 0
        time_cumulative = 0
        tic = time.time()

        for epoch in range(self.epoch_p):
            self.optimizer_p.step(lambda: self.__closure_p(train_r, train_m))

            # Evaluate the model.
            self.model.local_kernel_net.eval()
            t = time.time() - tic
            time_cumulative += t

            with torch.no_grad():
                # Evaluate training RMSE (using the local net output)
                pred_train, _ = self.model.local_kernel_net(train_r)

                # Move predictions to CPU and convert to float numpy array.
                pred_train_np = pred_train.float().cpu().detach().numpy()
                train_r_np = train_r.cpu().numpy()
                train_m_np = train_m.cpu().numpy()
                train_error = (
                    train_m_np * (np.clip(pred_train_np, 1.0, 5.0) - train_r_np) ** 2
                ).sum() / train_m_np.sum()
                train_rmse = np.sqrt(train_error)

                # For test RMSE, we run the full model (global + local).
                pred_test, _ = self.model(test_r, test_r)
                pred_test_np = pred_test.float().cpu().detach().numpy()
                test_r_np = test_r.cpu().numpy()
                test_m_np = test_m.cpu().numpy()
                test_error = (
                    test_m_np * (np.clip(pred_test_np, 1.0, 5.0) - test_r_np) ** 2
                ).sum() / test_m_np.sum()
                test_rmse = np.sqrt(test_error)

            # Early stopping logic.
            if last_rmse - train_rmse < tol_p:
                counter += 1
            else:
                counter = 0
            last_rmse = train_rmse

            if counter >= patience_p:
                print(
                    f"Epoch: {epoch+1}, Test RMSE: {test_rmse:.4f}, Train RMSE: {train_rmse:.4f}"
                )
                print(
                    f"Time (current epoch): {t:.2f} sec, Cumulative Time: {time_cumulative:.2f} sec"
                )
                print(".~" * 30)
                break

            if epoch % show_logs_per_epoch == 0:
                print("PRE-TRAINING")
                print(
                    f"Epoch: {epoch+1}, Test RMSE: {test_rmse:.4f}, Train RMSE: {train_rmse:.4f}"
                )
                print(
                    f"Time (current epoch): {t:.2f} sec, Cumulative Time: {time_cumulative:.2f} sec"
                )
                print(".~" * 30)
