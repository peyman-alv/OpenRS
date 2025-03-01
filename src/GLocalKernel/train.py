from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from src.utils.metrics import call_ndcg


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

    def pre_train_model(
        self,
        train_r: torch.Tensor,
        train_m: torch.Tensor,
        test_r: torch.Tensor,
        test_m: torch.Tensor,
        tol_p: float,
        patience_p: int,
        show_logs_per_epoch: Optional[int] = 5,
    ) -> torch.Tensor:

        def clouser_p():
            self.optimizer_p.zero_grad()
            self.model.local_kernel_net.train()
            pred, reg = self.model.local_kernel_net(train_r)
            loss = self.loss_fn(pred, reg, train_r, train_m)
            loss.backward()
            return loss

        # Convert the data to numpy arrays for error computation.
        train_r_np = train_r.float().cpu().detach().numpy()
        train_m_np = train_m.float().cpu().detach().numpy()
        test_r_np = test_r.float().cpu().detach().numpy()
        test_m_np = test_m.float().cpu().detach().numpy()

        # Ensure the data is in the same device as the model.
        train_r = train_r.double().to(self.device)
        train_m = train_m.double().to(self.device)
        test_r = test_r.double().to(self.device)
        test_m = test_m.double().to(self.device)

        # Initialize the variables to store the best model and the best RMSE.
        last_rmse = np.inf
        counter = 0

        # Pre-train the model.
        for epoch in range(self.epoch_p):
            self.optimizer_p.step(clouser_p)
            self.model.local_kernel_net.eval()

            pre, _ = self.model.local_kernel_net(train_r)
            pre_np = pre.float().cpu().detach().numpy()

            train_rmse = np.sqrt(
                (train_m_np * (np.clip(pre_np, 1.0, 5.0) - train_r_np) ** 2).sum()
                / train_m_np.sum()
            )
            test_rmse = np.sqrt(
                (test_m_np * (np.clip(pre_np, 1.0, 5.0) - test_r_np) ** 2).sum()
                / test_m_np.sum()
            )

            counter = counter + 1 if last_rmse - train_rmse < tol_p else 0
            last_rmse = train_rmse

            if patience_p == counter:
                print(
                    "Early stopping at epoch: ",
                    epoch,
                    "Train RMSE: ",
                    train_rmse,
                    "Test RMSE: ",
                    test_rmse,
                )
                print("~." * 50)
                break

            if show_logs_per_epoch is not None and epoch % show_logs_per_epoch == 0:
                print(
                    "Pre-training epoch: ",
                    epoch,
                    "Train RMSE: ",
                    train_rmse,
                    "Test RMSE: ",
                    test_rmse,
                )
                print("~." * 50)

        return pre

    def fine_tune_model(
        self,
        train_r: torch.Tensor,
        train_r_local: torch.Tensor,
        train_m: torch.Tensor,
        test_r: torch.Tensor,
        test_m: torch.Tensor,
        tol_f: float,
        patience_f: int,
        show_logs_per_epoch: Optional[int] = 20,
    ) -> None:

        def clouser_f():
            self.optimizer_f.zero_grad()
            x_local = (
                torch.Tensor(
                    np.clip(train_r_local.float().cpu().detach().numpy(), 1.0, 5.0)
                )
                .double()
                .to(self.device)
            )
            self.model.train()
            pred, reg = self.model(train_r, x_local)
            loss = self.loss_fn(pred, reg, train_r, train_m)
            loss.backward()
            return loss

        # Convert the data to numpy arrays for error computation.
        train_r_np = train_r.float().cpu().detach().numpy()
        train_m_np = train_m.float().cpu().detach().numpy()
        test_r_np = test_r.float().cpu().detach().numpy()
        test_m_np = test_m.float().cpu().detach().numpy()

        # Ensure the data is in the same device as the model.
        train_r = train_r.double().to(self.device)
        train_r_local = train_r_local.double().to(self.device)
        train_m = train_m.double().to(self.device)
        test_r = test_r.double().to(self.device)
        test_m = test_m.double().to(self.device)

        # Initialize the variables to store the best model and the best RMSE.
        last_rmse = np.inf
        counter = 0
        best_rmse_ep, best_mae_ep, best_ndcg_ep = 0, 0, 0
        best_rmse, best_mae, best_ndcg = float("inf"), float("inf"), 0

        # Fine-tune the model.
        for epoch in range(self.epoch_f):
            self.optimizer_f.step(clouser_f)
            self.model.eval()

            pre, _ = self.model(train_r, train_r_local)
            pre_np = pre.float().cpu().detach().numpy()
            pre_np = np.clip(pre_np, 1.0, 5.0)

            # Compute the RMSE for the train and test sets.
            train_rmse = np.sqrt(
                (train_m_np * (pre_np - train_r_np) ** 2).sum() / train_m_np.sum()
            )
            test_rmse = np.sqrt(
                (test_m_np * (pre_np - test_r_np) ** 2).sum() / test_m_np.sum()
            )

            # Compute the MAE for the train and test sets.
            train_mae = (
                train_m_np * np.abs(pre_np - train_r_np)
            ).sum() / train_m_np.sum()
            test_mae = (test_m_np * np.abs(pre_np - test_r_np)).sum() / test_m_np.sum()

            # Compute the NDCG for the train and test sets.
            train_ndcg = call_ndcg(pre_np, train_r_np)
            test_ndcg = call_ndcg(pre_np, test_r_np)

            if test_rmse < best_rmse:
                best_rmse = test_rmse
                best_rmse_ep = epoch + 1

            if test_mae < best_mae:
                best_mae = test_mae
                best_mae_ep = epoch + 1

            if best_ndcg < test_ndcg:
                best_ndcg = test_ndcg
                best_ndcg_ep = epoch + 1

            counter = counter + 1 if last_rmse - train_rmse < tol_f else 0
            last_rmse = train_rmse

            if patience_f == counter:
                print(
                    "Early stopping at epoch: ",
                    epoch,
                    "\n",
                    "Train RMSE: ",
                    train_rmse,
                    "Train MAE: ",
                    train_mae,
                    "Train NDCG: ",
                    train_ndcg,
                    "\n",
                    "Test RMSE: ",
                    test_rmse,
                    "Test MAE: ",
                    test_mae,
                    "Test NDCG: ",
                    test_ndcg,
                )
                print("~." * 50)
                break

            if show_logs_per_epoch is not None and epoch % show_logs_per_epoch == 0:
                print(
                    "Fine-tuning epoch: ",
                    epoch,
                    "\n",
                    "Train RMSE: ",
                    train_rmse,
                    "Train MAE: ",
                    train_mae,
                    "Train NDCG: ",
                    train_ndcg,
                    "\n",
                    "Test RMSE: ",
                    test_rmse,
                    "Test MAE: ",
                    test_mae,
                    "Test NDCG: ",
                    test_ndcg,
                )
                print("~." * 50)

        print("Epoch:", best_rmse_ep, " best rmse:", best_rmse)
        print("Epoch:", best_mae_ep, " best mae:", best_mae)
        print("Epoch:", best_ndcg_ep, " best ndcg:", best_ndcg)
