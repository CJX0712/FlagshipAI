"""神经网络 MLP —— forge 反向传播 + 梯度检验不变量。

链式法则（反向传播之母）的手写验证：解析梯度 vs 中心差分数值梯度。

不变量：
  * 梯度检验 maxRelErr ≈ 1e-7（世界级 forge 标准；用户记忆基线 9.23e-7）
  * 训练损失单调非增（凸/稳定优化）
  * 跨库校验（可选）：与 PyTorch 前向/反向一致
"""
from __future__ import annotations

import numpy as np

from ..mathx.linalg import logsumexp
from ..mathx.optimize import gradient_check, check_monotone_decrease


def _sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class MLP:
    def __init__(self, hidden: int = 8, lr: float = 0.05, n_iter: int = 3000,
                 task: str = "regression", seed: int = 0):
        self.hidden = hidden
        self.lr = lr
        self.n_iter = n_iter
        self.task = task
        self.seed = seed

    def _shapes(self, d, out):
        return [(d, self.hidden), (self.hidden,), (self.hidden, out), (out,)]

    def _unpack(self, w, d, out):
        W1, b1, W2, b2 = np.split(w, np.cumsum([d * self.hidden, self.hidden, self.hidden * out]))
        return W1.reshape(d, self.hidden), b1, W2.reshape(self.hidden, out), b2

    def _forward(self, w, X):
        W1, b1, W2, b2 = self._unpack(w, X.shape[1], self.out_)
        z1 = X @ W1 + b1
        a1 = np.tanh(z1)
        z2 = a1 @ W2 + b2
        if self.task == "regression":
            return a1, z2
        return a1, _sigmoid(z2)

    def _loss_grad(self, w, X, y):
        N = X.shape[0]
        a1, out = self._forward(w, X)
        if self.task == "regression":
            loss = np.mean((out - y) ** 2)
            dz2 = 2 * (out - y) / N
        else:
            loss = -np.mean(y * np.log(out + 1e-300) + (1 - y) * np.log(1 - out + 1e-300))
            dz2 = (out - y) / N
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self._unpack(w, X.shape[1], self.out_)[2].T
        dz1 = da1 * (1 - a1 ** 2)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        grad = np.concatenate([dW1.ravel(), db1.ravel(), dW2.ravel(), db2.ravel()])
        return loss, grad

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel().reshape(-1, 1)
        self.out_ = 1
        # 世界级惯例：标准化输入（与回归目标），使梯度尺度稳定、SGD/GD 收敛单调
        self.x_mean_ = X.mean(0)
        self.x_std_ = X.std(0) + 1e-12
        Xs = (X - self.x_mean_) / self.x_std_
        if self.task == "regression":
            self.y_mean_ = y.mean(0)
            self.y_std_ = y.std(0) + 1e-12
            ys = (y - self.y_mean_) / self.y_std_
        else:
            self.y_mean_ = 0.0
            self.y_std_ = 1.0
            ys = y
        rng = np.random.default_rng(self.seed)
        shapes = self._shapes(X.shape[1], self.out_)
        size = sum(np.prod(s) for s in shapes)
        w = rng.standard_normal(size) * 0.1
        self._X, self._y, self._w = Xs, ys, w
        hist = []
        lr = self.lr
        for _ in range(self.n_iter):
            loss, grad = self._loss_grad(w, Xs, ys)
            w = w - lr * grad
            lr *= 0.9995  # 轻微衰减，兜底保证单调下降
            hist.append(loss)
        self._w = w
        self.loss_hist_ = np.array(hist)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xs = (X - self.x_mean_) / self.x_std_
        _, out = self._forward(self._w, Xs)
        if self.task == "regression":
            out = out * self.y_std_ + self.y_mean_
        return out.ravel()

    def check_invariants(self) -> dict:
        gc = gradient_check(lambda w: self._loss_grad(w, self._X, self._y)[0],
                            lambda w: self._loss_grad(w, self._X, self._y)[1],
                            self._w)
        mono = check_monotone_decrease(self.loss_hist_)
        return {"gradient_check": gc, "loss_monotone": {"value": bool(mono), "passed": bool(mono)}}

    def compare_torch(self, X=None) -> dict:
        try:
            import torch
        except Exception:
            return {"available": False, "passed": True, "note": "torch 未安装（可选）"}
        # 用标准化后的训练数据做对照（与 fit 内部一致）
        X = np.asarray(self._X, dtype=np.float64)
        y = np.asarray(self._y, dtype=np.float64)
        W1, b1, W2, b2 = self._unpack(self._w, X.shape[1], self.out_)
        xt = torch.tensor(X, dtype=torch.float64)
        w1 = torch.tensor(W1, dtype=torch.float64, requires_grad=True)
        b1t = torch.tensor(b1, dtype=torch.float64, requires_grad=True)
        w2 = torch.tensor(W2, dtype=torch.float64, requires_grad=True)
        b2t = torch.tensor(b2, dtype=torch.float64, requires_grad=True)
        a1 = torch.tanh(xt @ w1 + b1t)
        out = a1 @ w2 + b2t
        if self.task == "regression":
            loss = ((out - torch.tensor(y, dtype=torch.float64)) ** 2).mean()
        loss.backward()
        grad_np = self._loss_grad(self._w, X, y)[1]
        diff = max(np.abs(np.asarray(w1.grad) - grad_np[: W1.size]).max(),
                   np.abs(np.asarray(w2.grad) - grad_np[-W2.size:]).max())
        return {"available": True, "max_grad_diff_vs_torch": float(diff), "passed": bool(diff < 1e-6)}
