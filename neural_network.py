"""
Multi-Task Neural Network in pure NumPy for cardiovascular and diabetes screening.
Architecture: Shared representation backbone with 2 classification heads (chd, diabetes).
Optimized with mini-batch Adam, inverted dropout, gradient clipping, and analytical input gradients.
"""
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    x_clipped = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))

def swish(x: np.ndarray) -> np.ndarray:
    return x * sigmoid(x)

def swish_deriv(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s + x * s * (1.0 - s)

class MultiTaskNeuralNetwork:
    """
    Pure NumPy Multi-Task Neural Network with classification heads for CHD and Diabetes.
    """
    def __init__(
        self,
        input_dim: int = 13,
        hidden_sizes: Union[List[int], Tuple[int, ...]] = (32, 16),
        heads: Union[List[str], Tuple[str, ...]] = ("chd", "diabetes"),
        dropout: float = 0.1,
        l2_reg: float = 1e-4,
        lr: float = 0.005,
        seed: int = 42
    ):
        self.input_dim = int(input_dim)
        self.hidden_sizes = list(hidden_sizes)
        self.heads = list(heads)
        self.dropout_rate = float(dropout)
        self.l2_reg = float(l2_reg)
        self.lr = float(lr)
        self.seed = int(seed)

        self.rng = np.random.default_rng(self.seed)

        # Shared layers
        self.W_shared: List[np.ndarray] = []
        self.b_shared: List[np.ndarray] = []
        prev_dim = self.input_dim
        for h_dim in self.hidden_sizes:
            # He normal initialization for Swish
            std = np.sqrt(2.0 / prev_dim)
            W = self.rng.normal(0.0, std, (prev_dim, h_dim)).astype(np.float64)
            b = np.zeros((1, h_dim), dtype=np.float64)
            self.W_shared.append(W)
            self.b_shared.append(b)
            prev_dim = h_dim

        # Classification heads (each outputs a single logit for sigmoid)
        self.W_heads: Dict[str, np.ndarray] = {}
        self.b_heads: Dict[str, np.ndarray] = {}
        last_hidden = prev_dim
        for head in self.heads:
            std = np.sqrt(2.0 / last_hidden)
            W_h = self.rng.normal(0.0, std, (last_hidden, 1)).astype(np.float64)
            b_h = np.zeros((1, 1), dtype=np.float64)
            self.W_heads[head] = W_h
            self.b_heads[head] = b_h

        # Adam optimizer state
        self.t = 0
        self.mW_shared = [np.zeros_like(W) for W in self.W_shared]
        self.vW_shared = [np.zeros_like(W) for W in self.W_shared]
        self.mb_shared = [np.zeros_like(b) for b in self.b_shared]
        self.vb_shared = [np.zeros_like(b) for b in self.b_shared]

        self.mW_heads = {h: np.zeros_like(W) for h, W in self.W_heads.items()}
        self.vW_heads = {h: np.zeros_like(W) for h, W in self.W_heads.items()}
        self.mb_heads = {h: np.zeros_like(b) for h, b in self.b_heads.items()}
        self.vb_heads = {h: np.zeros_like(b) for h, b in self.b_heads.items()}

        # Stored normalization metadata
        self.metadata: Dict[str, Any] = {}

    def forward(
        self,
        X: np.ndarray,
        training: bool = False
    ) -> Tuple[Dict[str, np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Forward pass.
        Returns:
            probs: Dict[head -> probabilities (m, 1)]
            activations: list of activations including input
            z_values: list of pre-activations
            drop_masks: list of dropout masks used
        """
        curr_A = np.asarray(X, dtype=np.float64)
        if curr_A.ndim == 1:
            curr_A = curr_A.reshape(1, -1)

        activations = [curr_A]
        z_values = []
        drop_masks = []

        for i in range(len(self.hidden_sizes)):
            W = self.W_shared[i]
            b = self.b_shared[i]
            Z = np.dot(curr_A, W) + b
            z_values.append(Z)
            A = swish(Z)

            if training and self.dropout_rate > 0.0:
                keep_prob = 1.0 - self.dropout_rate
                mask = (self.rng.uniform(0.0, 1.0, size=A.shape) < keep_prob).astype(np.float32) / keep_prob
                A = A * mask
                drop_masks.append(mask)
            else:
                drop_masks.append(None)

            curr_A = A
            activations.append(curr_A)

        latent = curr_A
        probs = {}
        for head in self.heads:
            Z_h = np.dot(latent, self.W_heads[head]) + self.b_heads[head]
            p_h = sigmoid(Z_h)
            probs[head] = p_h

        return probs, activations, z_values, drop_masks

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Deterministic inference.
        """
        probs, _, _, _ = self.forward(X, training=False)
        return probs

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        n_mc: int = 30,
        seed: int = 42
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Monte Carlo Dropout inference for epistemic uncertainty estimation.
        Returns:
            {head: {"mean": np.ndarray, "std": np.ndarray}}
        """
        old_state = self.rng.bit_generator.state
        self.rng = np.random.default_rng(seed)

        accum = {h: [] for h in self.heads}
        for _ in range(n_mc):
            probs, _, _, _ = self.forward(X, training=True)
            for h in self.heads:
                accum[h].append(probs[h])

        # Restore RNG state
        self.rng.bit_generator.state = old_state

        result = {}
        for h in self.heads:
            stacked = np.array(accum[h])  # shape (n_mc, N, 1)
            result[h] = {
                "mean": np.mean(stacked, axis=0),
                "std": np.std(stacked, axis=0)
            }
        return result

    def integrated_gradients(
        self,
        x: np.ndarray,
        baseline: np.ndarray,
        head: str = "chd",
        m_steps: int = 50
    ) -> np.ndarray:
        """
        Computes path-integrated gradients attribution vector from baseline to x.
        Completeness property: sum(attributions) approx p(x) - p(baseline).
        """
        x_in = np.asarray(x, dtype=np.float64).ravel()
        base_in = np.asarray(baseline, dtype=np.float64).ravel()
        diff = x_in - base_in

        # Accumulate gradients along linear path
        grads = []
        for k in range(1, m_steps + 1):
            alpha = k / float(m_steps)
            step_x = base_in + alpha * diff
            g = self.input_gradient(step_x, head=head)
            grads.append(g)

        avg_grad = np.mean(grads, axis=0)
        attributions = diff * avg_grad
        return attributions


    def compute_loss(
        self,
        probs: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        loss_weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Computes total loss (BCE + L2 regularization).
        """
        if loss_weights is None:
            loss_weights = {h: 1.0 for h in self.heads}

        total_loss = 0.0
        eps = 1e-7

        for head in self.heads:
            if head in targets:
                p = np.clip(probs[head], eps, 1.0 - eps)
                y = targets[head].reshape(-1, 1)
                bce = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
                total_loss += loss_weights.get(head, 1.0) * float(bce)

        # L2 Regularization
        if self.l2_reg > 0.0:
            l2_sum = sum(np.sum(W ** 2) for W in self.W_shared)
            l2_sum += sum(np.sum(W ** 2) for W in self.W_heads.values())
            total_loss += 0.5 * self.l2_reg * float(l2_sum)

        return float(total_loss)

    def train_step(
        self,
        X_batch: np.ndarray,
        y_batch: Dict[str, np.ndarray],
        loss_weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Executes one mini-batch training step with backpropagation and Adam update.
        """
        if loss_weights is None:
            loss_weights = {h: 1.0 for h in self.heads}

        m = len(X_batch)
        probs, activations, z_values, drop_masks = self.forward(X_batch, training=True)
        loss = self.compute_loss(probs, y_batch, loss_weights)

        # Backpropagation: Classification heads
        dW_heads = {}
        db_heads = {}
        dA_latent = np.zeros_like(activations[-1])

        for head in self.heads:
            if head in y_batch:
                y = y_batch[head].reshape(-1, 1)
                w_loss = loss_weights.get(head, 1.0)
                # For Sigmoid + BCE: dZ = (p - y)
                dZ_h = (probs[head] - y) * w_loss
                dW_heads[head] = (np.dot(activations[-1].T, dZ_h) / m) + self.l2_reg * self.W_heads[head]
                db_heads[head] = np.mean(dZ_h, axis=0, keepdims=True)
                dA_latent += np.dot(dZ_h, self.W_heads[head].T)

        # Backpropagation: Shared layers
        dW_shared = []
        db_shared = []
        curr_dA = dA_latent

        for i in reversed(range(len(self.hidden_sizes))):
            mask = drop_masks[i]
            if mask is not None:
                curr_dA = curr_dA * mask

            Z = z_values[i]
            dZ = curr_dA * swish_deriv(Z)
            prev_A = activations[i]

            dW = (np.dot(prev_A.T, dZ) / m) + self.l2_reg * self.W_shared[i]
            db = np.mean(dZ, axis=0, keepdims=True)

            dW_shared.insert(0, dW)
            db_shared.insert(0, db)

            curr_dA = np.dot(dZ, self.W_shared[i].T)

        # Adam optimizer update with gradient clipping
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        grad_clip = 5.0

        # Update shared layers
        for i in range(len(self.hidden_sizes)):
            gW = np.clip(dW_shared[i], -grad_clip, grad_clip)
            gb = np.clip(db_shared[i], -grad_clip, grad_clip)

            self.mW_shared[i] = beta1 * self.mW_shared[i] + (1 - beta1) * gW
            self.vW_shared[i] = beta2 * self.vW_shared[i] + (1 - beta2) * (gW ** 2)
            self.mb_shared[i] = beta1 * self.mb_shared[i] + (1 - beta1) * gb
            self.vb_shared[i] = beta2 * self.vb_shared[i] + (1 - beta2) * (gb ** 2)

            m_hat_w = self.mW_shared[i] / (1 - beta1 ** self.t)
            v_hat_w = self.vW_shared[i] / (1 - beta2 ** self.t)
            m_hat_b = self.mb_shared[i] / (1 - beta1 ** self.t)
            v_hat_b = self.vb_shared[i] / (1 - beta2 ** self.t)

            self.W_shared[i] -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + eps)
            self.b_shared[i] -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

        # Update heads
        for head in self.heads:
            if head in dW_heads:
                gW = np.clip(dW_heads[head], -grad_clip, grad_clip)
                gb = np.clip(db_heads[head], -grad_clip, grad_clip)

                self.mW_heads[head] = beta1 * self.mW_heads[head] + (1 - beta1) * gW
                self.vW_heads[head] = beta2 * self.vW_heads[head] + (1 - beta2) * (gW ** 2)
                self.mb_heads[head] = beta1 * self.mb_heads[head] + (1 - beta1) * gb
                self.vb_heads[head] = beta2 * self.vb_heads[head] + (1 - beta2) * (gb ** 2)

                m_hat_w = self.mW_heads[head] / (1 - beta1 ** self.t)
                v_hat_w = self.vW_heads[head] / (1 - beta2 ** self.t)
                m_hat_b = self.mb_heads[head] / (1 - beta1 ** self.t)
                v_hat_b = self.vb_heads[head] / (1 - beta2 ** self.t)

                self.W_heads[head] -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + eps)
                self.b_heads[head] -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

        return loss

    def input_gradient(self, x: np.ndarray, head: str = "chd") -> np.ndarray:
        """
        Computes the analytical gradient of the output probability p(x) with respect
        to input features x (shape 1, input_dim).
        Returns gradient vector of shape (input_dim,).
        """
        x_in = np.asarray(x, dtype=np.float32).reshape(1, -1)
        probs, activations, z_values, _ = self.forward(x_in, training=False)

        p = probs[head][0, 0]
        # dp/dZ_head = p * (1 - p)
        dZ_head = p * (1.0 - p)

        curr_dA = dZ_head * self.W_heads[head].T  # shape (1, last_hidden)

        for i in reversed(range(len(self.hidden_sizes))):
            Z = z_values[i]
            dZ = curr_dA * swish_deriv(Z)
            curr_dA = np.dot(dZ, self.W_shared[i].T)

        return curr_dA.ravel().astype(np.float64)

    def save(self, filepath: Union[str, Path], metadata: Optional[Dict[str, Any]] = None):
        """
        Saves network weights, architecture, and normalization metadata to .npz without pickle.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "format_version": np.array("2.0"),
            "input_dim": np.array(self.input_dim),
            "hidden_sizes": np.array(self.hidden_sizes),
            "heads": np.array(self.heads),
            "num_shared_layers": np.array(len(self.W_shared)),
            "created_at": np.array(datetime.now().isoformat()),
        }

        for i, (W, b) in enumerate(zip(self.W_shared, self.b_shared)):
            save_dict[f"W_shared_{i}"] = W
            save_dict[f"b_shared_{i}"] = b

        for head in self.heads:
            save_dict[f"W_head_{head}"] = self.W_heads[head]
            save_dict[f"b_head_{head}"] = self.b_heads[head]

        meta = metadata if metadata is not None else self.metadata
        if meta:
            save_dict["metadata_json"] = np.array(json.dumps(meta, ensure_ascii=False))

        np.savez_compressed(path, **save_dict)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "MultiTaskNeuralNetwork":
        """
        Loads network weights and metadata from .npz with allow_pickle=False.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Weights file not found: {path}")

        with np.load(path, allow_pickle=False) as data:
            input_dim = int(data["input_dim"])
            hidden_sizes = [int(x) for x in data["hidden_sizes"]]
            heads = [str(x) for x in data["heads"]]

            model = cls(input_dim=input_dim, hidden_sizes=hidden_sizes, heads=heads)

            num_shared = int(data["num_shared_layers"])
            model.W_shared = [data[f"W_shared_{i}"].copy() for i in range(num_shared)]
            model.b_shared = [data[f"b_shared_{i}"].copy() for i in range(num_shared)]

            for head in heads:
                model.W_heads[head] = data[f"W_head_{head}"].copy()
                model.b_heads[head] = data[f"b_head_{head}"].copy()

            if "metadata_json" in data:
                model.metadata = json.loads(str(data["metadata_json"]))

        return model
