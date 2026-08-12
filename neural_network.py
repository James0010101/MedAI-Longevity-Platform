import numpy as np

# Loss weight constants — used in BOTH backprop gradients AND total loss computation
# to ensure the optimizer minimizes the exact loss we report to the user.
LOSS_WEIGHTS = {
    "cardiac": 1.0,
    "diabetes": 1.0,
    "life": 0.005,
    "vascular": 0.005,
}

# Adam optimizer hyperparameters
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

# Gradient clipping threshold
GRAD_CLIP = 2.0

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(a):
    return a * (1.0 - a)

def swish(x):
    return x * sigmoid(x)

def swish_deriv(x):
    s = sigmoid(x)
    return s + x * s * (1.0 - s)

class MultiTaskNeuralNetwork:
    """
    Multi-Task Deep Neural Engine written in pure NumPy.
    Processes shared input biomarkers through deep shared hidden layers (Stage 1 & 2),
    and predicts 4 clinical heads simultaneously (Stage 3):
    1. Cardiac Disease Risk (Classification)
    2. Diabetes Risk (Classification)
    3. Life Expectancy (Regression)
    4. Biological Vascular Age (Regression)
    """
    def __init__(self, input_dim=10, hidden_sizes=[20, 12], learning_rate=0.02, l2_reg=1e-4, seed=42):
        self.input_dim = input_dim
        self.hidden_sizes = list(hidden_sizes)
        self.act_func = swish
        self.act_deriv = swish_deriv
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        
        rng = np.random.default_rng(seed)
        
        # Shared Hidden Layers
        self.W_shared = []
        self.b_shared = []
        self.m_w_shared, self.v_w_shared = [], []
        self.m_b_shared, self.v_b_shared = [], []
        
        prev_dim = input_dim
        for h_dim in hidden_sizes:
            scale = np.sqrt(2.0 / prev_dim) # He initialization for Swish
            W = rng.standard_normal((prev_dim, h_dim)) * scale
            b = np.zeros((1, h_dim))
            
            self.W_shared.append(W)
            self.b_shared.append(b)
            self.m_w_shared.append(np.zeros_like(W))
            self.v_w_shared.append(np.zeros_like(W))
            self.m_b_shared.append(np.zeros_like(b))
            self.v_b_shared.append(np.zeros_like(b))
            prev_dim = h_dim
            
        last_hidden_dim = prev_dim
        
        self.W_heads = {}
        self.b_heads = {}
        self.m_w_heads, self.v_w_heads = {}, {}
        self.m_b_heads, self.v_b_heads = {}, {}
        
        for head in ["cardiac", "diabetes", "life", "vascular"]:
            W = rng.standard_normal((last_hidden_dim, 1)) * np.sqrt(1.0 / last_hidden_dim)
            b = np.zeros((1, 1))
            if head == "life":
                b[0, 0] = 78.0
            elif head == "vascular":
                b[0, 0] = 50.0
                
            self.W_heads[head] = W
            self.b_heads[head] = b
            self.m_w_heads[head] = np.zeros_like(W)
            self.v_w_heads[head] = np.zeros_like(W)
            self.m_b_heads[head] = np.zeros_like(b)
            self.v_b_heads[head] = np.zeros_like(b)
            
        self.t = 0
        self.activations = []
        self.z_values = []

    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        curr_A = X
        for i in range(len(self.hidden_sizes)):
            W = self.W_shared[i]
            b = self.b_shared[i]
            Z = np.dot(curr_A, W) + b
            self.z_values.append(Z)
            curr_A = self.act_func(Z)
            self.activations.append(curr_A)
            
        latent_features = curr_A
        
        z_cardiac = np.dot(latent_features, self.W_heads["cardiac"]) + self.b_heads["cardiac"]
        p_cardiac = sigmoid(z_cardiac)
        
        z_diabetes = np.dot(latent_features, self.W_heads["diabetes"]) + self.b_heads["diabetes"]
        p_diabetes = sigmoid(z_diabetes)
        
        z_life = np.dot(latent_features, self.W_heads["life"]) + self.b_heads["life"]
        pred_life = z_life
        
        z_vascular = np.dot(latent_features, self.W_heads["vascular"]) + self.b_heads["vascular"]
        pred_vascular = z_vascular
        
        return {
            "cardiac": p_cardiac,
            "diabetes": p_diabetes,
            "life": pred_life,
            "vascular": pred_vascular,
            "latent": latent_features
        }

    def _adam_update(self, param, grad, m_buf, v_buf):
        """Single Adam optimizer step. Returns updated (param, m_buf, v_buf)."""
        m_buf = ADAM_BETA1 * m_buf + (1 - ADAM_BETA1) * grad
        v_buf = ADAM_BETA2 * v_buf + (1 - ADAM_BETA2) * (grad ** 2)
        m_corr = m_buf / (1 - ADAM_BETA1 ** self.t)
        v_corr = v_buf / (1 - ADAM_BETA2 ** self.t)
        param -= self.learning_rate * m_corr / (np.sqrt(v_corr) + ADAM_EPS)
        return param, m_buf, v_buf

    def train_step(self, X, y_cardiac, y_diabetes, y_life, y_vascular, dropout_rate=0.1):
        m = X.shape[0]
        self.t += 1
        
        # Stochastic forward pass with Inverted Dropout
        self.activations = [X]
        self.z_values = []
        self.dropout_masks = []
        
        curr_A = X
        for i in range(len(self.hidden_sizes)):
            W = self.W_shared[i]
            b = self.b_shared[i]
            Z = np.dot(curr_A, W) + b
            self.z_values.append(Z)
            A = self.act_func(Z)
            
            mask = (np.random.uniform(0, 1, size=A.shape) >= dropout_rate).astype(float)
            curr_A = (A * mask) / (1.0 - dropout_rate)
            
            self.dropout_masks.append(mask)
            self.activations.append(curr_A)
            
        latent_features = curr_A
        
        z_cardiac = np.dot(latent_features, self.W_heads["cardiac"]) + self.b_heads["cardiac"]
        preds_cardiac = sigmoid(z_cardiac)
        
        z_diabetes = np.dot(latent_features, self.W_heads["diabetes"]) + self.b_heads["diabetes"]
        preds_diabetes = sigmoid(z_diabetes)
        
        preds_life = np.dot(latent_features, self.W_heads["life"]) + self.b_heads["life"]
        preds_vascular = np.dot(latent_features, self.W_heads["vascular"]) + self.b_heads["vascular"]
        
        preds = {
            "cardiac": preds_cardiac,
            "diabetes": preds_diabetes,
            "life": preds_life,
            "vascular": preds_vascular
        }
        
        latent_A = self.activations[-1]
        
        # Gradient scaling uses the SAME LOSS_WEIGHTS as the reported total loss
        dZ_cardiac = LOSS_WEIGHTS["cardiac"] * (preds["cardiac"] - y_cardiac)
        dZ_diabetes = LOSS_WEIGHTS["diabetes"] * (preds["diabetes"] - y_diabetes)
        dZ_life = LOSS_WEIGHTS["life"] * 2.0 * (preds["life"] - y_life) / m
        dZ_vascular = LOSS_WEIGHTS["vascular"] * 2.0 * (preds["vascular"] - y_vascular) / m
        
        dZ_heads = {
            "cardiac": dZ_cardiac,
            "diabetes": dZ_diabetes,
            "life": dZ_life,
            "vascular": dZ_vascular
        }
        
        dA_latent = np.zeros_like(latent_A)
        
        for head in ["cardiac", "diabetes", "life", "vascular"]:
            dZ = dZ_heads[head]
            dW = (1.0 / m) * np.dot(latent_A.T, dZ) + self.l2_reg * self.W_heads[head]
            db = (1.0 / m) * np.sum(dZ, axis=0, keepdims=True)
            
            dW = np.clip(dW, -GRAD_CLIP, GRAD_CLIP)
            db = np.clip(db, -GRAD_CLIP, GRAD_CLIP)
            
            dA_latent += np.dot(dZ, self.W_heads[head].T)
            
            self.W_heads[head], self.m_w_heads[head], self.v_w_heads[head] = self._adam_update(
                self.W_heads[head], dW, self.m_w_heads[head], self.v_w_heads[head])
            self.b_heads[head], self.m_b_heads[head], self.v_b_heads[head] = self._adam_update(
                self.b_heads[head], db, self.m_b_heads[head], self.v_b_heads[head])
            
        dZ = dA_latent * (self.dropout_masks[-1] / (1.0 - dropout_rate)) * self.act_deriv(self.z_values[-1])
        
        for i in reversed(range(len(self.hidden_sizes))):
            A_prev = self.activations[i]
            dW = (1.0 / m) * np.dot(A_prev.T, dZ) + self.l2_reg * self.W_shared[i]
            db = (1.0 / m) * np.sum(dZ, axis=0, keepdims=True)
            
            dW = np.clip(dW, -GRAD_CLIP, GRAD_CLIP)
            db = np.clip(db, -GRAD_CLIP, GRAD_CLIP)
            
            if i > 0:
                dA_prev = np.dot(dZ, self.W_shared[i].T)
                dZ = dA_prev * (self.dropout_masks[i - 1] / (1.0 - dropout_rate)) * self.act_deriv(self.z_values[i - 1])
                
            self.W_shared[i], self.m_w_shared[i], self.v_w_shared[i] = self._adam_update(
                self.W_shared[i], dW, self.m_w_shared[i], self.v_w_shared[i])
            self.b_shared[i], self.m_b_shared[i], self.v_b_shared[i] = self._adam_update(
                self.b_shared[i], db, self.m_b_shared[i], self.v_b_shared[i])

        eps_log = 1e-15  # Separate name to avoid shadowing Adam's ADAM_EPS
        loss_cardiac = -np.mean(y_cardiac * np.log(preds["cardiac"] + eps_log) + (1 - y_cardiac) * np.log(1 - preds["cardiac"] + eps_log))
        loss_diabetes = -np.mean(y_diabetes * np.log(preds["diabetes"] + eps_log) + (1 - y_diabetes) * np.log(1 - preds["diabetes"] + eps_log))
        loss_life = np.mean((preds["life"] - y_life) ** 2)
        loss_vascular = np.mean((preds["vascular"] - y_vascular) ** 2)
        
        # Uses the SAME LOSS_WEIGHTS as the backprop gradients above
        total_loss = float(
            LOSS_WEIGHTS["cardiac"] * loss_cardiac +
            LOSS_WEIGHTS["diabetes"] * loss_diabetes +
            LOSS_WEIGHTS["life"] * loss_life +
            LOSS_WEIGHTS["vascular"] * loss_vascular
        )
        return total_loss

    def get_feature_attributions(self, X_vec):
        """
        Explainable AI (XAI): Axiomatic Attribution via Integrated Gradients.
        IG_i(x) = (x_i - x'_i) * sum_{k=1}^m dF(x' + k/m * (x - x')) / dx_i
        """
        baseline = np.zeros_like(X_vec)
        steps = 50
        
        scaled_inputs = [baseline + (float(i) / steps) * (X_vec - baseline) for i in range(steps + 1)]
        grads = []
        eps = 1e-4
        
        for x_curr in scaled_inputs:
            grad_curr = []
            for feat_idx in range(x_curr.shape[1]):
                x_plus = x_curr.copy()
                x_minus = x_curr.copy()
                x_plus[0, feat_idx] += eps
                x_minus[0, feat_idx] -= eps
                
                p_plus = float(self.forward(x_plus)["cardiac"][0, 0])
                p_minus = float(self.forward(x_minus)["cardiac"][0, 0])
                grad_curr.append((p_plus - p_minus) / (2 * eps))
            grads.append(grad_curr)
            
        avg_grads = np.mean(grads, axis=0)
        integrated_grad = (X_vec[0] - baseline[0]) * avg_grads
        total = np.sum(np.abs(integrated_grad)) + 1e-8
        percentages = (np.abs(integrated_grad) / total) * 100.0
        return percentages

    def get_stage_features(self, X):
        preds = self.forward(X)
        return {
            "stage1_input": X,
            "stage2_latent": preds["latent"],
            "all_hidden": self.activations[1:],
            "stage3_outputs": preds
        }

    def save_weights(self, filepath):
        """Save model parameters, architecture, and optimizer state to an NPZ file."""
        save_dict = {
            "input_dim": np.array(self.input_dim),
            "hidden_sizes": np.array(self.hidden_sizes),
            "learning_rate": np.array(self.learning_rate),
            "l2_reg": np.array(self.l2_reg),
            "t": np.array(self.t),
        }
        for i, (W, b) in enumerate(zip(self.W_shared, self.b_shared)):
            save_dict[f"W_shared_{i}"] = W
            save_dict[f"b_shared_{i}"] = b
            save_dict[f"m_w_shared_{i}"] = self.m_w_shared[i]
            save_dict[f"v_w_shared_{i}"] = self.v_w_shared[i]
            save_dict[f"m_b_shared_{i}"] = self.m_b_shared[i]
            save_dict[f"v_b_shared_{i}"] = self.v_b_shared[i]

        for head in ["cardiac", "diabetes", "life", "vascular"]:
            save_dict[f"W_head_{head}"] = self.W_heads[head]
            save_dict[f"b_head_{head}"] = self.b_heads[head]
            save_dict[f"m_w_head_{head}"] = self.m_w_heads[head]
            save_dict[f"v_w_head_{head}"] = self.v_w_heads[head]
            save_dict[f"m_b_head_{head}"] = self.m_b_heads[head]
            save_dict[f"v_b_head_{head}"] = self.v_b_heads[head]

        np.savez_compressed(filepath, **save_dict)

    def load_weights(self, filepath):
        """Load model weights from an NPZ file into current instance."""
        data = np.load(filepath, allow_pickle=True)
        self.input_dim = int(data["input_dim"])
        self.hidden_sizes = list(data["hidden_sizes"])
        self.learning_rate = float(data["learning_rate"])
        self.l2_reg = float(data["l2_reg"])
        self.t = int(data["t"])

        self.W_shared = []
        self.b_shared = []
        self.m_w_shared, self.v_w_shared = [], []
        self.m_b_shared, self.v_b_shared = [], []

        for i in range(len(self.hidden_sizes)):
            self.W_shared.append(data[f"W_shared_{i}"])
            self.b_shared.append(data[f"b_shared_{i}"])
            self.m_w_shared.append(data[f"m_w_shared_{i}"])
            self.v_w_shared.append(data[f"v_w_shared_{i}"])
            self.m_b_shared.append(data[f"m_b_shared_{i}"])
            self.v_b_shared.append(data[f"v_b_shared_{i}"])

        for head in ["cardiac", "diabetes", "life", "vascular"]:
            self.W_heads[head] = data[f"W_head_{head}"]
            self.b_heads[head] = data[f"b_head_{head}"]
            self.m_w_heads[head] = data[f"m_w_head_{head}"]
            self.v_w_heads[head] = data[f"v_w_head_{head}"]
            self.m_b_heads[head] = data[f"m_b_head_{head}"]
            self.v_b_heads[head] = data[f"v_b_head_{head}"]

    @classmethod
    def load_model(cls, filepath):
        """Class method to instantiate and load a saved model from an NPZ file."""
        data = np.load(filepath, allow_pickle=True)
        hidden_sizes = list(data["hidden_sizes"])
        lr = float(data["learning_rate"])
        l2 = float(data["l2_reg"])
        inp = int(data["input_dim"])
        nn = cls(input_dim=inp, hidden_sizes=hidden_sizes, learning_rate=lr, l2_reg=l2)
        nn.load_weights(filepath)
        return nn

    def predict_with_uncertainty(self, X, n_samples=30, dropout_rate=0.1, seed=42):
        """
        Monte Carlo Dropout Uncertainty Estimation:
        Performs N stochastic forward passes with active dropout masks to sample
        the model's predictive distribution. Returns mean (mu), std (sigma), and 95% CI.
        """
        rng = np.random.default_rng(seed)
        c_list, d_list, l_list, v_list = [], [], [], []

        for _ in range(n_samples):
            # Stochastic forward pass with Bernoulli dropout mask on shared layers
            curr_A = X
            for i in range(len(self.hidden_sizes)):
                Z = np.dot(curr_A, self.W_shared[i]) + self.b_shared[i]
                curr_A = self.act_func(Z)
                mask = (rng.uniform(0, 1, size=curr_A.shape) >= dropout_rate).astype(float)
                curr_A = (curr_A * mask) / (1.0 - dropout_rate)

            latent = curr_A
            p_c = sigmoid(np.dot(latent, self.W_heads["cardiac"]) + self.b_heads["cardiac"])
            p_d = sigmoid(np.dot(latent, self.W_heads["diabetes"]) + self.b_heads["diabetes"])
            pred_l = np.dot(latent, self.W_heads["life"]) + self.b_heads["life"]
            pred_v = np.dot(latent, self.W_heads["vascular"]) + self.b_heads["vascular"]

            c_list.append(p_c)
            d_list.append(p_d)
            l_list.append(pred_l)
            v_list.append(pred_v)

        c_arr = np.array(c_list)
        d_arr = np.array(d_list)
        l_arr = np.array(l_list)
        v_arr = np.array(v_list)

        results = {}
        for head, arr in [("cardiac", c_arr), ("diabetes", d_arr), ("life", l_arr), ("vascular", v_arr)]:
            mean = np.mean(arr, axis=0)
            std = np.std(arr, axis=0)
            ci95 = 1.96 * std
            results[head] = {
                "mean": mean,
                "std": std,
                "ci95": ci95,
                "low95": mean - ci95,
                "high95": mean + ci95
            }
        return results


