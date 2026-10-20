"""
Unit tests for neural_network.py
"""
import numpy as np
import pytest
from neural_network import MultiTaskNeuralNetwork

def test_model_initialization_and_forward():
    model = MultiTaskNeuralNetwork(input_dim=13, hidden_sizes=(32, 16), heads=("chd", "diabetes"), seed=42)
    X = np.random.normal(0, 1, size=(10, 13)).astype(np.float32)

    probs = model.predict(X)
    assert "chd" in probs
    assert "diabetes" in probs
    assert probs["chd"].shape == (10, 1)
    assert probs["diabetes"].shape == (10, 1)
    assert np.all(probs["chd"] >= 0.0) and np.all(probs["chd"] <= 1.0)
    assert np.all(probs["diabetes"] >= 0.0) and np.all(probs["diabetes"] <= 1.0)

def test_deterministic_inference():
    model = MultiTaskNeuralNetwork(input_dim=13, hidden_sizes=(16, 8), dropout=0.5, seed=123)
    X = np.random.normal(0, 1, size=(5, 13)).astype(np.float32)

    p1 = model.predict(X)["chd"]
    p2 = model.predict(X)["chd"]
    np.testing.assert_array_equal(p1, p2)

def test_input_gradient_vs_finite_difference():
    model = MultiTaskNeuralNetwork(input_dim=6, hidden_sizes=(16, 8), heads=("chd",), seed=777)
    x = np.array([[0.5, -0.3, 1.2, -0.8, 0.1, -0.5]], dtype=np.float64)

    # Analytical gradient
    grad_analytic = model.input_gradient(x, head="chd")
    assert grad_analytic.shape == (6,)

    # Finite difference gradient: (f(x + eps) - f(x - eps)) / (2 * eps)
    eps = 1e-6
    grad_num = np.zeros(6, dtype=np.float64)
    for i in range(6):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[0, i] += eps
        x_minus[0, i] -= eps
        p_plus = float(model.predict(x_plus)["chd"][0, 0])
        p_minus = float(model.predict(x_minus)["chd"][0, 0])
        grad_num[i] = (p_plus - p_minus) / (2.0 * eps)

    # Check relative error: ||g_analytic - g_num|| / (||g_num|| + eps)
    abs_diff = np.abs(grad_analytic - grad_num)
    rel_error = abs_diff / (np.abs(grad_num) + 1e-7)
    max_rel_error = np.max(rel_error)

    print(f"Max relative gradient error: {max_rel_error:.2e}")
    assert max_rel_error < 1e-4

def test_save_load_roundtrip(tmp_path):
    model = MultiTaskNeuralNetwork(input_dim=8, hidden_sizes=(12, 6), heads=("chd", "diabetes"), seed=99)
    metadata = {
        "feature_names": ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"],
        "means": [0.0] * 8,
        "stds": [1.0] * 8,
        "medians": [0.0] * 8,
    }

    save_path = tmp_path / "test_weights.npz"
    model.save(save_path, metadata=metadata)

    loaded_model = MultiTaskNeuralNetwork.load(save_path)
    assert loaded_model.input_dim == model.input_dim
    assert loaded_model.hidden_sizes == model.hidden_sizes
    assert loaded_model.heads == model.heads
    assert loaded_model.metadata == metadata

    X_test = np.random.normal(0, 1, size=(4, 8)).astype(np.float32)
    orig_preds = model.predict(X_test)
    loaded_preds = loaded_model.predict(X_test)

    np.testing.assert_allclose(orig_preds["chd"], loaded_preds["chd"], rtol=1e-6)
    np.testing.assert_allclose(orig_preds["diabetes"], loaded_preds["diabetes"], rtol=1e-6)

def test_train_step_loss_decreases():
    model = MultiTaskNeuralNetwork(input_dim=4, hidden_sizes=(16, 8), lr=0.05, l2_reg=0.0, seed=42)
    X = np.random.normal(0, 1, size=(64, 4)).astype(np.float32)
    y_chd = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
    y_dm = (X[:, 2] > 0.5).astype(np.float32)

    initial_loss = model.compute_loss(model.predict(X), {"chd": y_chd, "diabetes": y_dm})

    for _ in range(50):
        model.train_step(X, {"chd": y_chd, "diabetes": y_dm})

    final_loss = model.compute_loss(model.predict(X), {"chd": y_chd, "diabetes": y_dm})
    assert final_loss < initial_loss

def test_predict_with_uncertainty():
    model = MultiTaskNeuralNetwork(input_dim=6, hidden_sizes=(16, 8), dropout=0.2, seed=42)
    X = np.random.normal(0, 1, size=(5, 6)).astype(np.float32)

    unc1 = model.predict_with_uncertainty(X, n_mc=20, seed=10)
    unc2 = model.predict_with_uncertainty(X, n_mc=20, seed=10)

    assert "chd" in unc1 and "diabetes" in unc1
    assert "mean" in unc1["chd"] and "std" in unc1["chd"]
    # Deterministic with fixed seed
    np.testing.assert_array_equal(unc1["chd"]["mean"], unc2["chd"]["mean"])
    np.testing.assert_array_equal(unc1["chd"]["std"], unc2["chd"]["std"])
    # std should be non-negative
    assert np.all(unc1["chd"]["std"] >= 0.0)

def test_integrated_gradients_completeness():
    model = MultiTaskNeuralNetwork(input_dim=6, hidden_sizes=(16, 8), seed=42)
    baseline = np.zeros(6, dtype=np.float64)
    x = np.array([0.5, -0.3, 0.8, -0.2, 0.4, 0.1], dtype=np.float64)

    attr = model.integrated_gradients(x, baseline, head="chd", m_steps=100)
    assert attr.shape == (6,)

    p_x = float(model.predict(x.reshape(1, -1))["chd"][0, 0])
    p_base = float(model.predict(baseline.reshape(1, -1))["chd"][0, 0])
    target_diff = p_x - p_base

    # Completeness axiom: sum of attributions equals difference in probability within numerical tolerance
    attr_sum = float(np.sum(attr))
    print(f"attr_sum={attr_sum:.6f}, target_diff={target_diff:.6f}")
    assert np.isclose(attr_sum, target_diff, atol=1e-3)

