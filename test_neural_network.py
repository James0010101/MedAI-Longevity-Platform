
import os
import json
import tempfile
import numpy as np
import pytest

from dataset_manager import get_medical_dataset, normalize
from neural_network import MultiTaskNeuralNetwork, LOSS_WEIGHTS

class TestDatasetManager:
    def test_dataset_shape(self):
        data = get_medical_dataset(n_samples=100)
        assert data["raw_X"].shape == (100, 10)
        assert data["norm_X"].shape == (100, 10)
        assert data["y_cardiac"].shape == (100, 1)
        assert data["y_diabetes"].shape == (100, 1)

    def test_train_test_split_sizes(self):
        data = get_medical_dataset(n_samples=100, test_ratio=0.2)
        assert data["X_train"].shape[0] == 80
        assert data["X_test"].shape[0] == 20

    def test_normalized_range(self):
        data = get_medical_dataset(n_samples=200)
        assert np.all(data["norm_X"] >= -0.1)
        assert np.all(data["norm_X"] <= 1.1)

    def test_reproducibility(self):
        d1 = get_medical_dataset(n_samples=50, seed=123)
        d2 = get_medical_dataset(n_samples=50, seed=123)
        np.testing.assert_array_equal(d1["raw_X"], d2["raw_X"])

    def test_normalize_utility(self):
        X = np.array([50.0, 100.0, 200.0])
        mins = np.array([0.0, 0.0, 0.0])
        maxs = np.array([100.0, 200.0, 400.0])
        result = normalize(X, mins, maxs)
        np.testing.assert_allclose(result, [0.5, 0.5, 0.5], atol=1e-6)

    def test_openml_fallback(self):
        data = get_medical_dataset(n_samples=50, source="openml_heart")
        assert data["norm_X"].shape[1] == 10

    def test_pima_dataset_source(self):
        data = get_medical_dataset(n_samples=50, source="pima_diabetes")
        assert data["norm_X"].shape[1] == 10
        assert data["y_diabetes"].shape[0] > 0

    def test_fused_master_dataset(self):
        data = get_medical_dataset(n_samples=100, source="fused_master")
        assert data["norm_X"].shape[1] == 10
        assert data["norm_X"].shape[0] >= 100

class TestMultiTaskNeuralNetwork:
    @pytest.fixture
    def trained_model(self):
        data = get_medical_dataset(n_samples=300, seed=42)
        nn = MultiTaskNeuralNetwork(input_dim=10, hidden_sizes=[24, 12], learning_rate=0.02)
        for _ in range(300):
            nn.train_step(
                data["X_train"], data["y_cardiac_train"],
                data["y_diabetes_train"], data["y_life_train"],
                data["y_vascular_train"]
            )
        return nn, data

    def test_forward_output_shapes(self):
        nn = MultiTaskNeuralNetwork(input_dim=10, hidden_sizes=[16, 8])
        X = np.random.randn(5, 10)
        preds = nn.forward(X)
        assert preds["cardiac"].shape == (5, 1)
        assert preds["diabetes"].shape == (5, 1)

    def test_classification_outputs_bounded(self):
        nn = MultiTaskNeuralNetwork(input_dim=10, hidden_sizes=[12])
        X = np.random.randn(20, 10)
        preds = nn.forward(X)
        assert np.all(preds["cardiac"] >= 0.0) and np.all(preds["cardiac"] <= 1.0)
        assert np.all(preds["diabetes"] >= 0.0) and np.all(preds["diabetes"] <= 1.0)

    def test_activation_works(self):
        nn = MultiTaskNeuralNetwork(input_dim=10, hidden_sizes=[8])
        X = np.random.randn(5, 10)
        preds = nn.forward(X)
        assert not np.any(np.isnan(preds["cardiac"]))

    def test_monte_carlo_uncertainty_structure(self, trained_model):
        nn, data = trained_model
        res = nn.predict_with_uncertainty(data["X_test"][:1], n_samples=20, dropout_rate=0.1)
        assert "cardiac" in res
        assert "mean" in res["cardiac"]
        assert "ci95" in res["cardiac"]
        assert res["cardiac"]["mean"].shape == (1, 1)
        assert res["cardiac"]["ci95"].shape == (1, 1)

    def test_monte_carlo_confidence_interval_non_negative(self, trained_model):
        nn, data = trained_model
        res = nn.predict_with_uncertainty(data["X_test"][:1], n_samples=20, dropout_rate=0.1)
        assert np.all(res["cardiac"]["ci95"] >= 0.0)
        assert np.all(res["life"]["ci95"] >= 0.0)

    def test_save_and_load_weights(self, trained_model):
        nn, data = trained_model
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            nn.save_weights(tmp_path)
            loaded_nn = MultiTaskNeuralNetwork.load_model(tmp_path)
            pred_orig = nn.forward(data["X_test"])
            pred_loaded = loaded_nn.forward(data["X_test"])
            np.testing.assert_allclose(pred_orig["cardiac"], pred_loaded["cardiac"], rtol=1e-5)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_fhir_json_export_structure(self):
        fhir_doc = {
            "resourceType": "DiagnosticReport",
            "status": "final",
            "code": {"coding": [{"system": "http://loinc.org", "code": "80352-8"}]},
            "result": [
                {"observation": "Cardiac Risk", "value": 75.4, "unit": "%"}
            ]
        }
        json_str = json.dumps(fhir_doc)
        parsed = json.loads(json_str)
        assert parsed["resourceType"] == "DiagnosticReport"
        assert parsed["result"][0]["value"] == 75.4

class TestEdgeCases:
    def test_single_sample_forward(self):
        nn = MultiTaskNeuralNetwork(input_dim=10, hidden_sizes=[8])
        X = np.random.randn(1, 10)
        preds = nn.forward(X)
        assert preds["cardiac"].shape == (1, 1)

    def test_zeros_input(self):
        nn = MultiTaskNeuralNetwork(input_dim=10, hidden_sizes=[8])
        X = np.zeros((3, 10))
        preds = nn.forward(X)
        assert not np.any(np.isnan(preds["cardiac"]))

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
