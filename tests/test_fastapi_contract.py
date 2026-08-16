from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app import app as hmresnet_app


EXPECTED_DOI = "10.5281/zenodo.14600722"
EXPECTED_LABEL_COUNT = 23


def test_api_contract(model_service_mock):
    model_service_mock.labels = [f"label_{i}" for i in range(EXPECTED_LABEL_COUNT)]
    model_service_mock.description.return_value = {
        "name": "HMResNet-1D",
        "doi": EXPECTED_DOI,
        "label_count": EXPECTED_LABEL_COUNT,
        "labels": model_service_mock.labels,
        "max_length": 1024,
        "device": "cpu",
        "loaded": True,
    }
    model_service_mock.predict.return_value = {
        "predictions": [
            {
                "id": "seq_1",
                "original_length": 60,
                "model_length": 60,
                "removed_residues": 0,
                "truncated": False,
                "labels": [{"label": "label_3", "probability": 0.87}],
                "top_scores": [
                    {"label": "label_3", "probability": 0.87},
                    {"label": "label_7", "probability": 0.32},
                    {"label": "label_9", "probability": 0.24},
                ],
            }
        ],
        "metadata": {
            "model": "HMResNet-1D",
            "doi": EXPECTED_DOI,
            "device": "cpu",
            "threshold": 0.5,
            "top_k": 3,
            "max_length": 1024,
            "runtime_ms": 12.0,
        },
    }

    with TestClient(hmresnet_app) as client:
        health = client.get("/api/health").json()
        assert health["status"] == "ok"

        model_info = client.get("/api/model").json()
        assert model_info["doi"] == EXPECTED_DOI
        assert model_info["label_count"] == EXPECTED_LABEL_COUNT

        payload = {
            "fasta": ">seq_1\nMKKVIYFLCTGNSCRSQMAEGWAKKYLGDEWEVYSAGIEAHGLNPNAVKAMKEIGIDISNQTSDV",
            "threshold": 0.5,
            "top_k": 3,
        }
        response = client.post("/api/predict", json=payload).json()
        assert response["predictions"][0]["labels"][0]["label"] == "label_3"
        assert response["metadata"]["doi"] == EXPECTED_DOI
