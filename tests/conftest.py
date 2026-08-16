from __future__ import annotations

import pytest

from backend.app import app


@pytest.fixture(autouse=True)
def tmp_project_root(monkeypatch, tmp_path):
    import backend.config

    root = tmp_path / "hmresnet-web"
    assets = root / "model_assets" / "resnet"
    assets.mkdir(parents=True)
    (assets / "labels.json").write_text("[]", encoding="utf-8")
    (assets / "amino_acid_to_index.json").write_text("{}", encoding="utf-8")
    (assets / "best_model.pth").write_bytes(b"")

    monkeypatch.setattr(backend.config.Settings, "project_root", root)
    monkeypatch.setattr(backend.config, "settings", backend.config.Settings())
    yield root


@pytest.fixture
def model_service_mock(monkeypatch):
    from unittest.mock import MagicMock

    mock = MagicMock()
    mock.model = MagicMock()
    monkeypatch.setattr("backend.app.ModelService", lambda *args, **kwargs: mock)
    return mock

