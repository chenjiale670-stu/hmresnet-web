from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _csv_env(name: str, default: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in os.environ.get(name, default).split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    frontend_dir: Path = PROJECT_ROOT / "frontend"
    model_dir: Path = Path(
        os.environ.get("HMRESNET_MODEL_DIR", PROJECT_ROOT / "model_assets" / "resnet")
    ).resolve()
    device: str = os.environ.get("HMRESNET_DEVICE", "auto")
    batch_size: int = int(os.environ.get("HMRESNET_BATCH_SIZE", "16"))
    max_records: int = int(os.environ.get("HMRESNET_MAX_RECORDS", "50"))
    max_request_bytes: int = int(os.environ.get("HMRESNET_MAX_REQUEST_BYTES", "1000000"))
    max_concurrent_predictions: int = int(os.environ.get("HMRESNET_MAX_CONCURRENT", "1"))
    cors_origins: tuple[str, ...] = _csv_env(
        "HMRESNET_CORS_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000",
    )


settings = Settings()
