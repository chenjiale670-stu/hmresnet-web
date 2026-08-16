from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .fasta import ProteinRecord


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        dropout_prob: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        identity = value
        output = self.dropout(self.relu(self.bn1(self.conv1(value))))
        output = self.bn2(self.conv2(output))
        if self.downsample is not None:
            identity = self.downsample(value)
        return self.relu(output + identity)


class ResNet1D(nn.Module):
    def __init__(
        self,
        block: type[BasicBlock1D],
        layers: list[int],
        embedding_dim: int,
        num_classes: int,
        dropout_prob: float = 0.3,
    ) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(embedding_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], dropout_prob=dropout_prob)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dropout_prob=dropout_prob)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dropout_prob=dropout_prob)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dropout_prob=dropout_prob)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: type[BasicBlock1D],
        out_channels: int,
        blocks: int,
        stride: int = 1,
        dropout_prob: float = 0.3,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )
        modules: list[nn.Module] = [block(self.in_channels, out_channels, stride, downsample, dropout_prob)]
        self.in_channels = out_channels * block.expansion
        modules.extend(block(self.in_channels, out_channels, dropout_prob=dropout_prob) for _ in range(1, blocks))
        return nn.Sequential(*modules)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.maxpool(self.dropout1(self.relu(self.bn1(self.conv1(value)))))
        value = self.layer4(self.layer3(self.layer2(self.layer1(value))))
        return self.fc(torch.flatten(self.avgpool(value), 1))


class ResNet1DEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.resnet = ResNet1D(BasicBlock1D, [2, 2, 2, 2], embedding_dim, num_classes=num_classes, dropout_prob=0.3)
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.resnet(self.embedding(value).permute(0, 2, 1))


class ModelService:
    max_length = 1024
    embedding_dim = 128
    model_name = "HMResNet-1D"
    doi = "10.5281/zenodo.14600722"

    def __init__(self, model_dir: Path, device: str = "auto", batch_size: int = 16) -> None:
        self.model_dir = model_dir
        self.requested_device = device
        self.batch_size = max(1, batch_size)
        self.device = torch.device("cpu")
        self.labels: list[str] = []
        self.aa_to_index: dict[str, int] = {}
        self.model: ResNet1DEmbedding | None = None
        self.loaded_at: float | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        labels_path = self.model_dir / "labels.json"
        vocabulary_path = self.model_dir / "amino_acid_to_index.json"
        checkpoint_path = self.model_dir / "best_model.pth"
        missing = [str(path.name) for path in (labels_path, vocabulary_path, checkpoint_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"模型文件缺失: {', '.join(missing)}")

        self.labels = json.loads(labels_path.read_text(encoding="utf-8"))
        self.aa_to_index = json.loads(vocabulary_path.read_text(encoding="utf-8"))
        self.device = self._resolve_device(self.requested_device)
        model = ResNet1DEmbedding(
            vocab_size=len(self.aa_to_index) + 1,
            embedding_dim=self.embedding_dim,
            num_classes=len(self.labels),
        )
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            raise RuntimeError("Checkpoint must be a state-dict")
        model.load_state_dict(state, strict=True)
        model.to(self.device)
        model.eval()
        self.model = model
        self.loaded_at = time.time()

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"请求了设备 {requested}，但当前 PyTorch 无可用 CUDA")
        return device

    def _encode(self, record: ProteinRecord) -> list[int]:
        encoded = [self.aa_to_index.get(amino_acid, 0) for amino_acid in record.sequence]
        encoded = encoded[: self.max_length]
        return encoded + [0] * (self.max_length - len(encoded))

    def predict(
        self, records: list[ProteinRecord], *, threshold: float = 0.5, top_k: int = 5
    ) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("模型尚未加载")
        started = time.perf_counter()
        encoded = torch.tensor([self._encode(record) for record in records], dtype=torch.long)
        probability_parts: list[torch.Tensor] = []

        with self._lock, torch.inference_mode():
            for start in range(0, len(encoded), self.batch_size):
                batch = encoded[start : start + self.batch_size].to(self.device, non_blocking=True)
                probability_parts.append(torch.sigmoid(self.model(batch)).cpu())
        probabilities = torch.cat(probability_parts, dim=0)

        predictions: list[dict[str, Any]] = []
        for record, scores in zip(records, probabilities.tolist(), strict=True):
            ranked = sorted(
                (
                    {"label": label, "probability": round(float(score), 6)}
                    for label, score in zip(self.labels, scores, strict=True)
                ),
                key=lambda item: item["probability"],
                reverse=True,
            )
            predictions.append(
                {
                    "id": record.identifier,
                    "original_length": record.original_length,
                    "model_length": min(len(record.sequence), self.max_length),
                    "removed_residues": record.removed_residues,
                    "truncated": len(record.sequence) > self.max_length,
                    "labels": [item for item in ranked if item["probability"] >= threshold],
                    "top_scores": ranked[:top_k],
                }
            )

        return {
            "predictions": predictions,
            "metadata": {
                "model": self.model_name,
                "doi": self.doi,
                "device": str(self.device),
                "threshold": threshold,
                "top_k": top_k,
                "max_length": self.max_length,
                "runtime_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        }

    def description(self) -> dict[str, Any]:
        return {
            "name": self.model_name,
            "doi": self.doi,
            "architecture": "ResNet-18 1D + amino-acid embedding",
            "task": "protein metal-resistance multi-label prediction",
            "label_count": len(self.labels),
            "labels": self.labels,
            "max_length": self.max_length,
            "device": str(self.device),
            "loaded": self.model is not None,
        }
