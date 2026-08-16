from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import tempfile
import zipfile
from pathlib import Path


DEFAULT_OUTPUT = Path("model_assets/resnet")
ZENDO_DOI = "10.5281/zenodo.14600722"
EXPECTED_MD5 = "ccc5e276e60637928e81e0105e1b3dc7"
EXPECTED_SIZE = 522_231_243


def _download_file(url: str, target: Path) -> None:
    try:
        import httpx
    except ImportError as exc:
        raise SystemExit("Download script needs httpx. Install it with: pip install httpx") from exc
    with httpx.stream("GET", url, timeout=600.0) as response:
        response.raise_for_status()
        size = int(response.headers.get("content-length", "0") or 0)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            shutil.copyfileobj(response.iter_raw(), handle)
    if size and target.stat().st_size != size:
        raise RuntimeError(f"Downloaded size mismatch: got {target.stat().st_size}, expected {size}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and package the Zenodo ResNet model assets.")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT), help="Output model directory")
    parser.add_argument("--cache-dir", default=Path(tempfile.gettempdir()) / "hmresnet-download", help="Temporary download directory")
    parser.add_argument("--archive-url", default="https://zenodo.org/records/14600722/files/code.zip", help="Direct archive URL")
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    archive = cache_dir / "code.zip"
    extract_root = cache_dir / "zenodo_extract"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download] fetching {ZENDO_DOI} archive")
    _download_file(args.archive_url, archive)
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    if digest != EXPECTED_MD5:
        raise SystemExit(f"MD5 mismatch for {archive}: {digest}")
    if archive.stat().st_size != EXPECTED_SIZE:
        print(f"[warn] archive size differs from local record: {archive.stat().st_size} vs {EXPECTED_SIZE}")

    print("[download] extracting archive")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(extract_root)

    zenodo_resnet = extract_root / "zenodo" / "resnet"
    if not zenodo_resnet.is_dir():
        raise SystemExit(f"Expected directory not found: {zenodo_resnet}")

    for name in ["best_model.pth", "mlb.pkl", "amino_acid_to_index.pkl"]:
        shutil.copy2(zenodo_resnet / name, out_dir / name)

    with (zenodo_resnet / "mlb.pkl").open("rb") as handle:
        mlb = pickle.load(handle)
    with (zenodo_resnet / "amino_acid_to_index.pkl").open("rb") as handle:
        amino_acid_to_index = pickle.load(handle)

    (out_dir / "labels.json").write_text(
        json.dumps(list(map(str, mlb.classes_)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "amino_acid_to_index.json").write_text(
        json.dumps(amino_acid_to_index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "source_doi": ZENDO_DOI,
        "source_record": "https://zenodo.org/records/14600722",
        "source_archive": {"name": "code.zip", "size_bytes": EXPECTED_SIZE, "md5": EXPECTED_MD5},
        "files": {
            "best_model.pth": {"sha256": _sha256(out_dir / "best_model.pth")},
            "mlb.pkl": {"sha256": _sha256(out_dir / "mlb.pkl")},
            "amino_acid_to_index.pkl": {"sha256": _sha256(out_dir / "amino_acid_to_index.pkl")},
            "labels.json": {"purpose": "JSON export of MultiLabelBinarizer classes"},
            "amino_acid_to_index.json": {"purpose": "JSON export of amino-acid vocabulary"},
        },
        "note": "The Zenodo record page did not display an explicit license when checked on 2026-08-16. Verify redistribution terms before public release.",
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote model assets to {out_dir}")


if __name__ == "__main__":
    main()

