
HMResNet Web

Protein metal-resistance multi-label prediction service based on the Zenodo record
doi:10.5281/zenodo.14600722.


## Features

- Static HTML page for FASTA input, thresholds, and result export.
- FastAPI backend with CPU/CUDA inference.
- Original Zenodo source code preserved under vendor/zenodo/resnet/.
- Lightweight model package in model_assets/resnet/.
- Tencent Cloud / DuckDNS deployment notes.


## What this project predicts

The ResNet-1D model scores each protein sequence against 23 metal categories.
This is a screening tool, not a replacement for experimental validation.


## Local setup

Preferred environment: the existing pytorch_stable conda environment.


```bash
conda activate pytorch_stable
cd hmresnet-web
pip install -r requirements.txt
```


## Local run


```bash
cd hmresnet-web
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

If PyTorch has no usable CUDA device, add:


```bash
export HMRESNET_DEVICE=cpu
```

Then open:


```text
http://127.0.0.1:8000
```


## Rebuilding model assets from Zenodo


```bash
python scripts/download_model.py --out model_assets/resnet
```

This fetches the Zenodo archive, checks its MD5, extracts the ResNet model and
vocabulary artifacts, and writes JSON replacements for labels and vocabulary.


## Deployment

Use the files in deploy/tencent-cloud/ as a starting point.
The recommended pattern is:

- Tencent Cloud Nginx on 80/443
- NPS/NPC tunnel back to the GPU host
- local gateway and prediction service on loopback
- DuckDNS for the public hostname


## Release notes

The Zenodo record page did not expose an explicit license when checked on
2026-08-16. Please verify redistribution terms before public release.
