
HMResNet Web

[Online prediction website](https://fgaresnet.duckdns.org/hmresnet/) ·
[Zenodo source record](https://doi.org/10.5281/zenodo.14600722)

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14600722.svg)](https://doi.org/10.5281/zenodo.14600722)

Protein metal-resistance multi-label prediction service based on the Zenodo record
[10.5281/zenodo.14600722](https://doi.org/10.5281/zenodo.14600722).


## Features

- Static HTML page for FASTA input, thresholds, and result export.
- FastAPI backend with CPU/CUDA inference.
- Original Zenodo source code preserved under vendor/zenodo/resnet/.
- Lightweight model package in model_assets/resnet/.
- Tencent Cloud / DuckDNS deployment notes.


## What this project predicts

The ResNet-1D model scores each protein sequence against 23 metal categories.
This is a screening tool, not a replacement for experimental validation.

## Links

- Prediction website: <https://fgaresnet.duckdns.org/hmresnet/>
- Zenodo record: <https://doi.org/10.5281/zenodo.14600722>

## How to cite

If you use the model or prediction service, please cite the associated article:

> Chen, J., Gao, X., Zhang, C., & Ge, Y. (2025). Rapid identification of metal
> resistance genes using an enhanced ResNet deep learning model trained on a
> largely expanded BacMet-based database. *Journal of Hazardous Materials*,
> 497, 139625. https://doi.org/10.1016/j.jhazmat.2025.139625

Please also cite the released software and model record:

> Chen, Jiale. (2025). *Rapid Identification of Metal Resistance Genes Using
> Enhanced ResNet Deep Learning Model*. Zenodo. https://doi.org/10.5281/zenodo.14600722

The scope of the repository license is documented in `LICENSE` and
`THIRD_PARTY_NOTICES.md`.

## License

The original application, integration, user-interface, and deployment code in
this repository is released under the [MIT License](LICENSE). The MIT License
does not automatically apply to the model weights, `vendor/zenodo/`, or other
third-party material; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).


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

The live deployment uses the existing Tencent Cloud relay and is available at:

```text
https://fgaresnet.duckdns.org/hmresnet/
```

The existing FGA-DB route at `/` is left untouched. The deployment pattern is:

- Tencent Cloud Nginx on 80/443
- SSH reverse tunnel from the GPU host to Tencent Cloud loopback port 18012
- local CUDA prediction service on `127.0.0.1:8011`
- DuckDNS for the existing public hostname

See `deploy/tencent-cloud/deploy.md` and
`deploy/tencent-cloud/nginx/hmresnet-path.conf` for the exact configuration.


## Release notes

The Zenodo record page did not expose an explicit license when checked on
2026-08-16. Please verify redistribution terms before public release.
