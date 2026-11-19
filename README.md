# Ukrainian passport perforation OCR

This repository contains the inference code and the original Meta DINOv2 ViT-S/14 encoder weights. The task is to read two Cyrillic series letters and six digits from the perforated field of a Ukrainian internal passport booklet. Full-page OCR requires two additional project-specific checkpoints. Their commercial distribution status is unresolved.

## Files and terms

| Component | Terms and evidence |
|---|---|
| Project inference code and documentation | MIT. The original repository maintainer `askakalsky` set `license: mit` in commit `1f09f27` on April 14, 2026. Retain the included LICENSE when redistributing. |
| `data/external_cache/dinov2_vits14_pretrain.pth` | Unmodified DINOv2 ViT-S/14 encoder, Meta Platforms, Inc. and affiliates. Apache-2.0. License included in `licenses/DINOv2-Apache-2.0.txt`. SHA-256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`. |
| Runtime dependencies | Installed separately. The requirements file identifies the direct Python dependencies. Review the exact wheels and their bundled native libraries if distributing a runtime. |
| Local detector head and CRNN | The reference hashes and evidence gaps are in MODEL_CARD.md. The external bundle contains only the base encoder weights. |

The encoder publisher identifies the ordinary DINOv2 models and their terms in the [pinned upstream README](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8#license). The [original checkpoint](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) can also be obtained directly.

## Run

Use Git and Python 3.11. From the repository root, install the Python packages into a virtual environment. Clone the official implementation at the revision below. The loader verifies that revision, the tracked checkout and the encoder weight digest before use.

```shell
python -m pip install -r requirements.txt
git clone https://github.com/facebookresearch/dinov2.git data/external_cache/dinov2
git -C data/external_cache/dinov2 checkout --detach 7764ea0f912e53c92e82eb78a2a1631e92725fc8
python -m predict --help
```

To test the encoder alone, run the following from the same directory.

```shell
python -c "import torch; from src.localization.dinov2 import load_encoder; print(type(load_encoder(torch.device('cpu'))).__name__)"
```

With a compatible, authorized detector head and reader, run

```shell
python -m predict page.png --detector private-weights/detector.pth --reader private-weights/reader.pth --output result.json
```

Output coordinates are pixel edges in the decoded image, ordered left, top, right, bottom. The result includes the full prediction, character scores and review flags. Read both `geometry_review` (if present) and `readable`. Keep the resulting identifiers in the same private environment as the input documents.
