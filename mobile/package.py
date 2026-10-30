#!/usr/bin/env python3
"""Package a Ghana Voice Nano bundle for a sherpa-onnx mobile/web app.

    python mobile/package.py --model ghanaopenai/ghana-voice-nano-twi --variant steps4 --out dist/

Writes:
  dist/<name>/                    flat folder: drop into Android app/src/main/assets/, an iOS
                                  bundle, Flutter assets/ or the sherpa-onnx WASM assets dir
  dist/<name>.zip                 the same folder zipped (for download-on-first-launch apps)
  dist/<name>/sherpa-config.json  file names + settings for the app's OfflineTtsMatchaModelConfig

Only one acoustic variant is included to keep the app small (~130 MB with steps4 or steps2;
the two are the same size, steps2 is faster).
"""
import argparse
import json
import os
import shutil
from pathlib import Path


def get_args():
    p = argparse.ArgumentParser(description="Package a model bundle for sherpa-onnx apps")
    p.add_argument("--model", default="ghanaopenai/ghana-voice-nano-twi", help="HF repo id or local bundle")
    p.add_argument("--variant", default=None, help="acoustic variant to ship (default: the bundle default)")
    p.add_argument("--out", default="dist")
    p.add_argument("--name", default=None, help="output folder name (default: derived from the model)")
    return p.parse_args()


def main():
    a = get_args()
    if os.path.isdir(a.model):
        src = Path(a.model)
    else:
        from huggingface_hub import snapshot_download

        src = Path(snapshot_download(a.model))
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    variant = a.variant or cfg["default_acoustic"]
    acoustic = cfg["acoustic_models"][variant]
    name = a.name or f"{Path(a.model.rstrip('/')).name}-{variant}"

    dst = Path(a.out) / name
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in [acoustic, cfg["vocoder"], cfg["tokens"]]:
        shutil.copy(src / f, dst / f)
    shutil.copytree(src / cfg["data_dir"], dst / cfg["data_dir"])

    app_cfg = {
        "acousticModel": acoustic,
        "vocoder": cfg["vocoder"],
        "tokens": cfg["tokens"],
        "dataDir": cfg["data_dir"],
        "lexicon": "",
        "noiseScale": cfg["noise_scale"],
        "lengthScale": 1.0,
        "sampleRate": cfg["sample_rate"],
        "language": cfg["language"],
        "sherpaOnnxVersion": ">=1.12,<2.0 (2.x dropped the espeak-ng frontend for Matcha models)",
    }
    (dst / "sherpa-config.json").write_text(json.dumps(app_cfg, indent=2, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
    archive = shutil.make_archive(str(dst), "zip", root_dir=dst.parent, base_dir=dst.name)
    size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 2**20
    print(f"packaged {dst} ({size:.0f} MB) and {archive}")
    print(json.dumps(app_cfg, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
