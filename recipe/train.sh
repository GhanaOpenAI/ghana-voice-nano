#!/usr/bin/env bash
# Train a Ghana Voice Nano voice on data written by recipe/prep_data.py.
#
#   recipe/train.sh <data_dir> <warm_start.ckpt> [extra hydra overrides...]
#
# Warm start: for a Ghanaian language, start from the Twi checkpoint
# (hf download ghanaopenai/ghana-voice-nano-twi checkpoints/twi_epoch045.ckpt); it already
# knows the lfn phoneme set and West African prosody. For anything else, the English
# LJSpeech checkpoint (github.com/shivammehta25/Matcha-TTS-checkpoints) works too.
# Only weights are loaded; mel statistics always come from this dataset's stats.json.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$(cd "$1" && pwd)"
CKPT="$(realpath "$2")"
shift 2

read -r MEAN STD < <(python3 -c "import json;s=json.load(open('$DATA/stats.json'));print(s['mel_mean'],s['mel_std'])")
echo "[train] data=$DATA mel_mean=$MEAN mel_std=$STD warm_start=$CKPT"

cd "$ROOT/third_party/Matcha-TTS"

python matcha/train.py \
    --config-dir "$ROOT/recipe/configs" \
    experiment=nano \
    finetune_ckpt="$CKPT" \
    data.train_filelist_path="$DATA/train.txt" \
    data.valid_filelist_path="$DATA/val.txt" \
    data.data_statistics.mel_mean="$MEAN" \
    data.data_statistics.mel_std="$STD" \
    trainer.devices=[0] \
    trainer.precision=bf16-mixed \
    paths.log_dir="$ROOT/runs" \
    test=false \
    "$@"
