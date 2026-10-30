#!/usr/bin/env bash
# One-time setup of the training environment (Linux + NVIDIA GPU).
# Creates .venv/ and clones the Matcha-TTS engine fork into third_party/.
set -euo pipefail
cd "$(dirname "$0")/.."

MATCHA_REPO=https://github.com/michsethowusu/Matcha-TTS
MATCHA_COMMIT=093e0de   # tested engine version (includes the finetune mel-stat fix)

command -v espeak-ng >/dev/null || { echo "install espeak-ng first: sudo apt-get install -y espeak-ng"; exit 1; }

if [ ! -d third_party/Matcha-TTS ]; then
    git clone "$MATCHA_REPO" third_party/Matcha-TTS
fi
git -C third_party/Matcha-TTS checkout -q "$MATCHA_COMMIT"

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
# Match the torch wheel to your NVIDIA driver (nvidia-smi shows the max CUDA version), e.g.
# TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 recipe/setup.sh
pip install torch torchaudio --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
pip install -r recipe/requirements.txt

# Build the monotonic-alignment Cython extension in place (the engine's setup.py pins old
# build deps that fail on Python 3.12, so compile only the extension).
(cd third_party/Matcha-TTS && python - <<'PY'
import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

ext = Extension("matcha.utils.monotonic_align.core", ["matcha/utils/monotonic_align/core.pyx"],
                include_dirs=[numpy.get_include()])
setup(script_args=["build_ext", "--inplace", "-q"], ext_modules=cythonize([ext], quiet=True))
PY
)
# Make `import matcha` work from anywhere in this venv.
echo "$PWD/third_party/Matcha-TTS" > "$(python -c 'import site; print(site.getsitepackages()[0])')/matcha_engine.pth"

python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
PY
echo "done. Activate with: source .venv/bin/activate"
