#!/bin/bash -l
#SBATCH --job-name=plancrl-vllm021-cu129-flash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=plgcrlreason-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv_vllm0210_cu129_flash_py311}"
PLANCRL_BUILD_ROOT="${PLANCRL_BUILD_ROOT:-$REPO_ROOT/.cache/plan_crl_build_vllm0210_cu129_flash}"

VLLM_VERSION="${VLLM_VERSION:-0.21.0}"
VLLM_INSTALL_MODE="${VLLM_INSTALL_MODE:-wheel}"  # wheel or source
VLLM_GIT_REF="${VLLM_GIT_REF:-v${VLLM_VERSION}}"
CUDA_BACKEND="${CUDA_BACKEND:-cu129}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
FLASH_ATTN_REPO_URL="${FLASH_ATTN_REPO_URL:-https://github.com/Dao-AILab/flash-attention.git}"
FLASH_ATTN_REF="${FLASH_ATTN_REF:-}"  # empty means repository default branch
INSTALL_CUDA13_RUNTIME="${INSTALL_CUDA13_RUNTIME:-1}"
INSTALL_FLASHINFER="${INSTALL_FLASHINFER:-1}"
INSTALL_XGRAMMAR="${INSTALL_XGRAMMAR:-1}"
INSTALL_PLAN_CRL_EDITABLE="${INSTALL_PLAN_CRL_EDITABLE:-0}"
RESET_VENV="${RESET_VENV:-1}"

export REPO_ROOT VENV_DIR PLANCRL_BUILD_ROOT
export INSTALL_FLASH_ATTN INSTALL_CUDA13_RUNTIME INSTALL_FLASHINFER INSTALL_XGRAMMAR INSTALL_PLAN_CRL_EDITABLE RESET_VENV
cd "$REPO_ROOT"
mkdir -p "$PLANCRL_BUILD_ROOT" slurm

source "$REPO_ROOT/helios/code_env_gh200.sh"
if declare -F deactivate >/dev/null 2>&1; then
  deactivate || true
fi
hash -r

if [ "$(uname -m)" = "aarch64" ]; then
  PLANCRL_CUDA_HOST_CC="${PLANCRL_CUDA_HOST_CC:-/net/software/aarch64/el9/GCCcore/13.2.0/bin/gcc}"
  PLANCRL_CUDA_HOST_CXX="${PLANCRL_CUDA_HOST_CXX:-/net/software/aarch64/el9/GCCcore/13.2.0/bin/g++}"
  if [ ! -x "$PLANCRL_CUDA_HOST_CC" ] || [ ! -x "$PLANCRL_CUDA_HOST_CXX" ]; then
    echo "ERROR: CUDA host compiler override is missing: $PLANCRL_CUDA_HOST_CC / $PLANCRL_CUDA_HOST_CXX" >&2
    exit 1
  fi
  export CC="$PLANCRL_CUDA_HOST_CC"
  export CXX="$PLANCRL_CUDA_HOST_CXX"
  export CUDAHOSTCXX="$PLANCRL_CUDA_HOST_CXX"
fi

export TMPDIR="$PLANCRL_BUILD_ROOT/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export XDG_CACHE_HOME="$PLANCRL_BUILD_ROOT/xdg"
export PIP_CACHE_DIR="$PLANCRL_BUILD_ROOT/pip"
export UV_CACHE_DIR="$PLANCRL_BUILD_ROOT/uv"
export TORCH_EXTENSIONS_DIR="$PLANCRL_BUILD_ROOT/torch_extensions"
export FLASH_ATTN_DIR="$PLANCRL_BUILD_ROOT/flash-attention-src"
export VLLM_SRC_DIR="$PLANCRL_BUILD_ROOT/vllm-src"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-180}"
export PIP_RETRIES="${PIP_RETRIES:-20}"
export MAX_JOBS="${MAX_JOBS:-16}"
export NVCC_THREADS="${NVCC_THREADS:-4}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-90}"
export FLASH_ATTENTION_FORCE_BUILD="${FLASH_ATTENTION_FORCE_BUILD:-TRUE}"
export FLASH_ATTENTION_SKIP_CUDA_BUILD="${FLASH_ATTENTION_SKIP_CUDA_BUILD:-FALSE}"
export UV_TORCH_BACKEND="$CUDA_BACKEND"

mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

echo "repo=$REPO_ROOT"
echo "venv=$VENV_DIR"
echo "build_root=$PLANCRL_BUILD_ROOT"
echo "cuda_backend=$CUDA_BACKEND"
echo "vllm=$VLLM_VERSION mode=$VLLM_INSTALL_MODE ref=$VLLM_GIT_REF"
echo "flash_attn_ref=${FLASH_ATTN_REF:-<default-branch>}"
echo "nvcc=$(command -v nvcc || true)"
echo "cc=${CC:-$(command -v cc || true)}"
echo "cxx=${CXX:-$(command -v c++ || true)}"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc is not available after sourcing helios/code_env_gh200.sh; CUDA module load failed." >&2
  exit 1
fi
nvcc --version
"${CXX:-c++}" --version | head -1
nvidia-smi || true

if [ "$RESET_VENV" = "1" ] && [ -d "$VENV_DIR" ]; then
  rm -rf "$VENV_DIR"
fi

PYTHON_BIN_RESOLVED="$(command -v "$PYTHON_BIN")"
echo "python_for_venv=$PYTHON_BIN_RESOLVED"
"$PYTHON_BIN_RESOLVED" - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit(
        f"ERROR: Python >=3.10 is required for vLLM 0.21.0, got {sys.version.split()[0]}. "
        "Check that helios/code_env_gh200.sh loaded the requested Python module."
    )
print("python_version_ok", sys.version.replace("\n", " "))
PY
"$PYTHON_BIN_RESOLVED" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip "setuptools<82" wheel
python -m pip install --upgrade uv
uv pip install --python "$VENV_DIR/bin/python" cmake ninja packaging setuptools-scm psutil pybind11

ensure_cuda13_runtime_path() {
  local activate_script="$VENV_DIR/bin/activate"
  local cuda13_lib

  for cuda13_lib in \
    "$VENV_DIR"/lib/python*/site-packages/nvidia/cu13/lib \
    "$VENV_DIR"/lib/python*/site-packages/nvidia/cuda_nvrtc/lib \
    "$VENV_DIR"/lib/python*/site-packages/nvidia/nvjitlink/lib; do
    if [ -d "$cuda13_lib" ]; then
      case ":${LD_LIBRARY_PATH:-}:" in
        *":$cuda13_lib:"*) ;;
        *) export LD_LIBRARY_PATH="$cuda13_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
      esac
    fi
  done

  if [ -f "$activate_script" ] && ! grep -q "nvidia/cuda_nvrtc/lib" "$activate_script"; then
    cat >> "$activate_script" <<'EOF'

# Plan-CRL: vLLM 0.21.0 aarch64 wheels load libcudart.so.13.
for PLANCRL_CUDA13_RUNTIME_LIB in \
  "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/cu13/lib \
  "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/cuda_nvrtc/lib \
  "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/nvjitlink/lib; do
  if [ -d "$PLANCRL_CUDA13_RUNTIME_LIB" ]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *":$PLANCRL_CUDA13_RUNTIME_LIB:"*) ;;
      *) export LD_LIBRARY_PATH="$PLANCRL_CUDA13_RUNTIME_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
  fi
done
unset PLANCRL_CUDA13_RUNTIME_LIB
EOF
  fi
}

if [ "$VLLM_INSTALL_MODE" = "source" ]; then
  rm -rf "$VLLM_SRC_DIR"
  git clone --recursive https://github.com/vllm-project/vllm.git "$VLLM_SRC_DIR"
  git -C "$VLLM_SRC_DIR" checkout "$VLLM_GIT_REF"
  git -C "$VLLM_SRC_DIR" submodule update --init --recursive
  uv pip install --python "$VENV_DIR/bin/python" "torch" "torchvision" --torch-backend="$CUDA_BACKEND"
  (
    cd "$VLLM_SRC_DIR"
    uv pip install --python "$VENV_DIR/bin/python" -r requirements/build.txt || true
    uv pip install --python "$VENV_DIR/bin/python" -r requirements/cuda.txt || true
    VLLM_TARGET_DEVICE=cuda python -m pip install -v -e . --no-build-isolation
  )
else
  uv pip install --python "$VENV_DIR/bin/python" "vllm==${VLLM_VERSION}" --torch-backend="$CUDA_BACKEND"
fi

if [ "$INSTALL_CUDA13_RUNTIME" = "1" ]; then
  uv pip install --python "$VENV_DIR/bin/python" nvidia-cuda-runtime nvidia-cuda-nvrtc nvidia-nvjitlink
  ensure_cuda13_runtime_path
fi

python - <<'PY'
import re
import os
import subprocess
import torch
import vllm
from vllm import LLM
from vllm.utils.deep_gemm import get_mk_alignment_for_contiguous_layout, is_deep_gemm_supported

print("torch", torch.__version__, "torch_cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("vllm", vllm.__version__)
print("vllm_LLM", LLM)
print("deep_gemm_supported", is_deep_gemm_supported())
print("deep_gemm_alignment", get_mk_alignment_for_contiguous_layout())
nvcc = subprocess.run(["nvcc", "--version"], text=True, capture_output=True, check=False)
print(nvcc.stdout)
match = re.search(r"release\s+(\d+\.\d+)", nvcc.stdout)
if not match:
    raise RuntimeError("Could not parse nvcc version")
nvcc_major_minor = match.group(1)
torch_major_minor = ".".join(str(torch.version.cuda).split(".")[:2])
if torch_major_minor != nvcc_major_minor:
    raise RuntimeError(f"CUDA mismatch: torch {torch.version.cuda}, nvcc {nvcc_major_minor}")
expected_vllm = os.environ.get("VLLM_VERSION", "0.21.0")
if not str(vllm.__version__).startswith(expected_vllm):
    raise RuntimeError(f"Expected vLLM {expected_vllm}, got {vllm.__version__}")
PY

echo "Installing plan-crl runtime dependencies without repinning torch/vllm"
uv pip install --python "$VENV_DIR/bin/python" \
  "numpy<2" \
  "gym==0.23" \
  "gymnasium>=0.27,<1.0" \
  crafter \
  tyro \
  pyarrow \
  matplotlib \
  pandas \
  jinja2 \
  scipy \
  tensorboard \
  tensorboardx \
  "hydra-core>=1.3" \
  datasets \
  tqdm \
  seaborn \
  "wandb>=0.12.9" \
  wandb-osh \
  ipywidgets \
  ipympl \
  "imageio[ffmpeg]" \
  pylatexenc \
  math_verify \
  accelerate \
  "ray[default]" \
  peft \
  cloudpickle \
  sentencepiece \
  protobuf \
  hf-transfer \
  colorama \
  "mrunner @ git+https://gitlab.com/awarelab/mrunner.git"

if [ "$INSTALL_XGRAMMAR" = "1" ]; then
  uv pip install --python "$VENV_DIR/bin/python" xgrammar
fi

if [ "$INSTALL_CUDA13_RUNTIME" = "1" ]; then
  ensure_cuda13_runtime_path
fi

if [ "$INSTALL_FLASHINFER" = "1" ]; then
  uv pip install --python "$VENV_DIR/bin/python" flashinfer-python --torch-backend="$CUDA_BACKEND" || {
    echo "flashinfer-python install failed; continuing because vLLM may already provide the needed backend" >&2
  }
fi

if [ "$INSTALL_PLAN_CRL_EDITABLE" = "1" ]; then
  python -m pip install -e . --no-deps --no-build-isolation
else
  echo "Skipping editable plan-crl install; use PYTHONPATH=$REPO_ROOT"
fi

if [ "$INSTALL_FLASH_ATTN" = "1" ]; then
  echo "Building latest flash-attn against torch CUDA $(python - <<'PY'
import torch
print(torch.version.cuda)
PY
)"
  rm -rf "$FLASH_ATTN_DIR"
  git clone --recursive "$FLASH_ATTN_REPO_URL" "$FLASH_ATTN_DIR"
  if [ -n "$FLASH_ATTN_REF" ]; then
    git -C "$FLASH_ATTN_DIR" checkout "$FLASH_ATTN_REF"
    git -C "$FLASH_ATTN_DIR" submodule update --init --recursive
  fi
  (
    cd "$FLASH_ATTN_DIR"
    rm -rf build dist *.egg-info
    python -m pip install -v . --no-build-isolation --no-cache-dir
  )
else
  echo "Skipping external flash-attn build"
fi

python "$REPO_ROOT/helios/patch_vllm_shm_broadcast.py" || true

python - <<'PY'
import importlib
import os
import sys

import torch
import vllm
from vllm import LLM
from vllm.utils.deep_gemm import get_mk_alignment_for_contiguous_layout, is_deep_gemm_supported

print("FINAL torch", torch.__version__, "torch_cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("FINAL vllm", vllm.__version__)
print("FINAL vllm_LLM", LLM)
print("FINAL deep_gemm_supported", is_deep_gemm_supported())
print("FINAL deep_gemm_alignment", get_mk_alignment_for_contiguous_layout())
print("FINAL python", sys.version.replace("\n", " "))
print("FINAL env VLLM_ATTENTION_BACKEND", os.environ.get("VLLM_ATTENTION_BACKEND"))

for name in ["transformers", "xgrammar", "ray", "peft", "mrunner", "plan_crl"]:
    mod = importlib.import_module(name)
    print("import", name, getattr(mod, "__version__", ""), getattr(mod, "__file__", ""))

if os.environ.get("INSTALL_FLASH_ATTN", "0") == "1":
    flash_attn = importlib.import_module("flash_attn")
    print("import flash_attn", getattr(flash_attn, "__version__", ""), getattr(flash_attn, "__file__", ""))

try:
    import flashinfer
    print("import flashinfer", getattr(flashinfer, "__version__", ""), getattr(flashinfer, "__file__", ""))
except Exception as exc:
    print("flashinfer import warning", type(exc).__name__, exc)

try:
    from vllm.reasoning.gemma4_utils import parse_thinking_output
    print("gemma4_parser", parse_thinking_output.__name__)
except Exception as exc:
    print("gemma4_parser warning", type(exc).__name__, exc)

try:
    from vllm.reasoning import ReasoningParser
    print("reasoning_parser", ReasoningParser)
except Exception as exc:
    print("reasoning_parser warning", type(exc).__name__, exc)
PY

python -m pip check || true

echo "VENV_READY=$VENV_DIR"
