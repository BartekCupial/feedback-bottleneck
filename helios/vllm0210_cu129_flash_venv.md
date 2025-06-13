# Helios vLLM 0.21.0 CUDA 12.9 Flash-Attn Venv

This document describes how to build the GH200 evaluation environment used by Plan-CRL experiments. It is intentionally generic: it does not contain experiment-specific model lists or private side-project benchmark configuration.

## Script

```bash
helios/vllm0210_cu129_flash_venv_gh200.sbatch
```

The script creates:

```bash
.venv_vllm0210_cu129_flash_py311
```

## Environment Stack

This script builds a CUDA 12.9-compatible stack for GH200 nodes:

- `vllm==0.21.0`
- PyTorch selected through `uv` with `--torch-backend=cu129`
- vLLM's bundled FlashAttention 3 backend
- the official CUDA 13 runtime, NVRTC, and nvJitLink wheels, because the vLLM 0.21.0 aarch64 wheel loads `libcudart.so.13` and vendored DeepGEMM loads `libnvrtc.so.13`
- optional external `flash-attn` built from source against the installed Torch when `INSTALL_FLASH_ATTN=1`
- GH200/Hopper architecture flags:
  - `TORCH_CUDA_ARCH_LIST=9.0`
  - `FLASH_ATTN_CUDA_ARCHS=90`

It also verifies that `torch.version.cuda` matches `nvcc --version` before any optional `flash-attn` compile.

On GH200/aarch64 the script keeps the CUDA 12.9 module stack from `ML-bundle/25.10`, but builds native CUDA extensions with GCCcore 13.2 through `CC`, `CXX`, and `CUDAHOSTCXX`. This avoids the CUDA 12.9/PyTorch extension guard rejecting GCC 14.x while leaving `nvcc` on CUDA 12.9.

## Default Build

Run from the repo root on Helios:

```bash
cd /path/to/plan-crl
sbatch helios/vllm0210_cu129_flash_venv_gh200.sbatch
```

Useful output to check in the Slurm log:

```text
torch ... torch_cuda 12.9 ...
vllm 0.21.0
import xgrammar ...
import ray ...
import peft ...
import mrunner ...
import plan_crl ...
VENV_READY=.../.venv_vllm0210_cu129_flash_py311
```

## Source-Build vLLM Fallback

If the wheel path does not support a model/runtime combination we need, build vLLM from the `v0.21.0` tag:

```bash
cd /path/to/plan-crl
VLLM_INSTALL_MODE=source sbatch helios/vllm0210_cu129_flash_venv_gh200.sbatch
```

You can override the tag or commit:

```bash
VLLM_INSTALL_MODE=source \
VLLM_GIT_REF=v0.21.0 \
sbatch helios/vllm0210_cu129_flash_venv_gh200.sbatch
```

## Pinning Flash-Attn

When `INSTALL_FLASH_ATTN=1`, the script builds the current default branch of `Dao-AILab/flash-attention`. To pin a known-good commit or tag:

```bash
FLASH_ATTN_REF=<commit-or-tag> \
sbatch helios/vllm0210_cu129_flash_venv_gh200.sbatch
```

The default build relies on the vLLM wheel's internal FlashAttention 3 stack, which is the setup used by the current GH200 evaluation runs.

The build also appends a small `LD_LIBRARY_PATH` hook to `.venv_vllm0210_cu129_flash_py311/bin/activate`, so this is enough before launching an experiment:

```bash
source helios/code_env_gh200.sh
source .venv_vllm0210_cu129_flash_py311/bin/activate
```

To additionally build the external `flash-attn` package:

```bash
INSTALL_FLASH_ATTN=1 \
sbatch helios/vllm0210_cu129_flash_venv_gh200.sbatch
```

## Reusing The Venv

After the build succeeds, use the venv in run scripts by setting:

```bash
export REPO_ROOT=/path/to/plan-crl
export VENV_DIR=$REPO_ROOT/.venv_vllm0210_cu129_flash_py311
```

For example:

```bash
sbatch \
  --export=ALL,REPO_ROOT=/path/to/plan-crl,VENV_DIR=/path/to/plan-crl/.venv_vllm0210_cu129_flash_py311,CONFIG_DIR=<materialized-config-dir> \
  helios/jan_reruns_array_gh200.sbatch
```

## Main Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `VENV_DIR` | `$REPO_ROOT/.venv_vllm0210_cu129_flash_py311` | Output venv path. |
| `PLANCRL_BUILD_ROOT` | `$REPO_ROOT/.cache/plan_crl_build_vllm0210_cu129_flash` | Build/cache directory. |
| `VLLM_VERSION` | `0.21.0` | vLLM version for wheel installs. |
| `VLLM_INSTALL_MODE` | `wheel` | `wheel` or `source`. |
| `CUDA_BACKEND` | `cu129` | Torch backend selected by `uv`. |
| `INSTALL_CUDA13_RUNTIME` | `1` | Install CUDA 13 runtime, NVRTC, and nvJitLink libs required by the vLLM 0.21.0 aarch64 wheel and patch venv activation. |
| `INSTALL_FLASH_ATTN` | `0` | Build external `flash-attn`; the default uses vLLM's bundled FlashAttention 3 backend. |
| `FLASH_ATTN_REF` | empty | Optional flash-attn commit/tag. |
| `INSTALL_FLASHINFER` | `1` | Try to install `flashinfer-python`. |
| `INSTALL_XGRAMMAR` | `1` | Install `xgrammar`. |
| `RESET_VENV` | `1` | Recreate venv from scratch. |
| `MAX_JOBS` | `16` | Build parallelism for native extensions. |
| `NVCC_THREADS` | `4` | nvcc thread count. |
| `PLANCRL_CUDA_HOST_CC` | `/net/software/aarch64/el9/GCCcore/13.2.0/bin/gcc` on GH200 | C compiler used for CUDA extension builds. |
| `PLANCRL_CUDA_HOST_CXX` | `/net/software/aarch64/el9/GCCcore/13.2.0/bin/g++` on GH200 | C++ compiler used for CUDA extension builds. |

## Validation

The script checks:

- `torch.version.cuda` vs `nvcc --version`
- `vllm.__version__` starts with `0.21.0`
- imports for:
  - `torch`
  - `transformers`
  - `vllm`
  - `vllm.LLM`
  - vendored DeepGEMM availability
  - `flash_attn` when external build is enabled
  - `flashinfer` when available
  - `xgrammar`
  - `ray`
  - `peft`
  - `mrunner`
  - `plan_crl`

It also runs `python -m pip check` at the end, but does not fail the build on dependency warnings because this repo intentionally carries some older environment dependencies.
