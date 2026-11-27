#!/usr/bin/env bash
set -euo pipefail
WORKSPACE=$(pwd)
export CUDA_HOME=/opt/lhm-cuda
export PATH="$CUDA_HOME/bin:/opt/lhm-venv/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=2
for include_dir in /opt/lhm-venv/lib/python3.10/site-packages/nvidia/*/include; do
    cp -rs "$include_dir"/. "$CUDA_HOME/include/" 2>/dev/null || true
done
mkdir -p "$CUDA_HOME/lib64"
for lib_dir in /opt/lhm-venv/lib/python3.10/site-packages/nvidia/*/lib; do
    cp -rs "$lib_dir"/. "$CUDA_HOME/lib64/" 2>/dev/null || true
done
ln -sf /opt/lhm-venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12 "$CUDA_HOME/lib64/libcudart.so"
cd "$WORKSPACE/vendor/LHM-plusplus-main/lib/pointops"
python setup.py build_ext --inplace
cp pointops_cuda*.so /opt/lhm-venv/lib/python3.10/site-packages/
cd "$WORKSPACE/vendor/GauHuman-main/submodules/diff-gaussian-rasterization"
python setup.py build_ext --inplace
cp -r diff_gaussian_rasterization /opt/lhm-venv/lib/python3.10/site-packages/
