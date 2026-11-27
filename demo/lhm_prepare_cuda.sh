#!/usr/bin/env bash
set -euo pipefail
mkdir -p /opt/lhm-cuda
for package in vendor/cuda-*.tar.bz2; do
    tar -xjf "$package" -C /opt/lhm-cuda
done
/opt/lhm-cuda/bin/nvcc --version
test -f /opt/lhm-cuda/include/cuda_runtime.h
