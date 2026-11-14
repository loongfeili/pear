#!/usr/bin/env bash
# Scope compatibility libraries to this process tree; do not touch the system driver.
set -euo pipefail
: "${PEAR_CUDA_COMPAT_DIR:?Set the verified extracted cuda-compat-13-0 directory}"
[[ -e "$PEAR_CUDA_COMPAT_DIR/libcuda.so.1" ]] || { echo 'Missing compat libcuda.so.1' >&2; exit 2; }
[[ "${1:-}" != -- ]] || shift
(( $# > 0 )) || { echo 'Usage: with_cuda_compat.sh -- command args...' >&2; exit 2; }
export LD_LIBRARY_PATH=$PEAR_CUDA_COMPAT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
exec "$@"
