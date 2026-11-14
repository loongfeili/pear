# Sourced by preparation and GPU checks. Never changes the old shared UV/Python directories.
export USER=${USER:-$(id -un)}
export PEAR_ROOT=${PEAR_ROOT:-/mnt/bn/ic-vlm/personal/loongfei.li03/pear_20261114}
export WORK_DIR=$PEAR_ROOT/work_dir
export PEAR_RUNTIME=/opt/tiger/pear_runtime_20261114
export PEAR_ENVS_DIR=$PEAR_RUNTIME/envs
export VERL_ENV=$PEAR_ENVS_DIR/verl
export LIBERO_PYTHON=$PEAR_ENVS_DIR/libero/bin/python
export LIBERO_CONFIG_PATH=$PEAR_ENVS_DIR/libero/etc/libero
export LIBERO_MESA_DIR=$PEAR_ENVS_DIR/mesa
export PEAR_BIN_DIR=$PEAR_RUNTIME/bin
export PEAR_CACHE_DIR=$PEAR_RUNTIME/cache
export PEAR_PYTHON_INSTALL_DIR=$PEAR_ENVS_DIR/uv-python
export PEAR_UV_CACHE_DIR=$PEAR_CACHE_DIR/uv
export PEAR_UV_TOOL_DIR=$PEAR_CACHE_DIR/uv_tools
export PEAR_UV_CREDENTIALS_DIR=$PEAR_CACHE_DIR/uv_credentials
export PEAR_MAMBA_HOME=$PEAR_CACHE_DIR/micromamba
export PEAR_PIP_CACHE_DIR=$PEAR_CACHE_DIR/pip
export PEAR_TEST_CACHE_DIR=$PEAR_CACHE_DIR/environment_test
export LOCAL_DIR=$PEAR_RUNTIME/cache/runtime
export TMPDIR=$PEAR_RUNTIME/cache/tmp
export CUDA_ARCH=9.0 MAX_JOBS=8 UV_LINK_MODE=copy
export PEAR_CUDA_COMPAT_DIR=$PEAR_ROOT/compat/cuda-compat-13-0_580.196.07/usr/local/cuda-13.0/compat
export LD_LIBRARY_PATH=$PEAR_CUDA_COMPAT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
mkdir -p "$PEAR_RUNTIME" "$PEAR_ENVS_DIR" "$PEAR_BIN_DIR" "$PEAR_CACHE_DIR" "$TMPDIR"
