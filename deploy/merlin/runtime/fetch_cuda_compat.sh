#!/usr/bin/env bash
# Downloads and EXTRACTS NVIDIA's user-space compatibility libraries. No system installation.
set -euo pipefail
PEAR_ROOT=${PEAR_ROOT:-/mnt/bn/ic-vlm/personal/loongfei.li03/pear_20261114}
VERSION=580.196.07-1ubuntu1
PACKAGE=cuda-compat-13-0_${VERSION}_amd64.deb
URL=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/$PACKAGE
SHA256=b0ee4fb1f439e919ca5c5fd7c5cff3e8b8afdbdb0b5ee38f7c09c047ea41a035
DEST=$PEAR_ROOT/compat/cuda-compat-13-0_580.196.07
mkdir -p "$PEAR_ROOT/compat/downloads"
DEB=$PEAR_ROOT/compat/downloads/$PACKAGE
if [[ ! -f "$DEB" ]]; then
  curl --fail --location --retry 3 "$URL" -o "$DEB.partial"
  printf '%s  %s\n' "$SHA256" "$DEB.partial" | sha256sum --check -
  mv "$DEB.partial" "$DEB"
fi
printf '%s  %s\n' "$SHA256" "$DEB" | sha256sum --check -
if [[ ! -e "$DEST/.extracted" ]]; then
  mkdir -p "$DEST"
  dpkg-deb --extract "$DEB" "$DEST"
  [[ -e "$DEST/usr/local/cuda-13.0/compat/libcuda.so.1" ]]
  printf '%s\n' "$SHA256" > "$DEST/.extracted"
fi
printf 'PEAR_CUDA_COMPAT_DIR=%s\n' "$DEST/usr/local/cuda-13.0/compat"
