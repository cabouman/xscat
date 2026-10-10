#!/bin/bash
# Install the compiled dependencies into the env: PyTorch for the GPU driver,
# and LEAP (leapct) and XrayPhysics built from source, which pip cannot do from
# the package's dependency list.  clean_install_all.sh runs this through the
# after_env_create hook in config.sh; run it alone to rebuild them in the env.
#
# Linux: needs git, make and a C/C++ compiler.  The CUDA toolkit comes from
#   conda, so the build needs no GPU; LEAP's scatter model needs one to run.
# macOS: LEAP builds only with CUDA, so only PyTorch and XrayPhysics are installed.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"

die() { echo "Error: $*" >&2; exit 1; }

conda activate "$NAME"
OS="$(uname -s)"
[ "$OS" = Linux ] || [ "$OS" = Darwin ] || die "unsupported OS $OS (use Linux or macOS)"
# On macOS git, make and c++ always exist as stubs, so check for the tools themselves.
if [ "$OS" = Darwin ]; then
  xcode-select -p >/dev/null 2>&1 || die "the Xcode Command Line Tools are missing; run xcode-select --install"
fi
for tool in git make c++; do
  command -v "$tool" >/dev/null 2>&1 || die "$tool not found; install the C/C++ build tools" \
    "(Ubuntu: sudo apt install build-essential git; macOS: xcode-select --install)"
done
# Sources and build logs go under build/, which remove_package.sh deletes.
SRC_DIR="$REPO_ROOT/build/third_party"
mkdir -p "$SRC_DIR"

# Conda packages first, then pip.
if [ "$OS" = Linux ]; then
  # nvcc and the CUDA libraries for LEAP.  cuda-version keeps every component
  # of the toolkit at the same release.
  conda install -y -c nvidia "cuda-toolkit=$CUDA_TOOLKIT_VERSION" "cuda-version=$CUDA_TOOLKIT_VERSION"
  export CUDA_HOME="$CONDA_PREFIX" CUDA_PATH="$CONDA_PREFIX"
  export CUDA_TOOLKIT_ROOT_DIR="$CONDA_PREFIX" CUDACXX="$CONDA_PREFIX/bin/nvcc"
  [ -x "$CUDACXX" ] || die "the conda CUDA toolkit has no $CUDACXX"
  nvcc_release="$("$CUDACXX" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
  [ "$nvcc_release" = "$(echo "$CUDA_TOOLKIT_VERSION" | cut -d. -f1,2)" ] ||
    die "conda installed nvcc $nvcc_release, not CUDA $CUDA_TOOLKIT_VERSION"
else
  # Apple clang has no OpenMP runtime, which XrayPhysics needs; CMake finds
  # this one through CMAKE_PREFIX_PATH.
  conda install -y -c conda-forge llvm-openmp
  export CMAKE_PREFIX_PATH="$CONDA_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
fi

# CMake < 4: XrayPhysics declares cmake_minimum_required(VERSION 3.0), which
# CMake 4 rejects.
python -m pip install --upgrade pip setuptools wheel build "cmake<4"

# PyTorch before the package, so that pip keeps this build when it installs
# mbirtorch.
if [ "$OS" = Linux ]; then
  if [ "$TORCH_INDEX" = auto ]; then
    # Highest compute capability of the GPUs, empty without a GPU or driver.
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null |
          grep -E '^[0-9]+\.[0-9]+$' | sort -V | tail -n 1 || true)"
    if [ -n "$cc" ] && [ "${cc%%.*}" -ge 10 ]; then TORCH_INDEX=cu129; else TORCH_INDEX=cu126; fi
    echo "PyTorch wheels: $TORCH_INDEX (highest GPU compute capability: ${cc:-no GPU found})"
  fi
  python -m pip install "$TORCH_SPEC" --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"
else
  python -m pip install "$TORCH_SPEC"
fi

# Fetch a tag, branch or commit into a fresh directory, then build and install
# it.  setup.py runs CMake every time it is executed, and pip would execute it
# twice, so build the wheel once with this env's setuptools and cmake.  The
# output goes to a log; on failure, show it up to setup.py's own message.
build_from_source() {  # name url ref
  local dir="$SRC_DIR/$1" log="$SRC_DIR/$1-build.log"
  rm -rf "$dir"
  git init --quiet "$dir"
  git -C "$dir" fetch --quiet --depth 1 "$2" "$3"
  git -C "$dir" checkout --quiet FETCH_HEAD
  echo "Building $1 $3 (log: $log)"
  if ! (cd "$dir" && python -m build --wheel --no-isolation --skip-dependency-check --outdir dist . &&
        python -m pip install --no-deps --force-reinstall dist/*.whl) >"$log" 2>&1; then
    sed '/Failed to compile!/q' "$log" | tail -n 40 >&2
    die "$1 failed to build; the full log is $log"
  fi
}

if [ "$OS" = Linux ]; then
  echo "LEAP compiles CUDA code for every major GPU architecture; this takes several minutes."
  build_from_source LEAP https://github.com/LLNL/LEAP.git "$LEAP_REF"
fi
build_from_source XrayPhysics https://github.com/kylechampley/XrayPhysics.git "$XRAYPHYSICS_REF"
