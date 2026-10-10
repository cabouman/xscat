# Per-package settings for the dev scripts.  This file and the scripts its hooks
# call (install_leap.sh) are the only dev scripts that differ between packages;
# all the others are identical across repositories.
NAME="xscat"
PYTHON_VERSION="3.11"
EXTRAS="test,docs,viewer"   # pip extras installed with the editable package

# Compiled dependencies, installed by install_leap.sh before the package.
# CUDA toolkit (conda, nvidia channel) that compiles LEAP on Linux.  The NVIDIA
# driver must support this CUDA version (`nvidia-smi` shows the highest it does).
CUDA_TOOLKIT_VERSION="12.8"
# PyTorch wheel index on Linux.  mbirtorch needs torch >= 2.13, the cu128 index
# stops at torch 2.11, and the default PyPI wheel is built for CUDA 13 (driver
# >= 580).  "auto" picks cu129 (torch 2.13, GPUs sm_75 to sm_120, i.e. with
# Blackwell) when the machine has a Blackwell GPU, and cu126 (torch 2.14, sm_50
# to sm_90) otherwise.
TORCH_INDEX="auto"
TORCH_SPEC="torch>=2.13"
# Tag, branch or full commit SHA of https://github.com/LLNL/LEAP
LEAP_REF="v1.26"
# Tag, branch or full commit SHA of https://github.com/kylechampley/XrayPhysics:
# main as of 2025-01-19; the last tag (v1.2.1) lacks resample(), which xscat uses.
XRAYPHYSICS_REF="478388a30aba22ede01a7a01e4b751ee3891cad9"

# Ignore packages in ~/.local, which pip and Python would otherwise take as
# installed in the env.
export PYTHONNOUSERSITE=1

# Purdue RCAC clusters provide conda through environment modules.  Every dev
# script sources this file before it calls conda, so load them here.
case "${HOSTNAME:-$(hostname)}" in
  *gilbreth*) HOST_MODULES="external conda" ;;
  *gautschi*) HOST_MODULES="modtree/gpu conda" ;;
  *negishi*)  HOST_MODULES="conda" ;;
  *)          HOST_MODULES="" ;;
esac
if [ -n "$HOST_MODULES" ] && ! command -v conda >/dev/null 2>&1; then
  if ! type module >/dev/null 2>&1 && [ -n "${MODULESHOME:-}" ]; then
    source "$MODULESHOME/init/bash"
  fi
  module load $HOST_MODULES
fi

# Optional hooks: define a function to run extra steps, or leave it out to skip.
# before_env_create() { : ; }   # runs before the env is created (e.g. module load)
# after_env_create()  { : ; }   # runs in clean_install_all after the env is created
# extra_clean()       { : ; }   # runs during remove_package (e.g. a cache)
after_env_create() { bash "$SCRIPT_DIR/install_leap.sh"; }
