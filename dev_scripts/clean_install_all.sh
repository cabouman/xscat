#!/bin/bash
# Full clean install: remove, recreate the env, install, build docs.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

bash "$SCRIPT_DIR/remove_package.sh"
bash "$SCRIPT_DIR/install_empty_conda_environment.sh"
# Package-specific installs into the new env (e.g. compiled dependencies), if
# config.sh defines it.
if declare -f after_env_create >/dev/null; then after_env_create; fi
bash "$SCRIPT_DIR/install_package.sh"
bash "$SCRIPT_DIR/build_docs.sh"

# tput fails without a terminal (e.g. in CI); the message is plain then.
red=$(tput setaf 1 2>/dev/null || true); reset=$(tput sgr0 2>/dev/null || true)
echo
echo "Use  ${red}conda activate $NAME${reset}  to activate the environment."
