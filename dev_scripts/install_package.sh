#!/bin/bash
# Install the package (editable) with its developer extras into the env.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate "$NAME"
# Editable (-e) keeps the environment pointed at this checkout's code.
pip install -e "$REPO_ROOT[$EXTRAS]"
