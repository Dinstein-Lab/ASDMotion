#!/bin/bash
# Linux equivalent of run_in_env.bat
# Activates the open-mmlab conda environment and runs the given command.
#
# Usage: ./run_in_env.sh <command> [args...]
#
# Update CONDA_PATH and ENV_NAME below to match your system.

CONDA_PATH="${CONDA_PATH:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-open-mmlab}"

# Initialize conda for this shell
eval "$($CONDA_PATH/bin/conda shell.bash hook)"

# Activate the environment
conda activate "$ENV_NAME"

# Execute the command passed to this script
"$@"

# Deactivate
conda deactivate
