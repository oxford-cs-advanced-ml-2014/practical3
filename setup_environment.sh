#!/bin/bash

set -e

ROOT=$(dirname $(readlink -e $0))
LOCAL_MACHINE_SCRATCH_SPACE=/home/scratch
ENV="$(mktemp -u -d -p "$LOCAL_MACHINE_SCRATCH_SPACE" "conda_env.$USER.XXXXXXXXXXXX")/conda"

function safe_call {
    # usage:
    #   safe_call function param1 param2 ...

    HERE=$(pwd)
    "$@"
    cd "$HERE"
}

function conda_install {
    conda install --yes "$1"
}

function pip_install {
    pip install "$1"
}

conda create --yes --prefix "$ENV" python
source activate "$ENV"

# https://github.com/ContinuumIO/anaconda-issues/issues/32
cp .qt.conf "$ENV/bin/qt.conf"

echo "$ENV" > "$ROOT/.env"

safe_call conda_install numpy
safe_call conda_install scipy
safe_call conda_install scikit-learn
safe_call conda_install matplotlib pil
