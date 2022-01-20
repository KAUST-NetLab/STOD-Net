#!/bin/bash --login

set -e
export ENV_PREFIX=$PWD/env
export HOROVOD_CUDA_HOME=$CUDA_HOME
export HOROVOD_NCCL_HOME=$ENV_PREFIX
export HOROVOD_GPU_OPERATIONS=NCCL
conda env create --prefix $ENV_PREFIX --file environment.yml --force
conda activate $ENV_PREFIX
. postBuild