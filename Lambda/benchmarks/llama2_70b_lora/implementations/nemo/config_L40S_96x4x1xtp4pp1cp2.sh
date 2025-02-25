#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# hyperparameters
export LR=0.0005
export MAX_STEPS=1024
export MINIBS=1

export TP=4
export PP=1
export CP=2
export SP=1
export TP_COMM_OVERLAP=False
export MC_TP_OVERLAP_RS=False

export FP8=True
export FP8_AMAX_ALGO=most_recent
export FP8_REDUCE_AMAX=False
export FP8_AMAX_HISTORY=4
export FP8_DPA=False
export NVTE_FP8_DPA_BWD=False

export SKIP_EVALS=8
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# system parameters
export DGXNNODES=96
export DGXNGPU=4
export WALLTIME_MINUTES=240
export WALLTIME=$(( (${NEXP:-1} * WALLTIME_MINUTES) ))
