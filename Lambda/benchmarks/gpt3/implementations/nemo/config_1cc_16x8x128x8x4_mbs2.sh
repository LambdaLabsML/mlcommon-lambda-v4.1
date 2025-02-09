# dryrun "med" config

## DL params
export MINIBS=128
export TENSOR_MODEL_PARALLEL=8   #  training.model.tensor_model_parallel_size
export PIPELINE_MODEL_PARALLEL=4 #  training.model.pipeline_model_parallel_size
export DGXNNODES=16
#=======================================================================
## System run parms
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_MINUTES=180
export WALLTIME=$(( (${NEXP:-1} * WALLTIME_MINUTES) ))

## System config params
if [[ "${MLPERF_VERSION:-}" == "v31" ]]; then
    source $(dirname ${BASH_SOURCE[0]})/config_common_v31.sh
else
    source $(dirname ${BASH_SOURCE[0]})/config_common_1cc.sh
fi
source $(dirname ${BASH_SOURCE[0]})/config_fp8.sh

export MICRO_BATCH_SIZE=2

export TP_COMM_OVERLAP=True

#Set values for nvidia-smi boost-slider --vboost
if [ $MINIBS -eq 128 ];  then
    export VBOOST_VALUE=1
fi

# Rule: GBS % (DP * PP * MICRO_BATCH_SIZE) == 0
# This simplifies to MINIBS % PP == 0
if [[ $(($MINIBS % PIPELINE_MODEL_PARALLEL)) != 0 ]]; then
    echo "MINIBS should be divisble by PP"
    exit 1
fi
export INTERLEAVED_PIPELINE=4
export LIMIT_TRAIN_BATCHES=2000 # a 64xnode cluster uses 500, scaling accordingly
export LR=5e-6
export MIN_LR=5e-7

timestamp=$(date +'%y-%m-%d_%H-%M-%S')
export LOGDIR=./results/1cc_16x8x128x8x4_mbs2_${timestamp}
mkdir -p ${LOGDIR}