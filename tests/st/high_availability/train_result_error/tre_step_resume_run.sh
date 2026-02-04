#!/bin/bash

export HCCL_DETERMINISTIC=true
export ASCEND_LAUNCH_BLOCKING=1

export MS_ENABLE_TFT='{TRE:2, TRE_SNAPSHOT_STEPS:7}'
export MS_ENABLE_CKPT_D2H_ASYNC=1

export VLOG_v='(70001,)'

python "$(dirname $0)"/tre_step_train.py
