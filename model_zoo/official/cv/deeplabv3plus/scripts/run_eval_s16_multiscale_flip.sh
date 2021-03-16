#!/bin/bash
export DEVICE_ID=3
export SLOG_PRINT_TO_STDOUT=0
train_code_path=/PATH/TO/MODEL_ZOO_CODE
eval_path=/PATH/TO/EVAL

if [ -d ${eval_path} ]; then
  rm -rf ${eval_path}
fi
mkdir -p ${eval_path}

python ${train_code_path}/eval.py --data_root=/PATH/TO/DATA  \
                    --data_lst=/PATH/TO/DATA_lst.txt  \
                    --batch_size=16  \
                    --crop_size=513  \
                    --ignore_label=255  \
                    --num_classes=21  \
                    --model=DeepLabV3plus_s16  \
                    --scales=0.5  \
                    --scales=0.75  \
                    --scales=1.0  \
                    --scales=1.25  \
                    --scales=1.75  \
                    --flip  \
                    --freeze_bn  \
                    --ckpt_path=/PATH/TO/PRETRAIN_MODEL >${eval_path}/eval_log 2>&1 &

