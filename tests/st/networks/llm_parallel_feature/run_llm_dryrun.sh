#!/bin/bash
# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
set -e
BASE_PATH=$(cd "$(dirname $0)"; pwd)

RANK_SIZE=$1
RANK_LIST=$2
CONFIG_FILE=$3
OUTPUT_FILE=$4
CASE_NAME=$5
CELL_REUSE=${6:-None}
# convert rank list to array
IFS=',' read -r -a array <<< "$RANK_LIST"

unset MS_DEV_GRAPH_KERNEL_FLAGS
export MS_SIMULATION_LEVEL=1
export RANK_SIZE="$RANK_SIZE"
export PYTHONPATH=${BASE_PATH}/../mindformers:${PYTHONPATH}
export MS_ALLOC_CONF="memory_tracker:True,memory_tracker_path:$BASE_PATH/$CASE_NAME"
export MS_DEV_DUMP_IR_PASSES="step_parallel,validate,hwopt_d_after_inline_graph"
if [ "$CELL_REUSE" = "pp" ]; then
  echo "enable lazy inline in pp"
  export ENABLE_LAZY_INLINE=1
fi
if [ "$CELL_REUSE" = "no_pp" ]; then
  echo "enable lazy inline in no pp"
  export ENABLE_LAZY_INLINE_NO_PIPELINE=1
fi
if [ "$GPT_DATASET" = "gpt" ]; then
  echo "using gpt dataset."
  cd $MF_PATH/mindformers/dataset/blended_datasets/
  make
  cd ${BASE_PATH}
fi

bash env_compile_config.sh
for rank_id in "${array[@]}"
do
  export RANK_ID=$rank_id
  rm -rf "$BASE_PATH"/"$CASE_NAME"/rank_${RANK_ID}
  mkdir -p "$BASE_PATH"/"$CASE_NAME"/rank_${RANK_ID}
  python "$BASE_PATH"/../mindformers/run_mindformer.py \
        --config "$CONFIG_FILE" > "$BASE_PATH"/"$CASE_NAME"/rank_${RANK_ID}/"$OUTPUT_FILE" 2>&1
done
bash clear_env_compile_config.sh
