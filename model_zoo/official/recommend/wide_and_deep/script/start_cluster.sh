#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
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

execute_path=$(pwd)
echo ${execute_path}
script_self=$(readlink -f "$0")
SCRIPTPATH=$(dirname "${script_self}")
echo ${SCRIPTPATH}
# shellcheck source=/dev/null
source $SCRIPTPATH/common.sh
cluster_config_path=$1
RANK_SIZE=$(get_rank_size ${cluster_config_path})
RANK_START=0
node_list=$(get_cluster_list ${cluster_config_path})
EPOCH_SIZE=$2
VOCAB_SIZE=$3
EMB_DIM=$4
DATASET=$5
RANK_TABLE_FILE=$6
ENV_SH=$7
MODE=$8

for node in ${node_list}
do
  user=$(get_node_user ${cluster_config_path} ${node})
  passwd=$(get_node_passwd ${cluster_config_path} ${node})
  echo "------------------${user}@${node}---------------------"
  if [ $MODE == "host_device_mix" ] || [ $MODE == "field_slice_host_device_mix" ] || [ $MODE == "forward_unique" ]; then
    ssh_pass ${node} ${user} ${passwd} "mkdir -p ${execute_path}; cd ${execute_path}; bash ${SCRIPTPATH}/run_auto_parallel_train_cluster.sh ${RANK_SIZE} ${RANK_START} ${EPOCH_SIZE} ${VOCAB_SIZE} ${EMB_DIM} ${DATASET} ${ENV_SH} ${MODE} ${RANK_TABLE_FILE}"
  else
    echo "[ERROR] mode is wrong"
    exit 1
  fi
  RANK_START=$[RANK_START+8]
  if [[ $RANK_START -ge $RANK_SIZE ]]; then
    break;
  fi
done