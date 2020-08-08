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

SCRIPTPATH="$( cd "$(dirname "$0")" || exit ; pwd -P )"
# shellcheck source=/dev/null
source $SCRIPTPATH/common.sh
cluster_config_path=$1
execute_path=$2
RANK_SIZE=$(get_rank_size ${cluster_config_path})
RANK_START=0
node_list=$(get_cluster_list ${cluster_config_path})

for node in ${node_list}
do
  user=$(get_node_user ${cluster_config_path} ${node})
  passwd=$(get_node_passwd ${cluster_config_path} ${node})
  echo "------------------${user}@${node}---------------------"
  ssh_pass ${node} ${user} ${passwd} "rm -rf ${execute_path}"
  scp_pass ${node} ${user} ${passwd} $SCRIPTPATH/../../wide_and_deep ${execute_path}
  RANK_START=$[RANK_START+8]
  if [[ $RANK_START -ge $RANK_SIZE ]]; then
    break;
  fi
done