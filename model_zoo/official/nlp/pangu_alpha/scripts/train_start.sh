#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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

# hccl-controller组件生成的rank_table_file
export RANK_TABLE_FILE=/user/serverid/devindex/config/hccl.json

# 解析rank_table_file
function get_json_value()
{
  local json=$1
  local key=$2

  if [[ -z "$3" ]]; then
    local num=1
  else
    local num=$3
  fi

  local value=$(cat "${json}" | awk -F"[,:}]" '{for(i=1;i<=NF;i++){if($i~/'${key}'\042/){print $(i+1)}}}' |
                tr -d '"' | sed -n ${num}p)
  echo ${value}
}

# 检查rank_table_file文件状态
function check_hccl_status()
{
    local retry_times=60
    local retry_interval=5
    for (( n=1;n<=$retry_times;n++ ));do
    {
        local status=$(get_json_value ${RANK_TABLE_FILE} status)
        if [[ "$status" != "completed" ]]; then
            echo "hccl status is not completed, wait 5s and retry." | tee -a hccl.log
            sleep $retry_interval
            continue
        else
            return 0
        fi
    }
    done
    return 1
}

function get_server_id()
{
    local key="server_id"
    local srv_id=$(cat ${RANK_TABLE_FILE} | awk -F"[,:}]" '{for(i=1;i<=NF;i++){if($i~/'${key}'\042/){print $(i+1)}}}' |
                   awk '{print FNR ":" $1}' | grep ${XDL_IP} | awk -F ":" '{print $1}')
    if [[ -z $srv_id || $srv_id -lt 1 ]];then
        return 1
    fi
    srv_id=$(($srv_id-1))
    echo ${srv_id}
}

check_hccl_status
if [[ $? -eq 1 ]]; then
    echo "wait hccl status timeout, train job failed." | tee -a hccl.log
    exit 1
fi

sleep 1


# 获取hccl.json文件中的device_count字段
device_count=$(cat "${RANK_TABLE_FILE}" | grep -o device_id | wc -l)
if [[ "$device_count" -eq 0 ]]; then
    echo "device count is 0, train job failed." | tee -a hccl.log
    exit 1
fi

# 获取hccl.json文件中的server_count字段
server_count=$(get_json_value ${RANK_TABLE_FILE} server_count)
if [[ "$server_count" == "" ]]; then
    echo "server count is 0, train job failed." | tee -a hccl.log
    exit 1
fi

# 训练数据集路径，根据实际情况修改
dataset="/job/data/train_data"

# 单节点训练场景
if [[ "$server_count" == "1" ]]; then
    server_id=0
    if [ ${device_count} -lt 8 ]; then
        echo "Less than 8 card training is not supported for pangu alpha model." | tee log
    fi
    if [ ${device_count} -eq 8 ]; then
        bash main.sh ${device_count} ${server_count} ${RANK_TABLE_FILE} ${server_id} ${dataset}
    fi

# 分布式训练场景
else
    server_id=$(get_server_id)
    if [ $? -eq 1 ];then
        echo "get server id failed."
        exit 1
    fi
    echo "server id is: "${server_id}
    bash main.sh ${device_count} ${server_count} ${RANK_TABLE_FILE} ${server_id} ${dataset}
fi
