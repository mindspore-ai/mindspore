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

SSH="ssh -o StrictHostKeyChecking=no"
SCP="scp -o StrictHostKeyChecking=no"

error_msg()
{
         local msg="$*"
         echo "[ERROR]: $msg" 1>&2
         exit 1
}

ssh_pass()
{
         local node="$1"
         local user="$2"
         local passwd="$3"
         shift 3
         local cmd="$*"
         sshpass -p "${passwd}" ${SSH} "${user}"@"${node}" ${cmd}
}

scp_pass()
{
         local node="$1"
         local user="$2"
         local passwd="$3"
         local src="$4"
         local target="$5"
         sshpass -p "${passwd}" ${SCP} -r "${src}" "${user}"@"${node}":"${target}"
}

rscp_pass()
{
         local node="$1"
         local user="$2"
         local passwd="$3"
         local src="$4"
         local target="$5"
         sshpass -p "${passwd}" ${SCP} -r "${user}"@"${node}":"${src}" "${target}"
}

get_rank_size()
{
         local cluster_config=$1
         cat ${cluster_config} | python3 -c 'import sys,json;print(json.load(sys.stdin)["rank_size"])'
}

get_train_dataset()
{
         local cluster_config=$1
         cat ${cluster_config} | python3 -c 'import sys,json;print(json.load(sys.stdin)["train_dataset"])'
}

get_cluster_list()
{
         local cluster_config=$1
         cat ${cluster_config} | python3 -c 'import sys,json;[print(node) for node in json.load(sys.stdin)["cluster"].keys()]' | sort
}

get_node_user()
{
         local cluster_config=$1
         local node=$2
         cat ${cluster_config} | python3 -c 'import sys,json;print(json.load(sys.stdin)["cluster"]['\"${node}\"']["user"])'
}

get_node_passwd()
{
         local cluster_config=$1
         local node=$2
         cat ${cluster_config} | python3 -c 'import sys,json;print(json.load(sys.stdin)["cluster"]['\"${node}\"']["passwd"])'
}

rsync_sshpass()
{
         local node=$1
         local user="$2"
         local passwd="$3"
         scp_pass "${node}" "${user}" "${passwd}" /usr/local/bin/sshpass /usr/local/bin/sshpass
}
