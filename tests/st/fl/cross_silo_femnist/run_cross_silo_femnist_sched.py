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

import argparse
import subprocess

parser = argparse.ArgumentParser(description="Run test_cross_silo_femnist.py case")
parser.add_argument("--device_target", type=str, default="CPU")
parser.add_argument("--server_mode", type=str, default="FEDERATED_LEARNING")
parser.add_argument("--worker_num", type=int, default=1)
parser.add_argument("--server_num", type=int, default=2)
parser.add_argument("--scheduler_ip", type=str, default="127.0.0.1")
parser.add_argument("--scheduler_port", type=int, default=8113)
parser.add_argument("--scheduler_manage_port", type=int, default=11202)
parser.add_argument("--config_file_path", type=str, default="")
parser.add_argument("--dataset_path", type=str, default="")

args, _ = parser.parse_known_args()
device_target = args.device_target
server_mode = args.server_mode
worker_num = args.worker_num
server_num = args.server_num
scheduler_ip = args.scheduler_ip
scheduler_port = args.scheduler_port
scheduler_manage_port = args.scheduler_manage_port
config_file_path = args.config_file_path
dataset_path = args.dataset_path

cmd_sched = "execute_path=$(pwd) && self_path=$(dirname \"${script_self}\") && rm -rf ${execute_path}/scheduler/ &&"
cmd_sched += "mkdir ${execute_path}/scheduler/ &&"
cmd_sched += "cd ${execute_path}/scheduler/ || exit && export GLOG_v=1 &&"
cmd_sched += "python ${self_path}/../test_cross_silo_femnist.py"
cmd_sched += " --device_target=" + device_target
cmd_sched += " --server_mode=" + server_mode
cmd_sched += " --ms_role=MS_SCHED"
cmd_sched += " --worker_num=" + str(worker_num)
cmd_sched += " --server_num=" + str(server_num)
cmd_sched += " --config_file_path=" + str(config_file_path)
cmd_sched += " --scheduler_ip=" + scheduler_ip
cmd_sched += " --scheduler_port=" + str(scheduler_port)
cmd_sched += " --scheduler_manage_port=" + str(scheduler_manage_port)
cmd_sched += " --dataset_path=" + str(dataset_path)
cmd_sched += " --user_id=" + str(0)
cmd_sched += " > scheduler.log 2>&1 &"

subprocess.call(['bash', '-c', cmd_sched])
