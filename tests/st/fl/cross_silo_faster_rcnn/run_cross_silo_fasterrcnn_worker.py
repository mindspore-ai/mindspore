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

parser = argparse.ArgumentParser(description="Run test_cross_silo_fasterrcnn.py case")
parser.add_argument("--device_target", type=str, default="GPU")
parser.add_argument("--server_mode", type=str, default="FEDERATED_LEARNING")
parser.add_argument("--worker_num", type=int, default=1)
parser.add_argument("--server_num", type=int, default=2)
parser.add_argument("--scheduler_ip", type=str, default="127.0.0.1")
parser.add_argument("--scheduler_port", type=int, default=8113)
parser.add_argument("--fl_iteration_num", type=int, default=25)
parser.add_argument("--client_epoch_num", type=int, default=20)
parser.add_argument("--worker_step_num_per_iteration", type=int, default=65)
parser.add_argument("--local_worker_num", type=int, default=-1)
parser.add_argument("--config_file_path", type=str, default="")
parser.add_argument("--dataset_path", type=str, default="")

args, _ = parser.parse_known_args()
device_target = args.device_target
server_mode = args.server_mode
worker_num = args.worker_num
server_num = args.server_num
scheduler_ip = args.scheduler_ip
scheduler_port = args.scheduler_port
fl_iteration_num = args.fl_iteration_num
client_epoch_num = args.client_epoch_num
worker_step_num_per_iteration = args.worker_step_num_per_iteration
local_worker_num = args.local_worker_num
config_file_path = args.config_file_path
dataset_path = args.dataset_path

if local_worker_num == -1:
    local_worker_num = worker_num

assert local_worker_num <= worker_num, "The local worker number should not be bigger than total worker number."
for i in range(local_worker_num):
    cmd_worker = "execute_path=$(pwd) && self_path=$(dirname \"${script_self}\") && "
    cmd_worker += "rm -rf ${execute_path}/worker_" + str(i) + "/ &&"
    cmd_worker += "mkdir ${execute_path}/worker_" + str(i) + "/ &&"
    cmd_worker += "cd ${execute_path}/worker_" + str(i) + "/ || exit && export GLOG_v=1 &&"
    cmd_worker += "export CUDA_VISIBLE_DEVICES=" + str(i+4) + "&&"
    cmd_worker += "python ${self_path}/../test_fl_fasterrcnn.py"
    cmd_worker += " --device_target=" + device_target
    cmd_worker += " --server_mode=" + server_mode
    cmd_worker += " --ms_role=MS_WORKER"
    cmd_worker += " --worker_num=" + str(worker_num)
    cmd_worker += " --server_num=" + str(server_num)
    cmd_worker += " --scheduler_ip=" + scheduler_ip
    cmd_worker += " --scheduler_port=" + str(scheduler_port)
    cmd_worker += " --config_file_path=" + str(config_file_path)
    cmd_worker += " --fl_iteration_num=" + str(fl_iteration_num)
    cmd_worker += " --client_epoch_num=" + str(client_epoch_num)
    cmd_worker += " --worker_step_num_per_iteration=" + str(worker_step_num_per_iteration)
    cmd_worker += " --dataset_path=" + str(dataset_path)
    cmd_worker += " --user_id=" + str(i)
    cmd_worker += " > worker.log 2>&1 &"

    subprocess.call(['bash', '-c', cmd_worker])
