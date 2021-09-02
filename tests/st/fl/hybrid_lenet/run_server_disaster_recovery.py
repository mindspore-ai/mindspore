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

# The script runs the process of server's disaster recovery. It will kill the server process and launch it again.

import os
import ast
import argparse
import subprocess

parser = argparse.ArgumentParser(description="Run test_hybrid_train_lenet.py case")
parser.add_argument("--device_target", type=str, default="CPU")
parser.add_argument("--server_mode", type=str, default="HYBRID_TRAINING")
parser.add_argument("--worker_num", type=int, default=1)
parser.add_argument("--server_num", type=int, default=2)
parser.add_argument("--scheduler_ip", type=str, default="127.0.0.1")
parser.add_argument("--scheduler_port", type=int, default=8113)
#The fl server port of the server which needs to be killed.
parser.add_argument("--disaster_recovery_server_port", type=int, default=10976)
parser.add_argument("--node_id", type=str, default="")
parser.add_argument("--start_fl_job_threshold", type=int, default=1)
parser.add_argument("--start_fl_job_time_window", type=int, default=3000)
parser.add_argument("--update_model_ratio", type=float, default=1.0)
parser.add_argument("--update_model_time_window", type=int, default=3000)
parser.add_argument("--fl_name", type=str, default="Lenet")
parser.add_argument("--fl_iteration_num", type=int, default=25)
parser.add_argument("--client_epoch_num", type=int, default=20)
parser.add_argument("--client_batch_size", type=int, default=32)
parser.add_argument("--client_learning_rate", type=float, default=0.1)
parser.add_argument("--local_server_num", type=int, default=-1)
parser.add_argument("--root_first_ca_path", type=str, default="")
parser.add_argument("--root_second_ca_path", type=str, default="")
parser.add_argument("--pki_verify", type=ast.literal_eval, default=False)
parser.add_argument("--root_first_crl_path", type=str, default="")
parser.add_argument("--root_second_crl_path", type=str, default="")
parser.add_argument("--sts_jar_path", type=str, default="")
parser.add_argument("--sts_properties_path", type=str, default="")
parser.add_argument("--dp_eps", type=float, default=50.0)
parser.add_argument("--dp_delta", type=float, default=0.01)  # usually equals 1/start_fl_job_threshold
parser.add_argument("--dp_norm_clip", type=float, default=1.0)
parser.add_argument("--encrypt_type", type=str, default="NOT_ENCRYPT")
parser.add_argument("--config_file_path", type=str, default="")
parser.add_argument("--client_password", type=str, default="")
parser.add_argument("--server_password", type=str, default="")
parser.add_argument("--enable_ssl", type=ast.literal_eval, default=False)


args, _ = parser.parse_known_args()
device_target = args.device_target
server_mode = args.server_mode
worker_num = args.worker_num
server_num = args.server_num
scheduler_ip = args.scheduler_ip
scheduler_port = args.scheduler_port
disaster_recovery_server_port = args.disaster_recovery_server_port
node_id = args.node_id
start_fl_job_threshold = args.start_fl_job_threshold
start_fl_job_time_window = args.start_fl_job_time_window
update_model_ratio = args.update_model_ratio
update_model_time_window = args.update_model_time_window
fl_name = args.fl_name
fl_iteration_num = args.fl_iteration_num
client_epoch_num = args.client_epoch_num
client_batch_size = args.client_batch_size
client_learning_rate = args.client_learning_rate
local_server_num = args.local_server_num
config_file_path = args.config_file_path
root_first_ca_path = args.root_first_ca_path
root_second_ca_path = args.root_second_ca_path
pki_verify = args.pki_verify
root_first_crl_path = args.root_first_crl_path
root_second_crl_path = args.root_second_crl_path
sts_jar_path = args.sts_jar_path
sts_properties_path = args.sts_properties_path
dp_eps = args.dp_eps
dp_delta = args.dp_delta
dp_norm_clip = args.dp_norm_clip
encrypt_type = args.encrypt_type
client_password = args.client_password
server_password = args.server_password
enable_ssl = args.enable_ssl


#Step 1: make the server offline.
offline_cmd = "ps_demo_id=`ps -ef | grep " + str(disaster_recovery_server_port) \
              + "|grep -v cd  | grep -v grep | grep -v run_server_disaster_recovery | awk '{print $2}'`"
offline_cmd += " && for id in $ps_demo_id; do kill -9 $id && echo \"Killed server process: $id\"; done"
subprocess.call(['bash', '-c', offline_cmd])

#Step 2: Wait 3 seconds for recovery.
wait_cmd = "echo \"Start to sleep for 3 seconds\" && sleep 3"
subprocess.call(['bash', '-c', wait_cmd])

#Step 3: Launch the server again with the same fl server port.
os.environ['MS_NODE_ID'] = str(node_id)
cmd_server = "execute_path=$(pwd) && self_path=$(dirname \"${script_self}\") && "
cmd_server += "rm -rf ${execute_path}/disaster_recovery_server_" + str(disaster_recovery_server_port) + "/ &&"
cmd_server += "mkdir ${execute_path}/disaster_recovery_server_" + str(disaster_recovery_server_port) + "/ &&"
cmd_server += "cd ${execute_path}/disaster_recovery_server_" + str(disaster_recovery_server_port)\
              + "/ || exit && export GLOG_v=1 &&"
cmd_server += "python ${self_path}/../test_hybrid_train_lenet.py"
cmd_server += " --device_target=" + device_target
cmd_server += " --server_mode=" + server_mode
cmd_server += " --ms_role=MS_SERVER"
cmd_server += " --worker_num=" + str(worker_num)
cmd_server += " --server_num=" + str(server_num)
cmd_server += " --scheduler_ip=" + scheduler_ip
cmd_server += " --scheduler_port=" + str(scheduler_port)
cmd_server += " --fl_server_port=" + str(disaster_recovery_server_port)
cmd_server += " --start_fl_job_threshold=" + str(start_fl_job_threshold)
cmd_server += " --start_fl_job_time_window=" + str(start_fl_job_time_window)
cmd_server += " --update_model_ratio=" + str(update_model_ratio)
cmd_server += " --update_model_time_window=" + str(update_model_time_window)
cmd_server += " --fl_name=" + fl_name
cmd_server += " --fl_iteration_num=" + str(fl_iteration_num)
cmd_server += " --client_epoch_num=" + str(client_epoch_num)
cmd_server += " --client_batch_size=" + str(client_batch_size)
cmd_server += " --client_learning_rate=" + str(client_learning_rate)
cmd_server += " --dp_eps=" + str(dp_eps)
cmd_server += " --dp_delta=" + str(dp_delta)
cmd_server += " --dp_norm_clip=" + str(dp_norm_clip)
cmd_server += " --encrypt_type=" + str(encrypt_type)
cmd_server += " --config_file_path=" + str(config_file_path)
cmd_server += " --root_first_ca_path=" + str(root_first_ca_path)
cmd_server += " --root_second_ca_path=" + str(root_second_ca_path)
cmd_server += " --pki_verify=" + str(pki_verify)
cmd_server += " --root_first_crl_path=" + str(root_first_crl_path)
cmd_server += " --root_second_crl_path=" + str(root_second_crl_path)
cmd_server += " --root_second_crl_path=" + str(root_second_crl_path)
cmd_server += " --client_password=" + str(client_password)
cmd_server += " --server_password=" + str(server_password)
cmd_server += " --enable_ssl=" + str(enable_ssl)
cmd_server += " --sts_jar_path=" + str(sts_jar_path)
cmd_server += " --sts_properties_path=" + str(sts_properties_path)
cmd_server += " > server.log 2>&1 &"

subprocess.call(['bash', '-c', cmd_server])
