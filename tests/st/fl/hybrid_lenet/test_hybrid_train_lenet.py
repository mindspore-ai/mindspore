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
import ast
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.nn import WithLossCell
from src.cell_wrapper import TrainOneStepCellWithServerCommunicator
from src.model import LeNet5, PushMetrics
# from src.adam import AdamWeightDecayOp

parser = argparse.ArgumentParser(description="test_hybrid_train_lenet")
parser.add_argument("--device_target", type=str, default="CPU")
parser.add_argument("--server_mode", type=str, default="FEDERATED_LEARNING")
parser.add_argument("--ms_role", type=str, default="MS_WORKER")
parser.add_argument("--worker_num", type=int, default=0)
parser.add_argument("--server_num", type=int, default=1)
parser.add_argument("--scheduler_ip", type=str, default="127.0.0.1")
parser.add_argument("--scheduler_port", type=int, default=8113)
parser.add_argument("--fl_server_port", type=int, default=6666)
parser.add_argument("--start_fl_job_threshold", type=int, default=1)
parser.add_argument("--start_fl_job_time_window", type=int, default=3000)
parser.add_argument("--update_model_ratio", type=float, default=1.0)
parser.add_argument("--update_model_time_window", type=int, default=3000)
parser.add_argument("--fl_name", type=str, default="Lenet")
parser.add_argument("--fl_iteration_num", type=int, default=25)
parser.add_argument("--client_epoch_num", type=int, default=20)
parser.add_argument("--client_batch_size", type=int, default=32)
parser.add_argument("--client_learning_rate", type=float, default=0.1)
parser.add_argument("--worker_step_num_per_iteration", type=int, default=65)
parser.add_argument("--scheduler_manage_port", type=int, default=11202)
parser.add_argument("--config_file_path", type=str, default="")
parser.add_argument("--encrypt_type", type=str, default="NOT_ENCRYPT")
# parameters for encrypt_type='DP_ENCRYPT'
parser.add_argument("--dp_eps", type=float, default=50.0)
parser.add_argument("--dp_delta", type=float, default=0.01)  # 1/worker_num
parser.add_argument("--dp_norm_clip", type=float, default=1.0)
# parameters for encrypt_type='PW_ENCRYPT'
parser.add_argument("--share_secrets_ratio", type=float, default=1.0)
parser.add_argument("--cipher_time_window", type=int, default=300000)
parser.add_argument("--reconstruct_secrets_threshold", type=int, default=3)
parser.add_argument("--client_password", type=str, default="")
parser.add_argument("--server_password", type=str, default="")
parser.add_argument("--enable_ssl", type=ast.literal_eval, default=False)

args, _ = parser.parse_known_args()
device_target = args.device_target
server_mode = args.server_mode
ms_role = args.ms_role
worker_num = args.worker_num
server_num = args.server_num
scheduler_ip = args.scheduler_ip
scheduler_port = args.scheduler_port
fl_server_port = args.fl_server_port
start_fl_job_threshold = args.start_fl_job_threshold
start_fl_job_time_window = args.start_fl_job_time_window
update_model_ratio = args.update_model_ratio
update_model_time_window = args.update_model_time_window
fl_name = args.fl_name
fl_iteration_num = args.fl_iteration_num
client_epoch_num = args.client_epoch_num
client_batch_size = args.client_batch_size
client_learning_rate = args.client_learning_rate
worker_step_num_per_iteration = args.worker_step_num_per_iteration
scheduler_manage_port = args.scheduler_manage_port
config_file_path = args.config_file_path
encrypt_type = args.encrypt_type
share_secrets_ratio = args.share_secrets_ratio
cipher_time_window = args.cipher_time_window
reconstruct_secrets_threshold = args.reconstruct_secrets_threshold
dp_eps = args.dp_eps
dp_delta = args.dp_delta
dp_norm_clip = args.dp_norm_clip
client_password = args.client_password
server_password = args.server_password
enable_ssl = args.enable_ssl

ctx = {
    "enable_fl": True,
    "server_mode": server_mode,
    "ms_role": ms_role,
    "worker_num": worker_num,
    "server_num": server_num,
    "scheduler_ip": scheduler_ip,
    "scheduler_port": scheduler_port,
    "fl_server_port": fl_server_port,
    "start_fl_job_threshold": start_fl_job_threshold,
    "start_fl_job_time_window": start_fl_job_time_window,
    "update_model_ratio": update_model_ratio,
    "update_model_time_window": update_model_time_window,
    "fl_name": fl_name,
    "fl_iteration_num": fl_iteration_num,
    "client_epoch_num": client_epoch_num,
    "client_batch_size": client_batch_size,
    "client_learning_rate": client_learning_rate,
    "worker_step_num_per_iteration": worker_step_num_per_iteration,
    "scheduler_manage_port": scheduler_manage_port,
    "config_file_path": config_file_path,
    "share_secrets_ratio": share_secrets_ratio,
    "cipher_time_window": cipher_time_window,
    "reconstruct_secrets_threshold": reconstruct_secrets_threshold,
    "dp_eps": dp_eps,
    "dp_delta": dp_delta,
    "dp_norm_clip": dp_norm_clip,
    "encrypt_type": encrypt_type,
    "client_password": client_password,
    "server_password": server_password,
    "enable_ssl": enable_ssl
}

context.set_context(mode=context.GRAPH_MODE, device_target=device_target)
context.set_fl_context(**ctx)

if __name__ == "__main__":
    epoch = 50000
    np.random.seed(0)
    network = LeNet5(62)
    push_metrics = PushMetrics()
    if context.get_fl_context("ms_role") == "MS_WORKER":
        # Please do not freeze layers if you want to both get and overwrite these layers to servers, which is meaningless.
        network.conv1.weight.requires_grad = False
        network.conv2.weight.requires_grad = False
        # Get weights before running backbone.
        network.conv1.set_param_fl(pull_from_server=True)
        network.conv2.set_param_fl(pull_from_server=True)

        # Overwrite weights after running optimizers.
        network.fc1.set_param_fl(push_to_server=True)
        network.fc2.set_param_fl(push_to_server=True)
        network.fc3.set_param_fl(push_to_server=True)

    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)
    # net_opt = AdamWeightDecayOp(network.trainable_params(), weight_decay=0.1)
    net_with_criterion = WithLossCell(network, criterion)
    # train_network = TrainOneStepCell(net_with_criterion, net_opt)
    train_network = TrainOneStepCellWithServerCommunicator(net_with_criterion, net_opt)
    train_network.set_train()
    losses = []

    for i in range(epoch):
        data = Tensor(np.random.rand(32, 3, 32, 32).astype(np.float32))
        label = Tensor(np.random.randint(0, 61, (32)).astype(np.int32))
        loss = train_network(data, label).asnumpy()
        if context.get_fl_context("ms_role") == "MS_WORKER":
            if (i + 1) % worker_step_num_per_iteration == 0:
                push_metrics(Tensor(loss, mstype.float32), Tensor(loss, mstype.float32))
        losses.append(loss)
    print(losses)
