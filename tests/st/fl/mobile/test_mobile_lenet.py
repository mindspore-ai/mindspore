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
from mindspore.nn import TrainOneStepCell, WithLossCell
from src.model import LeNet5
from src.adam import AdamWeightDecayOp

parser = argparse.ArgumentParser(description="test_fl_lenet")
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
parser.add_argument("--scheduler_manage_port", type=int, default=11202)
parser.add_argument("--config_file_path", type=str, default="")
parser.add_argument("--pki_verify", type=ast.literal_eval, default=False)
# parameters used for pki_verify=True
# You can download root_first_ca, root_second_ca and equip_crl
# from https://pki.consumer.huawei.com/ca/
parser.add_argument("--root_first_ca_path", type=str, default="")
parser.add_argument("--root_second_ca_path", type=str, default="")
parser.add_argument("--equip_crl_path", type=str, default="")
parser.add_argument("--replay_attack_time_diff", type=int, default=600000)
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
# parameters for 'SIGNDS'
parser.add_argument("--sign_k", type=float, default=0.01)
parser.add_argument("--sign_eps", type=float, default=100)
parser.add_argument("--sign_thr_ratio", type=float, default=0.6)
parser.add_argument("--sign_global_lr", type=float, default=0.1)
parser.add_argument("--sign_dim_out", type=int, default=0)
parser.add_argument("--global_iteration_time_window", type=int, default=3600000)
# parameters for "compression"
parser.add_argument("--upload_compress_type", type=str, default="NO_COMPRESS",
                    choices=["NO_COMPRESS", "DIFF_SPARSE_QUANT"])
parser.add_argument("--upload_sparse_rate", type=float, default=0.5)
parser.add_argument("--download_compress_type", type=str, default="NO_COMPRESS",
                    choices=["NO_COMPRESS", "QUANT"])

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
share_secrets_ratio = args.share_secrets_ratio
cipher_time_window = args.cipher_time_window
reconstruct_secrets_threshold = args.reconstruct_secrets_threshold
fl_name = args.fl_name
fl_iteration_num = args.fl_iteration_num
client_epoch_num = args.client_epoch_num
client_batch_size = args.client_batch_size
client_learning_rate = args.client_learning_rate
scheduler_manage_port = args.scheduler_manage_port
config_file_path = args.config_file_path
dp_eps = args.dp_eps
dp_delta = args.dp_delta
dp_norm_clip = args.dp_norm_clip
encrypt_type = args.encrypt_type
pki_verify = args.pki_verify
root_first_ca_path = args.root_first_ca_path
root_second_ca_path = args.root_second_ca_path
equip_crl_path = args.equip_crl_path
replay_attack_time_diff = args.replay_attack_time_diff
client_password = args.client_password
server_password = args.server_password
enable_ssl = args.enable_ssl
sign_k = args.sign_k
sign_eps = args.sign_eps
sign_thr_ratio = args.sign_thr_ratio
sign_global_lr = args.sign_global_lr
sign_dim_out = args.sign_dim_out
global_iteration_time_window = args.global_iteration_time_window
upload_compress_type = args.upload_compress_type
upload_sparse_rate = args.upload_sparse_rate
download_compress_type = args.download_compress_type

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
    "share_secrets_ratio": share_secrets_ratio,
    "cipher_time_window": cipher_time_window,
    "reconstruct_secrets_threshold": reconstruct_secrets_threshold,
    "fl_name": fl_name,
    "fl_iteration_num": fl_iteration_num,
    "client_epoch_num": client_epoch_num,
    "client_batch_size": client_batch_size,
    "client_learning_rate": client_learning_rate,
    "scheduler_manage_port": scheduler_manage_port,
    "config_file_path": config_file_path,
    "pki_verify": pki_verify,
    "root_first_ca_path": root_first_ca_path,
    "root_second_ca_path": root_second_ca_path,
    "equip_crl_path": equip_crl_path,
    "replay_attack_time_diff": replay_attack_time_diff,
    "dp_eps": dp_eps,
    "dp_delta": dp_delta,
    "dp_norm_clip": dp_norm_clip,
    "encrypt_type": encrypt_type,
    "client_password": client_password,
    "server_password": server_password,
    "enable_ssl": enable_ssl,
    "sign_k": sign_k,
    "sign_eps": sign_eps,
    "sign_thr_ratio": sign_thr_ratio,
    "sign_global_lr": sign_global_lr,
    "sign_dim_out": sign_dim_out,
    "global_iteration_time_window": global_iteration_time_window,
    "upload_compress_type": upload_compress_type,
    "upload_sparse_rate": upload_sparse_rate,
    "download_compress_type": download_compress_type,
}

context.set_context(mode=context.GRAPH_MODE, device_target=device_target)
context.set_fl_context(**ctx)

if __name__ == "__main__":
    epoch = 5
    np.random.seed(0)
    network = LeNet5(62)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)
    net_adam_opt = AdamWeightDecayOp(network.trainable_params(), weight_decay=0.1)
    net_with_criterion = WithLossCell(network, criterion)
    train_network = TrainOneStepCell(net_with_criterion, net_opt)
    train_network.set_train()
    losses = []

    for _ in range(epoch):
        data = Tensor(np.random.rand(32, 3, 32, 32).astype(np.float32))
        label = Tensor(np.random.randint(0, 61, (32)).astype(np.int32))
        loss = train_network(data, label).asnumpy()
        losses.append(loss)
    print(losses)
