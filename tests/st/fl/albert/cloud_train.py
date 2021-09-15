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
import os
import sys
import ast
from time import time
import numpy as np
from mindspore import context, Tensor
from mindspore.train.serialization import save_checkpoint, load_checkpoint
from src.adam import AdamWeightDecayOp as AdamWeightDecay
from src.config import train_cfg, server_net_cfg
from src.utils import restore_params
from src.model import AlbertModelCLS
from src.cell_wrapper import NetworkWithCLSLoss, NetworkTrainCell


def parse_args():
    """
    parse args
    """
    parser = argparse.ArgumentParser(description='server task')
    parser.add_argument('--device_target', type=str, default='GPU', choices=['Ascend', 'GPU', 'CPU'])
    parser.add_argument('--device_id', type=str, default='0')
    parser.add_argument('--tokenizer_dir', type=str, default='../model_save/init/')
    parser.add_argument('--server_data_path', type=str, default='../datasets/semi_supervise/server/train.txt')
    parser.add_argument('--model_path', type=str, default='../model_save/init/albert_init.ckpt')
    parser.add_argument('--output_dir', type=str, default='../model_save/train_server/')
    parser.add_argument('--vocab_map_ids_path', type=str, default='../model_save/init/vocab_map_ids.txt')
    parser.add_argument('--logging_step', type=int, default=1)

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
    parser.add_argument("--dp_eps", type=float, default=50.0)
    parser.add_argument("--dp_delta", type=float, default=0.01)  # usually equals 1/start_fl_job_threshold
    parser.add_argument("--dp_norm_clip", type=float, default=1.0)
    parser.add_argument("--encrypt_type", type=str, default="NOT_ENCRYPT")
    parser.add_argument("--share_secrets_ratio", type=float, default=1.0)
    parser.add_argument("--cipher_time_window", type=int, default=300000)
    parser.add_argument("--reconstruct_secrets_threshold", type=int, default=3)
    parser.add_argument("--client_password", type=str, default="")
    parser.add_argument("--server_password", type=str, default="")
    parser.add_argument("--enable_ssl", type=ast.literal_eval, default=False)
    return parser.parse_args()


def server_train(args):
    start = time()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id
    model_path = args.model_path
    output_dir = args.output_dir

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
    scheduler_manage_port = args.scheduler_manage_port
    dp_delta = args.dp_delta
    dp_norm_clip = args.dp_norm_clip
    encrypt_type = args.encrypt_type
    share_secrets_ratio = args.share_secrets_ratio
    cipher_time_window = args.cipher_time_window
    reconstruct_secrets_threshold = args.reconstruct_secrets_threshold
    client_password = args.client_password
    server_password = args.server_password
    enable_ssl = args.enable_ssl

    # Replace some parameters with federated learning parameters.
    train_cfg.max_global_epoch = fl_iteration_num

    fl_ctx = {
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
        "scheduler_manage_port": scheduler_manage_port,
        "dp_delta": dp_delta,
        "dp_norm_clip": dp_norm_clip,
        "encrypt_type": encrypt_type,
        "share_secrets_ratio": share_secrets_ratio,
        "cipher_time_window": cipher_time_window,
        "reconstruct_secrets_threshold": reconstruct_secrets_threshold,
        "client_password": client_password,
        "server_password": server_password,
        "enable_ssl": enable_ssl
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # mindspore context
    context.set_context(mode=context.GRAPH_MODE, device_target=device_target)
    context.set_fl_context(**fl_ctx)
    print('Context setting is done! Time cost: {}'.format(time() - start))
    sys.stdout.flush()
    start = time()

    # construct model
    albert_model_cls = AlbertModelCLS(server_net_cfg)
    network_with_cls_loss = NetworkWithCLSLoss(albert_model_cls)
    network_with_cls_loss.set_train(True)

    print('Model construction is done! Time cost: {}'.format(time() - start))
    sys.stdout.flush()
    start = time()

    # train prepare
    param_dict = load_checkpoint(model_path)
    if 'learning_rate' in param_dict:
        del param_dict['learning_rate']

    # server optimizer
    server_params = [_ for _ in network_with_cls_loss.trainable_params()]
    server_decay_params = list(
        filter(train_cfg.optimizer_cfg.AdamWeightDecay.decay_filter, server_params)
    )
    server_other_params = list(
        filter(lambda x: not train_cfg.optimizer_cfg.AdamWeightDecay.decay_filter(x), server_params)
    )
    server_group_params = [
        {'params': server_decay_params, 'weight_decay': train_cfg.optimizer_cfg.AdamWeightDecay.weight_decay},
        {'params': server_other_params, 'weight_decay': 0.0},
        {'order_params': server_params}
    ]
    server_optimizer = AdamWeightDecay(server_group_params,
                                       learning_rate=train_cfg.server_cfg.learning_rate,
                                       eps=train_cfg.optimizer_cfg.AdamWeightDecay.eps)
    server_network_train_cell = NetworkTrainCell(network_with_cls_loss, optimizer=server_optimizer)

    restore_params(server_network_train_cell, param_dict)

    print('Optimizer construction is done! Time cost: {}'.format(time() - start))
    sys.stdout.flush()
    start = time()

    # train process
    for _ in range(1):
        input_ids = Tensor(np.zeros((train_cfg.batch_size, server_net_cfg.seq_length), np.int32))
        attention_mask = Tensor(np.zeros((train_cfg.batch_size, server_net_cfg.seq_length), np.int32))
        token_type_ids = Tensor(np.zeros((train_cfg.batch_size, server_net_cfg.seq_length), np.int32))
        label_ids = Tensor(np.zeros((train_cfg.batch_size,), np.int32))
        model_start_time = time()
        cls_loss = server_network_train_cell(input_ids, attention_mask, token_type_ids, label_ids)
        time_cost = time() - model_start_time
        print('server: cls_loss {} time_cost {}'.format(cls_loss, time_cost))
        sys.stdout.flush()
        del input_ids, attention_mask, token_type_ids, label_ids, cls_loss
        output_path = os.path.join(output_dir, 'final.ckpt')
        save_checkpoint(server_network_train_cell.network, output_path)

    print('Training process is done! Time cost: {}'.format(time() - start))


if __name__ == '__main__':
    args_opt = parse_args()
    server_train(args_opt)
