# Copyright 2024 Huawei Technologies Co., Ltd
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
"""fasterrcnn overfit test"""
import argparse
import ast
import os
from time import time

import numpy as np
import pytest
from src import get_network
from src.train_warpper import TrainOneStepCell
from src.utils import logger
from src.utils.common import init_env
from src.utils.config import Config, load_config, merge
from tests.st.utils import test_utils

import mindspore as ms
from mindspore import nn
from mindspore.amp import DynamicLossScaler, StaticLossScaler


def get_args_train(parents=None):
    parser = argparse.ArgumentParser(description="Train", parents=[parents] if parents else [])
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(current_dir, "config/faster_rcnn/faster_rcnn_resnet50_fpn_1x.yml"),
        help="Config file path",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/home/workspace/mindspore_dataset/fasterrcnn/faster_rcnn.ckpt",
        help="pre trained weights path",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="/home/workspace/mindspore_dataset/fasterrcnn/test_data",
        help="data dir path",
    )
    parser.add_argument("--seed", type=int, default=1234, help="runtime seed")
    parser.add_argument(
        "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)"
    )
    parser.add_argument("--ms_loss_scaler", type=str, default="static", help="train loss scaler, static/dynamic/none")
    parser.add_argument("--ms_loss_scaler_value", type=float, default=256.0, help="static loss scale value")
    parser.add_argument("--num_parallel_workers", type=int, default=8, help="num parallel worker for dataloader")
    parser.add_argument("--device_target", type=str, default="Ascend", help="device target, Ascend/GPU/CPU")
    parser.add_argument("--batch_size", type=int, default=2, help="total batch size for all device")
    parser.add_argument("--mix", type=ast.literal_eval, default=True, help="Mix Precision")
    parser.add_argument("--lr_init", type=float, default=0.0025, help="base learning rate")
    parser.add_argument("--save_dir", type=str, default="output", help="save dir")

    # profiling
    parser.add_argument("--run_profilor", type=ast.literal_eval, default=False, help="run profilor")
    # args for ModelArts
    parser.add_argument("--enable_modelarts", type=ast.literal_eval, default=False, help="enable modelarts")
    parser.add_argument("--data_url", type=str, default="", help="ModelArts: obs path to dataset folder")
    parser.add_argument("--ckpt_url", type=str, default="", help="ModelArts: obs path to dataset folder")
    parser.add_argument("--train_url", type=str, default="", help="ModelArts: obs path to dataset folder")
    parser.add_argument("--data_dir", type=str, default="/cache/data", help="ModelArts: obs path to dataset folder")
    args, _ = parser.parse_known_args()
    return args


def get_optimizer(cfg, params, lr):
    def init_group_params(params, weight_decay):
        decay_params = []
        no_decay_params = []

        for param in params:
            if len(param.data.shape) > 1:
                decay_params.append(param)
            else:
                no_decay_params.append(param)
        return [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params},
            {"order_params": params},
        ]

    weight_decay = cfg.weight_decay
    if cfg.filter_bias_and_bn:
        params = init_group_params(params, weight_decay)
        weight_decay = 0.0

    if cfg.type in ["momentum", "sgd"]:
        opt = nn.Momentum(params, lr, momentum=cfg.momentum, weight_decay=weight_decay, use_nesterov=cfg.nesterov)
        return opt
    raise ValueError(f"Not support {cfg.type}")


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@test_utils.run_test_with_On
def test_faster_rcnn_1p():
    """
    Feature: faster_rcnn 1p test
    Description: Test faster_rcnn 1p overfit training, check the start loss and end loss after 200 steps.
    Expectation: No exception.
    """
    args = get_args_train()
    config, _, _ = load_config(args.config)
    config = merge(args, config)
    config = Config(config)
    config.backbone.pretrained = False
    init_env(config)
    logger.info(config)
    network = get_network(config)
    optimizer = get_optimizer(config.optimizer, network.trainable_params(), config.lr_init)
    if config.mix:
        network.to_float(ms.float32)
        for _, cell in network.cells_and_names():
            if isinstance(cell, (nn.Dense, nn.Conv2d)):
                cell.to_float(ms.float16)

    ms.load_checkpoint(config.ckpt, network)
    logger.info(f"success to load pretrained ckpt {config.ckpt}")

    loss_scaler = StaticLossScaler(1.0)
    if config.ms_loss_scaler == "dynamic":
        loss_scaler = DynamicLossScaler(
            scale_value=config.get("ms_loss_scaler_value", 2 ** 16),
            scale_factor=config.get("scale_factor", 2),
            scale_window=config.get("scale_window", 2000),
        )
    elif config.ms_loss_scaler == "static":
        loss_scaler = StaticLossScaler(config.get("ms_loss_scaler_value", 2 ** 10))

    grad_reducer = nn.Identity()
    if config.rank_size > 1:
        mean = ms.context.get_auto_parallel_context("gradients_mean")
        grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, config.rank_size)
        params_num = len(network.trainable_params())
        ms.set_auto_parallel_context(all_reduce_fusion_config=[params_num // 2, params_num // 3 * 2])

    train_net = TrainOneStepCell(network, optimizer, loss_scaler, grad_reducer, clip_grads=config.backbone.frozen_bn)

    image = ms.Tensor(np.load(os.path.join(config.test_data, "image.npy")))
    gt_bbox = ms.Tensor(np.load(os.path.join(config.test_data, "gt_bbox.npy")))
    gt_class = ms.Tensor(np.load(os.path.join(config.test_data, "gt_class.npy")))
    test_data = (image, gt_bbox, gt_class)
    train_steps = 200
    step_times = 0
    for i in range(train_steps):
        step_start = time()
        loss = train_net(*test_data)
        step_time = time() - step_start
        print(f"step: {i:<3d}, rank: {config.rank}, loss: {loss}", end="  ")
        print(f"step time: {(step_time * 1000):.2f}")
        if isinstance(loss, tuple):
            loss = loss[0]

        if i == 0:
            first_step_time = step_time
            loss_start = loss.asnumpy()
        else:
            step_times += step_time

        if i == 1:
            compile_time = first_step_time - step_time

        if i == train_steps - 1:
            loss_end = loss.asnumpy()

    average_step_time = step_times / 199 * 1000
    print(f"Average step time is: {average_step_time:.2f}ms")
    print(f"Compile time is: {compile_time:.2f}s")
    print(f"Loss start is: {loss_start:.2f}")
    print(f"Loss end   is: {loss_end:.2f}")

    assert 5.08 <= loss_start <= 5.14, f"Loss start should in [5.08, 5.14], but got {loss_start}"
    assert loss_end <= 0.205, f"Loss end should <= 0.2, but got {loss_end}"
    # assert average_step_time < 117.26, f"Average step time should shorter than 117.26ms"
