# Copyright 2023 Huawei Technologies Co., Ltd
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

# For more details, please refer to MindCV (https://github.com/mindspore-lab/mindcv)

""" Model training pipeline """
import logging
import os
import sys
from time import time
from multiprocessing import Process, Queue

import numpy as np
import pytest

import mindspore as ms
from mindspore.communication import get_group_size, get_rank, init

from tests.st.utils import test_utils

workspace = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(workspace, "mindcv/tests")):
    os.rename(os.path.join(workspace, "mindcv/tests"), os.path.join(workspace, "mindcv/mindcv_tests"))
sys.path.insert(0, os.path.join(workspace, "mindcv"))

from mindcv.loss import create_loss
from mindcv.models import create_model
from mindcv.optim import create_optimizer
from mindcv.scheduler import create_scheduler
from mindcv.utils import create_trainer, get_metrics, require_customized_train_step, set_logger, set_seed

from config import parse_args, save_args


logger = logging.getLogger("mindcv.train")
MINDSPORE_HCCL_CONFIG_PATH = "/home/workspace/mindspore_config/hccl/rank_table_8p.json"


def train(args, device_id=0, rank_id=0, device_num=1):
    os.environ["RANK_ID"] = str(rank_id)
    os.environ["RANK_SIZE"] = str(device_num)
    ms.set_context(mode=args.mode, device_id=device_id)
    ms.set_context(deterministic="ON")
    ms.set_context(jit_level="O2")

    # change learning rate
    args.lr = args.lr / 8
    args.warmup_epochs = 0

    if device_num > 1:
        init()
        rank_id, device_num = get_rank(), get_group_size()
        ms.set_auto_parallel_context(
            device_num=device_num,
            parallel_mode="data_parallel",
            gradients_mean=True,
            # we should but cannot set parameter_broadcast=True, which will cause error on gpu.
        )

    set_seed(args.seed)
    set_logger(name="mindcv", output_dir=args.ckpt_save_dir, rank=rank_id, color=False)
    logger.info(
        "We recommend installing `termcolor` via `pip install termcolor` "
        "and setup logger by `set_logger(..., color=True)`"
    )

    # calculate number of steps in each epoch
    num_batches = 1281168 // args.batch_size
    train_count = args.batch_size

    # create model
    network = create_model(
        model_name=args.model,
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
        pretrained=False,
        checkpoint_path=args.ckpt_path,
        ema=args.ema,
    )
    num_params = sum([param.size for param in network.get_parameters()])

    # create loss
    loss = create_loss(
        name=args.loss,
        reduction=args.reduction,
        label_smoothing=args.label_smoothing,
        aux_factor=args.aux_factor,
    )

    # create learning rate schedule
    lr_scheduler = create_scheduler(
        num_batches,
        scheduler=args.scheduler,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs,
        warmup_factor=args.warmup_factor,
        decay_epochs=args.decay_epochs,
        decay_rate=args.decay_rate,
        milestones=args.multi_step_decay_milestones,
        num_epochs=args.epoch_size,
        num_cycles=args.num_cycles,
        cycle_decay=args.cycle_decay,
        lr_epoch_stair=args.lr_epoch_stair,
    )

    opt_ckpt_path = ""

    # create optimizer
    if (
            args.loss_scale_type == "fixed"
            and args.drop_overflow_update is False
            and not require_customized_train_step(
                args.ema,
                args.clip_grad,
                args.gradient_accumulation_steps,
                args.amp_cast_list)
    ):
        optimizer_loss_scale = args.loss_scale
    else:
        optimizer_loss_scale = 1.0
    optimizer = create_optimizer(
        network.trainable_params(),
        opt=args.opt,
        lr=lr_scheduler,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        nesterov=args.use_nesterov,
        filter_bias_and_bn=args.filter_bias_and_bn,
        loss_scale=optimizer_loss_scale,
        checkpoint_path=opt_ckpt_path,
        eps=args.eps,
    )

    # define eval metrics.
    metrics = get_metrics(args.num_classes)

    # create trainer
    trainer = create_trainer(
        network,
        loss,
        optimizer,
        metrics,
        amp_level=args.amp_level,
        amp_cast_list=args.amp_cast_list,
        loss_scale_type=args.loss_scale_type,
        loss_scale=args.loss_scale,
        drop_overflow_update=args.drop_overflow_update,
        ema=args.ema,
        ema_decay=args.ema_decay,
        clip_grad=args.clip_grad,
        clip_value=args.clip_value,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    essential_cfg_msg = "\n".join(
        [
            "Essential Experiment Configurations:",
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
            f"Number of devices: {device_num if device_num is not None else 1}",
            f"Number of training samples: {train_count}",
            f"Number of classes: {args.num_classes}",
            f"Number of batches: {num_batches}",
            f"Batch size: {args.batch_size}",
            f"Model: {args.model}",
            f"Model parameters: {num_params}",
            f"Number of epochs: {args.epoch_size}",
            f"Optimizer: {args.opt}",
            f"Learning rate: {args.lr}",
            f"LR Scheduler: {args.scheduler}",
            f"Momentum: {args.momentum}",
            f"Weight decay: {args.weight_decay}",
            f"Auto mixed precision: {args.amp_level}",
            f"Loss scale: {args.loss_scale}({args.loss_scale_type})",
        ]
    )
    logger.info(essential_cfg_msg)
    save_args(args, os.path.join(args.ckpt_save_dir, f"{args.model}.yaml"), rank_id)

    logger.info("Start training")

    test_datapath = "/home/workspace/mindspore_dataset/overfit_test_data/test_data"
    test_ckptpath = "/home/workspace/mindspore_dataset/overfit_test_data/test_ckpt"
    train_steps = 200

    if args.image_resize == 224:
        data1 = ms.Tensor(np.load(os.path.join(test_datapath, "image.npy")))[: args.batch_size, :, :, :]
        data2 = ms.Tensor(np.load(os.path.join(test_datapath, "label.npy")))[: args.batch_size]
    elif args.image_resize == 299:
        data1 = ms.Tensor(np.load(os.path.join(test_datapath, "image_299.npy")))[: args.batch_size, :, :, :]
        data2 = ms.Tensor(np.load(os.path.join(test_datapath, "label_299.npy")))[: args.batch_size]
    data = (data1, data2)

    train_net = trainer.train_network
    train_net.set_train(True)
    ms.load_checkpoint(os.path.join(test_ckptpath, f"test_{args.model}.ckpt"), train_net)

    step_times = 0
    for i in range(train_steps):
        step_start = time()
        loss = train_net(*data)
        step_time = time() - step_start
        print(f"step: {i:<3d}, rank: {rank_id}, loss: {loss}", end="  ")
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

    return loss_start, loss_end, average_step_time, compile_time


def compute_process(q, device_id, device_num, args):
    os.environ["RANK_TABLE_FILE"] = MINDSPORE_HCCL_CONFIG_PATH
    _, loss_end, _, _ = train(
        args, device_id=device_id, rank_id=device_id, device_num=device_num
    )
    q.put(loss_end)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_resnet_50_1p():
    """
    Feature: MindCV resnet50 1p test
    Description: Test resnet50 1p overfit training, check the start loss and end loss after 200 steps.
    Expectation: No exception.
    """
    args = parse_args([f"--config={workspace}/mindcv/configs/resnet/resnet_50_ascend.yaml"])

    device_id = int(os.environ.get("DEVICE_ID", "0"))
    loss_start, loss_end, _, _ = train(args, device_id=device_id)

    assert 7.25 <= loss_start <= 7.35, f"Loss start should in [7.25, 7.35], but got {loss_start}"
    assert 0.97 <= loss_end <= 1.07, f"Loss start should in [0.97, 1.07], but got {loss_end}"
    # assert average_step_time < 122.97, f"Average step time should shorter than 122.97"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_resnet_50_8p():
    """
    Feature: MindCV resnet50 8p test
    Description: Test resnet50 8p overfit training, check the start loss and end loss after 200 steps.
    Expectation: No exception.
    """
    q = Queue()
    device_num = 8
    args = parse_args([f"--config={workspace}/mindcv/configs/resnet/resnet_50_ascend.yaml"])
    process = []
    for i in range(device_num):
        device_id = i
        process.append(Process(target=compute_process, args=(q, device_id, device_num, args)))

    for i in range(device_num):
        process[i].start()

    print("Waiting for all subprocesses done...")

    for i in range(device_num):
        process[i].join()

    res0 = q.get()
    assert 0.97 <= res0 <= 1.07, f"Loss start should in [7.25, 7.35], but got {res0}"

@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_mobilenetv3_small_1p():
    """
    Feature: MindCV mobilenetv3 1p test
    Description: Test mobilenetv3 1p overfit training, check the start loss and end loss after 200 steps.
    Expectation: No exception.
    """
    args = parse_args([f"--config={workspace}/mindcv/configs/mobilenetv3/mobilenet_v3_small_ascend.yaml"])

    device_id = int(os.environ.get("DEVICE_ID", "0"))
    loss_start, loss_end, _, _ = train(args, device_id=device_id)

    assert 6.86 <= loss_start <= 6.96, f"Loss start should in [6.86, 6.96], but got {loss_start}"
    assert 1.02 <= loss_end <= 1.12, f"Loss start should in [1.02, 1.12], but got {loss_end}"
    # assert average_step_time < 117.26, f"Average step time should shorter than 117.26ms"


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@test_utils.run_test_with_On
def test_inception_v3_1p():
    """
    Feature: MindCV inception_v3 1p test
    Description: Test inception_v3 1p overfit training, check the start loss and end loss after 200 steps.
    Expectation: No exception.
    """
    args = parse_args([f"--config={workspace}/mindcv/configs/inceptionv3/inception_v3_ascend.yaml"])

    device_id = int(os.environ.get("DEVICE_ID", "0"))
    loss_start, loss_end, _, _ = train(args, device_id=device_id)

    assert 7.59 <= loss_start <= 7.69, f"Loss start should in [7.59, 7.69], but got {loss_start}"
    assert 1.09 <= loss_end <= 1.19, f"Loss start should in [1.09, 1.19], but got {loss_end}"
    # assert average_step_time < 216.74, f"Average step time should shorter than 216.74ms"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@test_utils.run_test_with_On
def test_vit_b32_1p():
    """
    Feature: MindCV vit 1p test
    Description: Test vit 1p overfit training, check the start loss and end loss after 200 steps.
    Expectation: No exception.
    """
    args = parse_args([f"--config={workspace}/mindcv/configs/vit/vit_b32_224_ascend.yaml"])

    device_id = int(os.environ.get("DEVICE_ID", "0"))
    loss_start, loss_end, _, _ = train(args, device_id=device_id)

    assert 7.04 <= loss_start <= 7.14, f"Loss start should in [7.04, 7.14], but got {loss_start}"
    assert 0.98 <= loss_end <= 1.08, f"Loss start should in [0.98, 1.08], but got {loss_end}"
    # assert average_step_time < 809.58, f"Average step time should shorter than 809.58ms"
