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

# For more details, please refer to MindOCR (https://github.com/mindspore-lab/mindocr)


"""
Model training
"""
import os
import sys
import time

import numpy as np
import pytest
from addict import Dict
from mindspore.train.callback._callback import _handle_loss

import mindspore as ms
from mindspore import load_checkpoint, set_context

workspace = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(workspace, "mindocr/tests")):
    os.rename(os.path.join(workspace, "mindocr/tests"), os.path.join(workspace, "mindocr/mindocr_tests"))
sys.path.insert(0, os.path.join(workspace, "mindocr"))

data_dir = "/home/workspace/mindspore_dataset/overfit_test_data/"

from mindocr.losses import build_loss
from mindocr.models import build_model
from mindocr.optim import create_group_params, create_optimizer
from mindocr.utils.ema import EMA
from mindocr.utils.loss_scaler import get_loss_scales
from mindocr.utils.model_wrapper import NetWithLossWrapper
from mindocr.utils.seed import set_seed
from mindocr.utils.train_step_wrapper import TrainOneStepWrapper
from tools.arg_parser import parse_args_and_config


class TestDataSet:
    def __init__(self, base_dir, name, output_columns=None, len_each=800, **kwargs):
        base_dir = os.path.join(base_dir, name)
        np_path = os.path.join(base_dir, f"{name}.npy")
        self.data = np.load(np_path, allow_pickle=True).item()
        self.output_columns = output_columns
        self.len_each = len_each

    def __len__(self):
        return self.len_each

    def __getitem__(self, index):
        return tuple(self.data[k] for k in self.output_columns)


def build_dataset(
        base_dir,
        name,
        dataset_config: dict,
        **kwargs,
        ):
    dataset = TestDataSet(base_dir, name, **dataset_config)

    dataset_column_names = dataset_config.output_columns
    ds = ms.dataset.GeneratorDataset(
        dataset,
        column_names=dataset_column_names,
        num_parallel_workers=1,
    )
    batch_size = 8
    dataloader = ds.batch(
        batch_size,
    )
    return dataloader


def main_test_process(args, cfg):
    basename = os.path.basename(args.config)
    case_name, _ = os.path.splitext(basename)
    # init env
    set_context(mode=cfg.system.mode, deterministic="ON")
    set_seed(42)

    if "DEVICE_ID" in os.environ:
        pass
    else:
        device_id = cfg.system.get("device_id", 0)
        set_context(device_id=device_id)

    loader_train = build_dataset(
        data_dir,
        case_name,
        cfg.train.dataset,
    )

    # create model
    amp_level = cfg.system.get("amp_level", "O0")
    if (
            ms.get_context("device_target") == "GPU"
            and cfg.system.val_while_train
            and amp_level == "O3"
    ):
        amp_level = "O2"
    if cfg.model.backbone.name == "abinet_backbone":
        cfg.model.backbone.batchsize = 8
    if cfg.model.head.name == "ABINetHead":
        cfg.model.head.batchsize = 8
    cfg.model.backbone.pretrained = False
    network = build_model(cfg.model, amp_level=amp_level)

    # create loss
    if cfg.loss.name == "CTCLoss":
        cfg.loss.batch_size = 8
    loss_fn = build_loss(cfg.loss.pop("name"), **cfg["loss"])

    net_with_loss = NetWithLossWrapper(
        network,
        loss_fn,
        input_indices=cfg.train.dataset.pop("net_input_column_index", None),
        label_indices=cfg.train.dataset.pop("label_column_index", None),
        pred_cast_fp32=cfg.train.pop("pred_cast_fp32", amp_level != "O0"),
    )  # wrap train-one-step cell

    # get loss scale setting for mixed precision training
    loss_scale_manager, optimizer_loss_scale = get_loss_scales(cfg)

    # build lr scheduler
    num_epoch = 1 if cfg.model.head.name == 'DBHead' else 3
    base_lr_dict = {
        "db_r50_icdar15": cfg.scheduler.lr / 1000,
        "dbpp_r50_icdar15": cfg.scheduler.lr,
        "crnn_vgg7": cfg.scheduler.lr / 10,
        "abinet_resnet45_en": cfg.scheduler.lr / 10,
        "svtr_tiny": cfg.scheduler.lr / 10,
    }
    base_lr = base_lr_dict[case_name]
    lr_scheduler = [base_lr for _ in range(num_epoch * 100)]

    # build optimizer
    cfg.optimizer.update({"lr": lr_scheduler, "loss_scale": optimizer_loss_scale})
    params = create_group_params(network.trainable_params(), **cfg.optimizer)
    optimizer = create_optimizer(params, **cfg.optimizer)

    # resume ckpt

    # build train step cell
    gradient_accumulation_steps = 1
    clip_grad = cfg.train.get("clip_grad", False)
    use_ema = cfg.train.get("ema", False)
    ema = (
        EMA(network, ema_decay=cfg.train.get("ema_decay", 0.9999), updates=0)
        if use_ema
        else None
    )

    train_net = TrainOneStepWrapper(
        net_with_loss,
        optimizer=optimizer,
        scale_sense=loss_scale_manager,
        drop_overflow_update=cfg.system.drop_overflow_update,
        gradient_accumulation_steps=gradient_accumulation_steps,
        clip_grad=clip_grad,
        clip_norm=cfg.train.get("clip_norm", 1.0),
        ema=ema,
    )

    # training
    train_net = ms.Model(train_net)
    train_net = train_net.train_network
    train_net.set_train(True)
    load_checkpoint(os.path.join(data_dir, f"{case_name}/{case_name}.ckpt"), network)

    loss = None
    train_start_time = time.time()
    for cur_epoch in range(num_epoch):
        for cur_step, data in enumerate(loader_train):
            loss = _handle_loss(train_net(*data))
            if cur_epoch == 0 and cur_step == 0:
                first_step_end_time = time.time()
                loss_start = loss.numpy()
    loss_end = loss.numpy()
    train_end_time = time.time()
    average_step_time = (
        (train_end_time - first_step_end_time) / (num_epoch * 100 - 1) * 1000
    )
    time_compile = (first_step_end_time - train_start_time) - average_step_time / 1000
    print(f"Compile time is {time_compile:.2f} s")
    print(f"Start loss is {loss_start:.3f}")
    print(f"Final loss is {loss_end:.3f}")
    print(f"Per step time is {average_step_time:.2f} ms")

    return loss_start, loss_end, average_step_time, time_compile


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_db_r50_1p():
    """
    Feature: MindOCR dbnet-resnet50 1p test
    Description: Test dbnet-resnet50 1p overfit training, check the start loss and end loss after 100 steps.
    Expectation: No exception.
    """
    args, config = parse_args_and_config(
        [f"--config={workspace}/mindocr/configs/det/dbnet/db_r50_icdar15.yaml"]
    )
    config = Dict(config)

    loss_start, loss_end, _, _ = main_test_process(args, config)

    assert (
        35.55 <= loss_start <= 35.65
    ), f"Loss start should in [35.55, 35.65], but got {loss_start}"
    assert (
        10.06 <= loss_end <= 10.45
    ), f"Loss end should in less than 10.45, but got {loss_end}"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_crnn_vgg7_1p():
    """
    Feature: MindOCR crnn_vgg7 1p test
    Description: Test crnn_vgg7 1p overfit training, check the start loss and end loss after 300 steps.
    Expectation: No exception.
    """
    args, config = parse_args_and_config(
        [f"--config={workspace}/mindocr/configs/rec/crnn/crnn_vgg7.yaml"]
    )
    config = Dict(config)

    loss_start, loss_end, _, _ = main_test_process(args, config)

    assert (
        71.89 <= loss_start <= 71.99
    ), f"Loss start should in [71.89, 71.99], but got {loss_start}"
    assert (
        0.28 <= loss_end <= 0.38
    ), f"Loss end should in [0.28, 0.38], but got {loss_end}"
