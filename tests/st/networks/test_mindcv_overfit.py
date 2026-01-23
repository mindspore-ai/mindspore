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

import numpy as np

import mindspore as ms
from mindspore.communication import get_group_size, get_rank, init

workspace = os.path.dirname(os.path.realpath(__file__))
repo_root = os.path.abspath(os.path.join(workspace, "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from tests.mark_utils import arg_mark
from tests.st.utils import test_utils

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
    if "RANK_ID" in os.environ:
        rank_id = int(os.environ["RANK_ID"])
    else:
        os.environ["RANK_ID"] = str(rank_id)
    if "RANK_SIZE" in os.environ:
        device_num = int(os.environ["RANK_SIZE"])
    else:
        os.environ["RANK_SIZE"] = str(device_num)
    if "DEVICE_ID" in os.environ:
        device_id = int(os.environ["DEVICE_ID"])
    ms.set_context(mode=args.mode, device_id=device_id)
    ms.set_context(deterministic="ON")
    # O2 does not support the non-contiguous tensor.
    ms.set_context(jit_level="O0")

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
    num_params = sum(param.size for param in network.get_parameters())

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
        data1 = ms.Tensor(np.load(os.path.join(test_datapath, "image.npy")))[: args.batch_size, :, :, :] * 1
        data2 = ms.Tensor(np.load(os.path.join(test_datapath, "label.npy")))[: args.batch_size] * 1
    elif args.image_resize == 299:
        data1 = ms.Tensor(np.load(os.path.join(test_datapath, "image_299.npy")))[: args.batch_size, :, :, :] * 1
        data2 = ms.Tensor(np.load(os.path.join(test_datapath, "label_299.npy")))[: args.batch_size] * 1
    else:
        raise ValueError(f"Unsupported image_resize: {args.image_resize}")
    data = (data1, data2)

    train_net = trainer.train_network
    train_net.set_train(True)
    ms.load_checkpoint(os.path.join(test_ckptpath, f"test_{args.model}.ckpt"), train_net)

    step_times = 0
    first_step_time = None
    compile_time = None
    loss_start = None
    loss_end = None
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


def train_entry():
    """Entry point for msrun to execute training."""
    config_path = None
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--config="):
                config_path = arg.split("=", 1)[1]
            elif arg == "--config" and len(sys.argv) > sys.argv.index(arg) + 1:
                config_path = sys.argv[sys.argv.index(arg) + 1]
    if not config_path:
        config_path = os.environ.get("MINDCV_CONFIG_PATH")
    if not config_path:
        error_msg = ("Config path must be provided via --config argument "
                     "or MINDCV_CONFIG_PATH environment variable")
        raise ValueError(error_msg)

    args = parse_args([f"--config={config_path}"])
    _, loss_end, _, _ = train(args)
    rank_id = int(os.environ.get("RANK_ID", "0"))
    result_file = f"/tmp/mindcv_train_result_rank_{rank_id}.txt"
    loss_end_np = loss_end.asnumpy() if hasattr(loss_end, "asnumpy") else np.asarray(loss_end)
    with open(result_file, "w", encoding='utf-8') as f:
        f.write(str(loss_end_np))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2384.0)
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


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_resnet_50_8p():
    """
    Feature: MindCV resnet50 8p test
    Description: Test resnet50 8p overfit training, check the start loss and end loss after 200 steps.
    Expectation: No exception.
    """
    import subprocess

    device_num = 8
    config_path = f"{workspace}/mindcv/configs/resnet/resnet_50_ascend.yaml"
    current_file = os.path.abspath(__file__)

    master_port = 29500
    log_dir = "/tmp/msrun_log_resnet50_8p"
    os.makedirs(log_dir, exist_ok=True)

    msrun_cmd = [
        "msrun",
        f"--worker_num={device_num}",
        f"--local_worker_num={device_num}",
        f"--master_port={master_port}",
        "--join=True",
        f"--log_dir={log_dir}",
        "python",
        "-c",
        f"import sys; sys.path.insert(0, r'{workspace}'); "
        f"import importlib.util; "
        f"spec = importlib.util.spec_from_file_location('test_mindcv_overfit', r'{current_file}'); "
        f"module = importlib.util.module_from_spec(spec); "
        f"spec.loader.exec_module(module); "
        f"import sys; sys.argv = ['train_entry', '--config={config_path}']; "
        f"module.train_entry()"
    ]

    result = subprocess.run(msrun_cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        raise RuntimeError(f"msrun failed with return code {result.returncode}, "
                         f"stderr: {result.stderr}, stdout: {result.stdout}")

    result_file = "/tmp/mindcv_train_result_rank_0.txt"
    if os.path.exists(result_file):
        with open(result_file, "r", encoding='utf-8') as f:
            res0 = float(f.read().strip())
    else:
        raise RuntimeError(f"Result file {result_file} not found")

    assert 0.97 <= res0 <= 1.07, f"Loss start should in [7.25, 7.35], but got {res0}"

    # Cleanup result files
    for i in range(device_num):
        result_file = f"/tmp/mindcv_train_result_rank_{i}.txt"
        if os.path.exists(result_file):
            os.remove(result_file)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=698.0)
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

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=3294.0)
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


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=6568.0)
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
