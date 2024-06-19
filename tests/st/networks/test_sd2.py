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

# For more details, please refer to MindCV (https://github.com/mindspore-lab/mindcv)

""" Model training pipeline """
import sys
import importlib
import argparse
from time import time
import logging
import os
import shutil
import yaml
import numpy as np
import pytest
from omegaconf import OmegaConf

import mindspore as ms
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell, FixedLossScaleUpdateCell

workspace = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(workspace, "mindone/examples/stable_diffusion_v2/tests")):
    os.rename(os.path.join(workspace, "mindone/examples/stable_diffusion_v2/tests"),
              os.path.join(workspace, "mindone/examples/stable_diffusion_v2/sd2_tests"))
sys.path.insert(0, os.path.join(workspace, "mindone/examples/stable_diffusion_v2/"))

from ldm.data.dataset import build_dataset
from ldm.modules.logger import set_logger
from ldm.modules.lora import inject_trainable_lora, inject_trainable_lora_to_textencoder
from ldm.modules.train.checkpoint import resume_train_network
from ldm.modules.train.ema import EMA
from ldm.modules.train.lr_schedule import create_scheduler
from ldm.modules.train.optim import build_optimizer
from ldm.modules.train.trainer import TrainOneStepWrapper
from ldm.util import count_params, str2bool


logger = logging.getLogger(__name__)
os.environ["MS_MEMORY_STATISTIC"] = "1"
DATA_PATH = "/home/workspace/mindspore_dataset/mindone/pokemon_blip_one"
CKPT_PATH = "/home/workspace/mindspore_dataset/mindone/sd_v2_base-57526ee4.ckpt"


def init_env(mode=0, debug=False, seed=42):
    ms.set_seed(seed)

    if debug and mode == ms.GRAPH_MODE:  # force PyNative mode when debugging
        logger.warning("Debug mode is on, switching execution mode to PyNative.")
        mode = ms.PYNATIVE_MODE
    device_num = 1
    device_id = int(os.getenv("DEVICE_ID", "0"))
    rank_id = 0

    ms.set_context(
        mode=mode,
        device_target="Ascend",
        device_id=device_id,
        ascend_config={"precision_mode": "allow_fp32_to_fp16"},  # Only effective on Ascend 901B
    )

    return device_id, rank_id, device_num


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_pretrained_model(pretrained_ckpt, net, unet_initialize_random=False):
    logger.info("Loading pretrained model from %s", pretrained_ckpt)
    if os.path.exists(pretrained_ckpt):
        param_dict = ms.load_checkpoint(pretrained_ckpt)

        if unet_initialize_random:
            pnames = list(param_dict.keys())
            # pop unet params from pretrained weight
            for pname in pnames:
                if pname.startswith("model.diffusion_model"):
                    param_dict.pop(pname)
            logger.warning("UNet will be initialized randomly")

        ms.load_param_into_net(net, param_dict)
    else:
        logger.warning("Checkpoint file %s dose not exist!!!", pretrained_ckpt)


def build_model_from_config(config, args=None):
    config = OmegaConf.load(config).model
    if args is not None:
        if args.enable_flash_attention is not None:
            config["params"]["unet_config"]["params"]["enable_flash_attention"] = args.enable_flash_attention
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        if config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    config_params = config.get("params", dict())
    return get_obj_from_str(config["target"])(**config_params)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_config",
        default="",
        type=str,
        help="train config path to load a yaml file that override the default arguments",
    )
    parser.add_argument("--mode", default=0, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument(
        "--replace_small_images",
        default=True,
        type=str2bool,
        help="replace the small-size images with other training samples",
    )
    parser.add_argument("--enable_modelarts", default=False, type=str2bool, help="run codes in ModelArts platform")
    parser.add_argument("--num_workers", default=1, type=int, help="the number of modelarts workers")
    parser.add_argument(
        "--json_data_path",
        default="mindone/examples/stable_diffusion_v2/ldm/data/num_samples_64_part.json",
        type=str,
        help="the path of num_samples.json containing a dictionary with 64 parts. "
        "Each part is a large dictionary containing counts of samples of 533 tar packages.",
    )
    parser.add_argument("--data_path", default="dataset", type=str, help="data path")
    parser.add_argument("--output_path", default="output/", type=str, help="output directory to save training results")
    parser.add_argument(
        "--resume",
        default=False,
        type=str,
        help="resume training, can set True or path to resume checkpoint.(default=False)",
    )
    parser.add_argument("--profile", default=False, type=str2bool, help="Profile or not")
    parser.add_argument("--model_config", default="configs/v1-train-chinese.yaml", type=str, help="model config path")
    parser.add_argument("--custom_text_encoder", default="", type=str, help="use this to plug in custom clip model")
    parser.add_argument(
        "--pretrained_model_path", default="", type=str, help="Specify the pretrained model from this checkpoint"
    )
    parser.add_argument("--use_lora", default=False, type=str2bool, help="use lora finetuning")
    parser.add_argument("--lora_ft_unet", default=True, type=str2bool, help="whether to apply lora finetune to unet")
    parser.add_argument(
        "--lora_ft_text_encoder", default=False, type=str2bool, help="whether to apply lora finetune to text encoder"
    )
    parser.add_argument(
        "--lora_rank",
        default=4,
        type=int,
        help="lora rank. The bigger, the larger the LoRA model will be, but usually gives better generation quality.",
    )
    parser.add_argument("--lora_fp16", default=True, type=str2bool, help="Whether use fp16 for LoRA params.")
    parser.add_argument(
        "--lora_scale",
        default=1.0,
        type=float,
        help="scale, the higher, the more LoRA weights will affect original SD. If 0, LoRA has no effect.",
    )

    parser.add_argument("--unet_initialize_random", default=False, type=str2bool, help="initialize unet randomly")
    parser.add_argument("--dataset_sink_mode", default=False, type=str2bool, help="sink mode")
    parser.add_argument("--optim", default="adamw", type=str, help="optimizer")
    parser.add_argument(
        "--betas", type=float, default=[0.9, 0.999], help="Specify the [beta1, beta2] parameter for the Adam optimizer."
    )
    parser.add_argument("--weight_decay", default=1e-6, type=float, help="Weight decay.")
    parser.add_argument("--seed", default=3407, type=int, help="data path")
    parser.add_argument("--warmup_steps", default=1000, type=int, help="warmup steps")
    parser.add_argument("--train_batch_size", default=4, type=int, help="batch size")
    parser.add_argument("--callback_size", default=1, type=int, help="callback size.")
    parser.add_argument("--start_learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--end_learning_rate", default=1e-7, type=float, help="The end learning rate for Adam.")
    parser.add_argument("--decay_steps", default=0, type=int, help="lr decay steps.")
    parser.add_argument("--scheduler", default="cosine_decay", type=str, help="scheduler.")
    parser.add_argument("--epochs", default=10, type=int, help="epochs")
    parser.add_argument("--init_loss_scale", default=65536, type=float, help="loss scale")
    parser.add_argument("--loss_scale_factor", default=2, type=float, help="loss scale factor")
    parser.add_argument("--scale_window", default=1000, type=float, help="scale window")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="gradient accumulation steps")
    parser.add_argument("--use_ema", default=False, type=str2bool, help="whether use EMA")
    parser.add_argument("--clip_grad", default=False, type=str2bool, help="whether apply gradient clipping")
    parser.add_argument(
        "--enable_flash_attention",
        default=None,
        type=str2bool,
        help="whether enable flash attention. If not None, it will overwrite the value in model config yaml.",
    )
    parser.add_argument("--drop_overflow_update", default=True, type=str2bool, help="drop overflow update")
    parser.add_argument("--loss_scaler_type", default="dynamic", type=str, help="dynamic or static")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="max gradient norm for clipping, effective when `clip_grad` enabled.",
    )

    parser.add_argument("--ckpt_save_interval", default=1, type=int, help="save checkpoint every this epochs or steps")
    parser.add_argument(
        "--step_mode",
        default=False,
        type=str2bool,
        help="whether save ckpt by steps. If False, save ckpt by epochs.",
    )
    parser.add_argument("--random_crop", default=False, type=str2bool, help="random crop")
    parser.add_argument("--filter_small_size", default=True, type=str2bool, help="filter small images")
    parser.add_argument("--image_size", default=512, type=int, help="images size")
    parser.add_argument("--image_filter_size", default=256, type=int, help="image filter size")

    parser.add_argument(
        "--log_level",
        type=str,
        default="logging.INFO",
        help="log level, options: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR",
    )

    abs_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ""))
    default_args = parser.parse_args(args)
    if default_args.train_config:
        default_args.train_config = os.path.join(abs_path, default_args.train_config)
        with open(default_args.train_config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args(args)
    args.model_config = os.path.join(abs_path, args.model_config)

    logger.info(args)
    return args


def main(args):
    ms.set_context(jit_level='O0')
    os.environ["MS_ENABLE_ACLNN"] = "1"
    # init
    _, rank_id, device_num = init_env(
        args.mode,
        seed=args.seed
    )
    # TODO jit_config = ms.JitConfig(jit_level="O0")
    set_logger(name="", output_dir=args.output_path, rank=rank_id, log_level=logging.INFO)

    # build model
    print("===", args.model_config, args.enable_flash_attention)
    latent_diffusion_with_loss = build_model_from_config(args.model_config, args)
    load_pretrained_model(
        args.pretrained_model_path, latent_diffusion_with_loss, unet_initialize_random=args.unet_initialize_random
    )

    # build dataset
    tokenizer = latent_diffusion_with_loss.cond_stage_model.tokenizer
    dataset = build_dataset(
        data_path=args.data_path,
        train_batch_size=args.train_batch_size,
        tokenizer=tokenizer,
        image_size=args.image_size,
        image_filter_size=args.image_filter_size,
        device_num=device_num,
        rank_id=rank_id,
        random_crop=args.random_crop,
        filter_small_size=args.filter_small_size,
        replace=args.replace_small_images,
        enable_modelarts=args.enable_modelarts,
    )

    # lora injection
    if args.use_lora:
        # freeze network
        for param in latent_diffusion_with_loss.get_parameters():
            param.requires_grad = False

        # inject lora params
        num_injected_params = 0
        if args.lora_ft_unet:
            _, unet_lora_params = inject_trainable_lora(
                latent_diffusion_with_loss,
                rank=args.lora_rank,
                use_fp16=args.lora_fp16,
                scale=args.lora_scale,
            )
            num_injected_params += len(unet_lora_params)
        if args.lora_ft_text_encoder:
            _, text_encoder_lora_params = inject_trainable_lora_to_textencoder(
                latent_diffusion_with_loss,
                rank=args.lora_rank,
                use_fp16=args.lora_fp16,
                scale=args.lora_scale,
            )
            num_injected_params += len(text_encoder_lora_params)

        assert (
            len(latent_diffusion_with_loss.trainable_params()) == num_injected_params
        ), "Only lora params {} should be trainable. but got {} trainable params".format(
            num_injected_params, len(latent_diffusion_with_loss.trainable_params())
        )
        # print('Trainable params: ', latent_diffusion_with_loss.model.trainable_params())
    dataset_size = dataset.get_dataset_size()
    if not args.decay_steps:
        args.decay_steps = args.epochs * dataset_size - args.warmup_steps  # fix lr scheduling
        if args.decay_steps <= 0:
            logger.warning(
                "decay_steps is 0, please check epochs, dataset_size and warmup_steps. "
                "Will force decay_steps to be set to 1."
            )
            args.decay_steps = 1

    # build learning rate scheduler
    lr = create_scheduler(
        steps_per_epoch=dataset_size,
        scheduler=args.scheduler,
        lr=args.start_learning_rate,
        min_lr=args.end_learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
        num_epochs=args.epochs,
    )

    # build optimizer
    optimizer = build_optimizer(
        model=latent_diffusion_with_loss,
        name=args.optim,
        betas=args.betas,
        weight_decay=args.weight_decay,
        lr=lr,
    )

    if args.loss_scaler_type == "dynamic":
        loss_scaler = DynamicLossScaleUpdateCell(
            loss_scale_value=args.init_loss_scale, scale_factor=args.loss_scale_factor, scale_window=args.scale_window
        )
    elif args.loss_scaler_type == "static":
        loss_scaler = FixedLossScaleUpdateCell(args.init_loss_scale)
    else:
        raise ValueError

    # resume ckpt
    if rank_id == 0:
        ckpt_dir = os.path.join(args.output_path, "ckpt")
        os.makedirs(ckpt_dir, exist_ok=True)

    if args.resume:
        resume_ckpt = os.path.join(ckpt_dir, "train_resume.ckpt") if isinstance(args.resume, bool) else args.resume

        _, loss_scale, cur_iter, last_overflow_iter = resume_train_network(
            latent_diffusion_with_loss, optimizer, resume_ckpt
        )
        loss_scaler.loss_scale_value = loss_scale
        loss_scaler.cur_iter = cur_iter
        loss_scaler.last_overflow_iter = last_overflow_iter

    # trainer (standalone and distributed)
    ema = (
        EMA(
            latent_diffusion_with_loss,  # .model, #TODO: remove .model if not only train UNet
            ema_decay=0.9999,
        )
        if args.use_ema
        else None
    )

    net_with_grads = TrainOneStepWrapper(
        latent_diffusion_with_loss,
        optimizer=optimizer,
        scale_sense=loss_scaler,
        drop_overflow_update=args.drop_overflow_update,  # TODO: allow config
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        clip_grad=args.clip_grad,
        clip_norm=args.max_grad_norm,
        ema=ema,
    )

    # log
    if rank_id == 0:
        num_params_unet, _ = count_params(latent_diffusion_with_loss.model.diffusion_model)
        num_params_text_encoder, _ = count_params(latent_diffusion_with_loss.cond_stage_model)
        num_params_vae, _ = count_params(latent_diffusion_with_loss.first_stage_model)
        num_params, num_trainable_params = count_params(latent_diffusion_with_loss)

        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
                f"Distributed mode: {args.use_parallel}",
                f"Data path: {args.data_path}",
                f"Num params: {num_params:,} (unet: {num_params_unet:,}, "
                f"text encoder: {num_params_text_encoder:,}, vae: {num_params_vae:,})",
                f"Num trainable params: {num_trainable_params:,}",
                f"Precision: {latent_diffusion_with_loss.model.diffusion_model.dtype}",
                f"Use LoRA: {args.use_lora}",
                f"LoRA rank: {args.lora_rank}",
                f"Learning rate: {args.start_learning_rate}",
                f"Batch size: {args.train_batch_size}",
                f"Weight decay: {args.weight_decay}",
                f"Grad accumulation steps: {args.gradient_accumulation_steps}",
                f"Num epochs: {args.epochs}",
                f"Grad clipping: {args.clip_grad}",
                f"Max grad norm: {args.max_grad_norm}",
                f"EMA: {args.use_ema}",
            ]
        )
        key_info += "\n" + "=" * 50
        logger.info(key_info)

        logger.info("Start training...")
        # backup config files
        shutil.copyfile(args.model_config, os.path.join(args.output_path, "model_config.yaml"))

    # train
    loader = dataset.create_tuple_iterator(output_numpy=False, num_epochs=-1)
    net_with_grads.set_train(True)
    # TODO net_with_grads.set_jit_config(jit_config)
    train_net = net_with_grads

    for epoch in range(args.epochs):
        cost_list = []
        for i, data in enumerate(loader):
            s = time()
            loss, cond, _ = train_net(*data)
            loss = loss.asnumpy()
            assert not np.any(np.isnan(loss)), "loss is nan"
            t = (time() - s) * 1000
            if i == 0 and epoch == 0:
                logger.info("the first step cost %f ms", t)
            else:
                cost_list.append(t)
            logger.info(
                "epoch: %d step: %d loss: %f overflow: %s cost %f ms",
                epoch, i + 1, float(loss), bool(cond.asnumpy()), t
            )
        min_cost = float(np.array(cost_list).min())
        assert min_cost < 700, "min cost need less than 1000 ms"
        logger.info("epoch: %d avg time %f", epoch, float(np.array(cost_list).mean()))
    os.environ.pop("MS_ENABLE_ACLNN")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_sd2_1p():
    """
    Feature: Stable diffusion 2.0 1p test
    Description: Test Stable diffusion 2.0 1p overfit training, check the start loss and end loss
    Expectation: No exception.
    """
    logger.info("process id: %d", os.getpid())
    args = parse_args(
        [
            f"--train_config={workspace}/mindone/examples/stable_diffusion_v2/"
            f"configs/train/train_config_vanilla_v2.yaml",
            f"--model_config={workspace}/v2-train-small.yaml",
            "--epochs=2",
            f"--data_path={DATA_PATH}",
            "--output_path=./output",
            f"--pretrained_model_path={CKPT_PATH}",
            "--unet_initialize_random=True",
            "--enable_flash_attention=False",
        ]
    )
    main(args)


if __name__ == "__main__":
    test_sd2_1p()
