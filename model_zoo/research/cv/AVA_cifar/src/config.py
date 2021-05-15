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
"""
network config setting, will be used in train.py and eval.py
"""

import time
import json
import logging
from easydict import EasyDict as ed

def get_config():
    """config of training"""

    time_prefix = time.strftime("-%Y%m%d-%H%M%S", time.localtime())
    prefix = "AVA-cifar10-resnet18"
    config = ed({
        # base setting
        "description": "Your description for training",
        "prefix": prefix,
        "time_prefix": time_prefix,
        "net_work": "resnet18",
        "low_dims": 128,
        "use_MLP": False,

        # save
        "save_checkpoint": True,
        "save_checkpoint_epochs": 5,
        "keep_checkpoint_max": 2,

        # optimizer
        "base_lr": 0.03,
        "type": "SGD",
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "loss_scale": 1,
        "sigma": 0.1,

        # trainer
        "breakpoint_training_path": "",
        "batch_size": 128,
        "epochs": 1000,
        "epoch_stage": [600, 400],
        "lr_schedule": "cosine_lr",
        "lr_mode": "epoch",
        "warmup_epoch": 0,
    })
    return config


def save_config(paths, config, args_opt):
    if not isinstance(paths, list):
        paths = [paths]
    for path in paths:
        file = open(path, "w")
        dicts = dict(config, **args_opt)
        json.dump(dicts, file, indent=4)
        file.close()


def get_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode="w+")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
