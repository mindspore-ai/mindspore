# Copyright 2020 Huawei Technologies Co., Ltd
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
import os
from easydict import EasyDict as ed

def set_config(args):
    if not args.run_distribute:
        args.run_distribute = False
    config_cpu = ed({
        "num_classes": 26,
        "image_height": 224,
        "image_width": 224,
        "batch_size": 150,
        "epoch_size": 15,
        "warmup_epochs": 0,
        "lr_init": .0,
        "lr_end": 0.03,
        "lr_max": 0.03,
        "momentum": 0.9,
        "weight_decay": 4e-5,
        "label_smooth": 0.1,
        "loss_scale": 1024,
        "save_checkpoint": True,
        "save_checkpoint_epochs": 1,
        "keep_checkpoint_max": 20,
        "save_checkpoint_path": "./",
        "platform": args.platform,
        "run_distribute": args.run_distribute,
        "activation": "Softmax"
    })
    config_gpu = ed({
        "num_classes": 1000,
        "image_height": 224,
        "image_width": 224,
        "batch_size": 150,
        "epoch_size": 200,
        "warmup_epochs": 0,
        "lr_init": .0,
        "lr_end": .0,
        "lr_max": 0.8,
        "momentum": 0.9,
        "weight_decay": 4e-5,
        "label_smooth": 0.1,
        "loss_scale": 1024,
        "save_checkpoint": True,
        "save_checkpoint_epochs": 1,
        "keep_checkpoint_max": 200,
        "save_checkpoint_path": "./",
        "platform": args.platform,
        "run_distribute": args.run_distribute,
        "activation": "Softmax"
    })
    config_ascend = ed({
        "num_classes": 1000,
        "image_height": 224,
        "image_width": 224,
        "batch_size": 256,
        "epoch_size": 200,
        "warmup_epochs": 4,
        "lr_init": 0.00,
        "lr_end": 0.00,
        "lr_max": 0.4,
        "momentum": 0.9,
        "weight_decay": 4e-5,
        "label_smooth": 0.1,
        "loss_scale": 1024,
        "save_checkpoint": True,
        "save_checkpoint_epochs": 1,
        "keep_checkpoint_max": 200,
        "save_checkpoint_path": "./",
        "platform": args.platform,
        "device_id": int(os.getenv('DEVICE_ID', '0')),
        "rank_id": int(os.getenv('RANK_ID', '0')),
        "rank_size": int(os.getenv('RANK_SIZE', '1')),
        "run_distribute": int(os.getenv('RANK_SIZE', '1')) > 1.,
        "activation": "Softmax"
    })
    config = ed({"CPU": config_cpu,
                 "GPU": config_gpu,
                 "Ascend": config_ascend})

    if args.platform not in config.keys():
        raise ValueError("Unsupported platform.")

    return config[args.platform]
