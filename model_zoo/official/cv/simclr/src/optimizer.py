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
"""optimizer generator"""
from mindspore import nn, Tensor
from .lr_generator import get_lr

def get_train_optimizer(net, steps_per_epoch, args):
    """
    generate optimizer for updating the weights.
    """
    if args.optimizer == "Adam":
        lr = get_lr(lr_init=1e-4, lr_end=1e-6, lr_max=9e-4,
                    warmup_epochs=args.warmup_epochs, total_epochs=args.epoch_size,
                    steps_per_epoch=steps_per_epoch,
                    lr_decay_mode="linear")
        lr = Tensor(lr)
        decayed_params = []
        no_decayed_params = []
        for param in net.trainable_params():
            if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
                decayed_params.append(param)
            else:
                no_decayed_params.append(param)
        group_params = [{'params': decayed_params, 'weight_decay': args.weight_decay},
                        {'params': no_decayed_params},
                        {'order_params': net.trainable_params()}]
        optimizer = nn.Adam(params=group_params, learning_rate=lr)
    else:
        raise ValueError("Unsupported optimizer.")

    return optimizer

def get_eval_optimizer(net, steps_per_epoch, args):
    lr = get_lr(lr_init=1e-3, lr_end=6e-6, lr_max=1e-2,
                warmup_epochs=5, total_epochs=args.epoch_size,
                steps_per_epoch=steps_per_epoch,
                lr_decay_mode="linear")
    lr = Tensor(lr)
    optimizer = nn.Adam(params=net.trainable_params(), learning_rate=lr)
    return optimizer
