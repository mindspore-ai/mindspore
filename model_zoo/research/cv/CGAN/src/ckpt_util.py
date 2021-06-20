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
"""ckpt_util"""
import os
from mindspore.train.serialization import save_checkpoint, load_checkpoint, load_param_into_net


def save_ckpt(args, G, D, epoch):
    # should remove old ckpt
    save_checkpoint(G, os.path.join(args.ckpt_dir, f"G_{epoch}.ckpt"))


def load_ckpt(args, G, D, epoch):
    if args.ckpt_dir is not None:
        param_G = load_checkpoint(os.path.join(
            args.ckpt_dir, f"G_{epoch}.ckpt"))
        load_param_into_net(G, param_G)
    if args.ckpt_dir is not None and D is not None:
        param_D = load_checkpoint(os.path.join(
            args.ckpt_dir, f"G_{epoch}.ckpt"))
        load_param_into_net(D, param_D)
