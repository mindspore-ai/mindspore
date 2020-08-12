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
"""Export MobilenetV2 on ImageNet"""

import argparse
import numpy as np

import mindspore
from mindspore import Tensor
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.quant import quant

from src.mobilenetV2 import mobilenetV2
from src.config import config_ascend

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')
parser.add_argument('--device_target', type=str, default=None, help='Run device target')
args_opt = parser.parse_args()

if __name__ == '__main__':
    cfg = None
    if args_opt.device_target == "Ascend":
        cfg = config_ascend
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)
    else:
        raise ValueError("Unsupported device target: {}.".format(args_opt.device_target))

    # define fusion network
    network = mobilenetV2(num_classes=cfg.num_classes)
    # convert fusion network to quantization aware network
    network = quant.convert_quant_network(network, bn_fold=True, per_channel=[True, False], symmetric=[True, False])
    # load checkpoint
    param_dict = load_checkpoint(args_opt.checkpoint_path)
    load_param_into_net(network, param_dict)

    # export network
    print("============== Starting export ==============")
    inputs = Tensor(np.ones([1, 3, cfg.image_height, cfg.image_width]), mindspore.float32)
    quant.export(network, inputs, file_name="mobilenet_quant", file_format='AIR')
    print("============== End export ==============")
