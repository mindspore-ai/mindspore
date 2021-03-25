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
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export
from mindspore.compression.quant import QuantizationAwareTraining

from src.mobilenetV2 import mobilenetV2
from src.config import config_quant

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')
parser.add_argument("--file_format", type=str, choices=["AIR", "MINDIR"], default="MINDIR", help="file format")
parser.add_argument('--device_target', type=str, default=None, help='Run device target')
args_opt = parser.parse_args()

if __name__ == '__main__':
    cfg = config_quant(args_opt.device_target)
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target, save_graphs=False)
    # define fusion network
    network = mobilenetV2(num_classes=cfg.num_classes)
    # convert fusion network to quantization aware network
    quantizer = QuantizationAwareTraining(bn_fold=True,
                                          per_channel=[True, False],
                                          symmetric=[True, False])
    network = quantizer.quantize(network)
    # load checkpoint
    param_dict = load_checkpoint(args_opt.checkpoint_path)
    load_param_into_net(network, param_dict)

    # export network
    print("============== Starting export ==============")
    inputs = Tensor(np.ones([1, 3, cfg.image_height, cfg.image_width]), mindspore.float32)
    if args_opt.file_format == 'MINDIR':
        export(network, inputs, file_name="mobilenet_quant", file_format='MINDIR', quant_mode='AUTO')
    else:
        export(network, inputs, file_name="mobilenet_quant", file_format='AIR',
               quant_mode='AUTO', mean=0., std_dev=48.106)
    print("============== End export ==============")
