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
##############export checkpoint file into air and onnx models#################
python export.py --net squeezenet --dataset cifar10 --checkpoint_path squeezenet_cifar10-120_1562.ckpt
"""

import argparse
import numpy as np
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image classification')
    parser.add_argument('--net', type=str, default='squeezenet', choices=['squeezenet', 'squeezenet_residual'],
                        help='Model.')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'imagenet'], help='Dataset.')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')
    args_opt = parser.parse_args()

    if args_opt.net == "squeezenet":
        from src.squeezenet import SqueezeNet as squeezenet
    else:
        from src.squeezenet import SqueezeNet_Residual as squeezenet
    if args_opt.dataset == "cifar10":
        num_classes = 10
    else:
        num_classes = 1000

    onnx_filename = args_opt.net + '_' + args_opt.dataset
    air_filename = args_opt.net + '_' + args_opt.dataset

    net = squeezenet(num_classes=num_classes)

    assert args_opt.checkpoint_path is not None, "checkpoint_path is None."

    param_dict = load_checkpoint(args_opt.checkpoint_path)
    load_param_into_net(net, param_dict)

    input_arr = Tensor(np.zeros([1, 3, 227, 227], np.float32))
    export(net, input_arr, file_name=onnx_filename, file_format="ONNX")
    export(net, input_arr, file_name=air_filename, file_format="AIR")
