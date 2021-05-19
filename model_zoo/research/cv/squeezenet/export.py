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

import numpy as np
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export
from model_utils.config import config

if __name__ == '__main__':
    if config.net_name == "squeezenet":
        from src.squeezenet import SqueezeNet as squeezenet
    else:
        from src.squeezenet import SqueezeNet_Residual as squeezenet
    if config.dataset == "cifar10":
        num_classes = 10
    else:
        num_classes = 1000

    onnx_filename = config.net_name + '_' + config.dataset
    air_filename = config.net_name + '_' + config.dataset

    net = squeezenet(num_classes=num_classes)

    assert config.checkpoint_file_path is not None, "checkpoint_file_path is None."

    param_dict = load_checkpoint(config.checkpoint_file_path)
    load_param_into_net(net, param_dict)

    input_arr = Tensor(np.zeros([1, 3, 227, 227], np.float32))
    export(net, input_arr, file_name=onnx_filename, file_format="ONNX")
    export(net, input_arr, file_name=air_filename, file_format="AIR")
