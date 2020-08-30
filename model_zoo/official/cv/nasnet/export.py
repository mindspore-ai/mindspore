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
##############export checkpoint file into geir and onnx models#################
"""
import argparse
import numpy as np

import mindspore as ms
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export

from src.config import nasnet_a_mobile_config_gpu as cfg
from src.nasnet_a_mobile import NASNetAMobile

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='checkpoint export')
    parser.add_argument('--checkpoint', type=str, default='', help='checkpoint of nasnet_a_mobile (Default: None)')
    args_opt = parser.parse_args()

    net = NASNetAMobile(num_classes=cfg.num_classes, is_training=False)
    param_dict = load_checkpoint(args_opt.checkpoint)
    load_param_into_net(net, param_dict)

    input_arr = Tensor(np.random.uniform(0.0, 1.0, size=[1, 3, cfg.image_size, cfg.image_size]), ms.float32)
    export(net, input_arr, file_name=cfg.onnx_filename, file_format="ONNX")
    export(net, input_arr, file_name=cfg.geir_filename, file_format="GEIR")
