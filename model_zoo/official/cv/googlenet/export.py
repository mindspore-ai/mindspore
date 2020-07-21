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
python export.py
"""
import numpy as np

import mindspore as ms
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export

from src.config import cifar_cfg as cfg
from src.googlenet import GoogleNet


if __name__ == '__main__':
    net = GoogleNet(num_classes=cfg.num_classes)
    param_dict = load_checkpoint(cfg.checkpoint_path)
    load_param_into_net(net, param_dict)

    input_arr = Tensor(np.random.uniform(0.0, 1.0, size=[1, 3, 224, 224]), ms.float32)
    export(net, input_arr, file_name=cfg.onnx_filename, file_format="ONNX")
    export(net, input_arr, file_name=cfg.geir_filename, file_format="GEIR")
