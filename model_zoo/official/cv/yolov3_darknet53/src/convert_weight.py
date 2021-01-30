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
"""Convert weight to mindspore ckpt."""
import os
import argparse
import numpy as np
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor

from src.yolo import YOLOV3DarkNet53

def load_weight(weights_file):
    """Loads pre-trained weights."""
    if not os.path.isfile(weights_file):
        raise ValueError(f'"{weights_file}" is not a valid weight file.')
    with open(weights_file, 'rb') as fp:
        np.fromfile(fp, dtype=np.int32, count=5)
        return np.fromfile(fp, dtype=np.float32)


def build_network():
    """Build YOLOv3 network."""
    network = YOLOV3DarkNet53(is_training=True)
    params = network.get_parameters()
    params = [p for p in params if 'backbone' in p.name]
    return params


def convert(weights_file, output_file):
    """Convert weight to mindspore ckpt."""
    params = build_network()
    weights = load_weight(weights_file)
    index = 0
    param_list = []
    for i in range(0, len(params), 5):
        weight = params[i]
        mean = params[i+1]
        var = params[i+2]
        gamma = params[i+3]
        beta = params[i+4]
        beta_data = weights[index: index+beta.size].reshape(beta.shape)
        index += beta.size
        gamma_data = weights[index: index+gamma.size].reshape(gamma.shape)
        index += gamma.size
        mean_data = weights[index: index+mean.size].reshape(mean.shape)
        index += mean.size
        var_data = weights[index: index + var.size].reshape(var.shape)
        index += var.size
        weight_data = weights[index: index+weight.size].reshape(weight.shape)
        index += weight.size

        param_list.append({'name': weight.name, 'type': weight.dtype, 'shape': weight.shape,
                           'data': Tensor(weight_data)})
        param_list.append({'name': mean.name, 'type': mean.dtype, 'shape': mean.shape, 'data': Tensor(mean_data)})
        param_list.append({'name': var.name, 'type': var.dtype, 'shape': var.shape, 'data': Tensor(var_data)})
        param_list.append({'name': gamma.name, 'type': gamma.dtype, 'shape': gamma.shape, 'data': Tensor(gamma_data)})
        param_list.append({'name': beta.name, 'type': beta.dtype, 'shape': beta.shape, 'data': Tensor(beta_data)})

    save_checkpoint(param_list, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="yolov3 weight convert.")
    parser.add_argument("--input_file", type=str, default="./darknet53.conv.74", help="input file path.")
    parser.add_argument("--output_file", type=str, default="./ackbone_darknet53.ckpt", help="output file path.")
    args_opt = parser.parse_args()

    convert(args_opt.input_file, args_opt.output_file)
