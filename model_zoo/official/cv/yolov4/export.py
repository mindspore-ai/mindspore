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
"""Convert ckpt to air."""
import os
import argparse

import numpy as np

import mindspore
from mindspore import context
from mindspore import Tensor
from mindspore.train.serialization import export, load_checkpoint, load_param_into_net

from src.yolo import YOLOV4CspDarkNet53

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)

def save_air():
    """Save mindir file"""
    print('============= YOLOV4 start save air ==================')

    parser = argparse.ArgumentParser(description='Convert ckpt to air')
    parser.add_argument('--pretrained', type=str, default='', help='pretrained model to load')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')

    args = parser.parse_args()
    network = YOLOV4CspDarkNet53(is_training=False)
    input_shape = Tensor(tuple([416, 416]), mindspore.float32)
    if os.path.isfile(args.pretrained):
        param_dict = load_checkpoint(args.pretrained)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('yolo_network.'):
                param_dict_new[key[13:]] = values

            else:
                param_dict_new[key] = values

        load_param_into_net(network, param_dict_new)
        print('load model {} success'.format(args.pretrained))

        input_data = np.random.uniform(low=0, high=1.0, size=(args.batch_size, 3, 416, 416)).astype(np.float32)

        tensor_input_data = Tensor(input_data)
        export(network, tensor_input_data, input_shape, file_name='yolov4.air', file_format='AIR')

        print("export model success.")


if __name__ == "__main__":
    save_air()
