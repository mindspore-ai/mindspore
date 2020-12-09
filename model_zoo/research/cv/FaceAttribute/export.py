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

from mindspore import context
from mindspore import Tensor
from mindspore.train.serialization import export, load_checkpoint, load_param_into_net

from src.FaceAttribute.resnet18_softmax import get_resnet18
from src.config import config

devid = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, device_id=devid)


def main(args):
    network = get_resnet18(args)
    ckpt_path = args.model_path
    if os.path.isfile(ckpt_path):
        param_dict = load_checkpoint(ckpt_path)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('network.'):
                param_dict_new[key[8:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(network, param_dict_new)
        print('-----------------------load model success-----------------------')
    else:
        print('-----------------------load model failed -----------------------')

    input_data = np.random.uniform(low=0, high=1.0, size=(args.batch_size, 3, 112, 112)).astype(np.float32)
    tensor_input_data = Tensor(input_data)

    export(network, tensor_input_data, file_name=ckpt_path.replace('.ckpt', '_' + str(args.batch_size) + 'b.air'),
           file_format='AIR')
    print('-----------------------export model success-----------------------')

def parse_args():
    """parse_args"""
    parser = argparse.ArgumentParser(description='Convert ckpt to air')
    parser.add_argument('--model_path', type=str, default='', help='pretrained model to load')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')

    args_opt = parser.parse_args()
    return args_opt

if __name__ == "__main__":
    args_1 = parse_args()

    args_1.dst_h = config.dst_h
    args_1.dst_w = config.dst_w
    args_1.attri_num = config.attri_num
    args_1.classes = config.classes
    args_1.flat_dim = config.flat_dim
    args_1.fc_dim = config.fc_dim

    main(args_1)
