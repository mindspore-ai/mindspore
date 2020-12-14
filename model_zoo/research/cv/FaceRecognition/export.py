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

from src.backbone.resnet import get_backbone

devid = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, device_id=devid)


def main(args):
    network = get_backbone(args)

    ckpt_path = args.pretrained
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

    network.add_flags_recursive(fp16=True)
    network.set_train(False)

    input_data = np.random.uniform(low=0, high=1.0, size=(args.batch_size, 3, 112, 112)).astype(np.float32)
    tensor_input_data = Tensor(input_data)

    file_path = ckpt_path.replace('.ckpt', '_' + str(args.batch_size) + 'b.air')
    export(network, tensor_input_data, file_name=file_path, file_format='AIR')
    print('-----------------------export model success, save file:{}-----------------------'.format(file_path))


def parse_args():
    '''parse_args'''
    parser = argparse.ArgumentParser(description='Convert ckpt to air')
    parser.add_argument('--pretrained', type=str, default='', help='pretrained model to load')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--pre_bn', type=int, default=0, help='1: bn-conv-bn-conv-bn, 0: conv-bn-conv-bn')
    parser.add_argument('--inference', type=int, default=1, help='use inference backbone')
    parser.add_argument('--use_se', type=int, default=0, help='use se block or not')
    parser.add_argument('--emb_size', type=int, default=256, help='embedding size of the network')
    parser.add_argument('--act_type', type=str, default='relu', help='activation layer type')
    parser.add_argument('--backbone', type=str, default='r100', help='backbone network')
    parser.add_argument('--head', type=str, default='0', help='head type, default is 0')
    parser.add_argument('--use_drop', type=int, default=0, help='whether use dropout in network')

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    arg = parse_args()
    main(arg)
