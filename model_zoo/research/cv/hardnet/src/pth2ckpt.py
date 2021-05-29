# Copyright 2021 Huawei Technologies Co., Ltd
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
"""pth --> ckpt"""
import argparse
import torch
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor

def replace_self(name1, str1, str2):
    return name1.replace(str1, str2)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--pth_path', type=str, default='/disk3/pth2ckpt/hardnet85.pth',
                    help='pth path')
parser.add_argument('--device_target', type=str, default='cpu',
                    help='device target')
args = parser.parse_args()
print(args)

if __name__ == '__main__':

    par_dict = torch.load(args.pth_path, map_location=args.device_target)
    new_params_list = []

    for name in par_dict:
        param_dict = {}
        parameter = par_dict[name]

        print(name)
        name = replace_self(name, ".layers.", ".layer_list.")
        name = replace_self(name, "base.", "net.layers.")
        name = replace_self(name, "conv", "ConvLayer_Conv")
        name = replace_self(name, "norm", "ConvLayer_BN")
        name = replace_self(name, "base.16.3.weight", "head.dense.weight")
        name = replace_self(name, "base.16.3.bias", "head.dense.bias")
        if name.endswith('ConvLayer_BN.weight'):
            name = name[:name.rfind('ConvLayer_BN.weight')]
            name = name + 'ConvLayer_BN.gamma'
        elif name.endswith('ConvLayer_BN.bias'):
            name = name[:name.rfind('ConvLayer_BN.bias')]
            name = name + 'ConvLayer_BN.beta'
        elif name.endswith('.running_mean'):
            name = name[:name.rfind('.running_mean')]
            name = name + '.moving_mean'
        elif name.endswith('.running_var'):
            name = name[:name.rfind('.running_var')]
            name = name + '.moving_variance'
        print('========================hardnet_name', name)

        param_dict['name'] = name
        param_dict['data'] = Tensor(parameter.numpy())
        new_params_list.append(param_dict)

    save_checkpoint(new_params_list, 'HarDNet85.ckpt')
    