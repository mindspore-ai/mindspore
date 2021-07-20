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
"""pth2ckpt."""
from mindspore import Tensor
from mindspore.train.serialization import save_checkpoint
import torch
import numpy as np


def pytorch2mindspore():
    """pth to ckpt."""
    par_dict = torch.load('/opt_data/xidian_wks/czp/midas/ckpt/ig_resnext101_32x8-c38310e5.pth', map_location='cpu')
    new_params_list = []
    for name in par_dict:
        print(name)
        param_dict = {}
        parameter = par_dict[name]
        name = name.replace('layer', 'backbone_layer', 1)
        name = name.replace('running_mean', 'moving_mean', 1)
        name = name.replace('running_var', 'moving_variance', 1)
        temp = name
        if name.endswith('conv2.weight'):
            x = parameter.numpy()
            y = np.split(x, 32)
            for i in range(32):
                name = temp[:temp.rfind('weight')] + 'convs.' + str(i) + '.weight'
                data = Tensor(y[i])
                new_params_list.append({"name": name, 'data': data})
            continue
        if name.startswith('bn1'):
            name = name.replace('bn1', 'backbone_bn', 1)
            name = name.replace('bias', 'beta', 1)
            name = name.replace('weight', 'gamma', 1)
        if name.startswith('conv1.weight'):
            name = 'backbone_conv.weight'
        if name.endswith('layer1.0.weight'):
            name = 'backbone_conv.weight'
        if name.endswith('layer1.1.weight'):
            name = 'backbone_bn.gamma'
        if name.endswith('layer1.1.bias'):
            name = 'backbone_bn.beta'
        if name.endswith('bn1.weight'):
            name = name[:name.rfind('weight')]
            name = name + 'gamma'
        if name.endswith('bn1.bias'):
            name = name[:name.rfind('bias')]
            name = name + 'beta'
        if name.endswith('bn2.weight'):
            name = name[:name.rfind('weight')]
            name = name + 'gamma'
        if name.endswith('bn2.bias'):
            name = name[:name.rfind('bias')]
            name = name + 'beta'
        if name.endswith('bn3.weight'):
            name = name[:name.rfind('weight')]
            name = name + 'gamma'
        if name.endswith('bn3.bias'):
            name = name[:name.rfind('bias')]
            name = name + 'beta'
        if name.find('downsample') != -1:
            name = name.replace("downsample.1", 'down_sample.bn')
            name = name.replace("bn.weight", 'bn.gamma')
            name = name.replace("bias", 'beta')
            name = name.replace("downsample.0.weight", 'down_sample.conv.weight')
        print("----------------", name)
        param_dict['name'] = name
        param_dict['data'] = Tensor(parameter.numpy())
        new_params_list.append(param_dict)
    save_checkpoint(new_params_list, 'midas_pth.ckpt')


if __name__ == '__main__':
    pytorch2mindspore()
