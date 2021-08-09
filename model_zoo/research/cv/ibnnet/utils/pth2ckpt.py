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
"""
python pth2ckpt.py
"""
import torch
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor

param = {
    'bn1.bias': 'bn1.beta',
    'bn1.weight': 'bn1.gamma',
    'IN.weight': 'IN.gamma',
    'IN.bias': 'IN.beta',
    'BN.bias': 'BN.beta',
    'in.weight': 'in.gamma',
    'bn.weight': 'bn.gamma',
    'bn.bias': 'bn.beta',
    'bn2.weight': 'bn2.gamma',
    'bn2.bias': 'bn2.beta',
    'bn3.bias': 'bn3.beta',
    'bn3.weight': 'bn3.gamma',
    'BN.running_mean': 'BN.moving_mean',
    'BN.running_var': 'BN.moving_variance',
    'bn.running_mean': 'bn.moving_mean',
    'bn.running_var': 'bn.moving_variance',
    'bn1.running_mean': 'bn1.moving_mean',
    'bn1.running_var': 'bn1.moving_variance',
    'bn2.running_mean': 'bn2.moving_mean',
    'bn2.running_var': 'bn2.moving_variance',
    'bn3.running_mean': 'bn3.moving_mean',
    'bn3.running_var': 'bn3.moving_variance',
    'downsample.1.running_mean': 'downsample.1.moving_mean',
    'downsample.1.running_var': 'downsample.1.moving_variance',
    'downsample.0.weight': 'downsample.1.weight',
    'downsample.1.bias': 'downsample.1.beta',
    'downsample.1.weight': 'downsample.1.gamma'
}


def pytorch2mindspore():
    """

    Returns:
        object:
    """
    par_dict = torch.load('resnet50_ibn_a-d9d0bb7b.pth', map_location='cpu')
    new_params_list = []
    for name in par_dict:
        param_dict = {}
        parameter = par_dict[name]
        print(name)
        for fix in param:
            if name.endswith(fix):
                name = name[:name.rfind(fix)]
                name = name + param[fix]

        print('========================ibn_name', name)

        param_dict['name'] = name
        param_dict['data'] = Tensor(parameter.numpy())
        new_params_list.append(param_dict)

    save_checkpoint(new_params_list, 'resnet50_ibn_a.ckpt')


pytorch2mindspore()
