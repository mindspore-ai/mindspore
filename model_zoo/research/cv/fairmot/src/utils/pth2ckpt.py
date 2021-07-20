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
pth to ckpt
"""
import torch
from mindspore import Tensor
from mindspore.train.serialization import save_checkpoint


def pth2ckpt(path='fairmot_dla34.pth'):
    """pth to ckpt """
    par_dict = torch.load(path, map_location='cpu')
    new_params_list = []
    for name in par_dict:
        param_dict = {}
        parameter = par_dict[name]
        name = 'base.' + name
        name = name.replace('level0', 'level0.0', 1)
        name = name.replace('level1', 'level1.0', 1)
        name = name.replace('hm', 'hm_fc', 1)
        name = name.replace('id.0', 'id_fc.0', 1)
        name = name.replace('id.2', 'id_fc.2', 1)
        name = name.replace('reg', 'reg_fc', 1)
        name = name.replace('wh', 'wh_fc', 1)
        name = name.replace('bn.running_mean', 'bn.moving_mean', 1)
        name = name.replace('bn.running_var', 'bn.moving_variance', 1)
        if name.endswith('.0.weight'):
            name = name[:name.rfind('0.weight')]
            name = name + 'conv.weight'
        if name.endswith('.1.weight'):
            name = name[:name.rfind('1.weight')]
            name = name + 'batchnorm.gamma'
        if name.endswith('.1.bias'):
            name = name[:name.rfind('1.bias')]
            name = name + 'batchnorm.beta'
        if name.endswith('.1.running_mean'):
            name = name[:name.rfind('1.running_mean')]
            name = name + 'batchnorm.moving_mean'
        if name.endswith('.1.running_var'):
            name = name[:name.rfind('1.running_var')]
            name = name + 'batchnorm.moving_variance'
        if name.endswith('conv1.weight'):
            name = name[:name.rfind('conv1.weight')]
            name = name + 'conv_bn_act.conv.weight'
        if name.endswith('bn1.weight'):
            name = name[:name.rfind('bn1.weight')]
            name = name + 'conv_bn_act.batchnorm.gamma'
        if name.endswith('bn1.bias'):
            name = name[:name.rfind('bn1.bias')]
            name = name + 'conv_bn_act.batchnorm.beta'
        if name.endswith('bn1.running_mean'):
            name = name[:name.rfind('bn1.running_mean')]
            name = name + 'conv_bn_act.batchnorm.moving_mean'
        if name.endswith('bn1.running_var'):
            name = name[:name.rfind('bn1.running_var')]
            name = name + 'conv_bn_act.batchnorm.moving_variance'
        if name.endswith('conv2.weight'):
            name = name[:name.rfind('conv2.weight')]
            name = name + 'conv_bn.conv.weight'
        if name.endswith('bn2.weight'):
            name = name[:name.rfind('bn2.weight')]
            name = name + 'conv_bn.batchnorm.gamma'
        if name.endswith('bn2.bias'):
            name = name[:name.rfind('bn2.bias')]
            name = name + 'conv_bn.batchnorm.beta'
        if name.endswith('bn2.running_mean'):
            name = name[:name.rfind('bn2.running_mean')]
            name = name + 'conv_bn.batchnorm.moving_mean'
        if name.endswith('bn2.running_var'):
            name = name[:name.rfind('bn2.running_var')]
            name = name + 'conv_bn.batchnorm.moving_variance'
        if name.endswith('bn.weight'):
            name = name[:name.rfind('bn.weight')]
            name = name + 'bn.gamma'
        if name.endswith('bn.bias'):
            name = name[:name.rfind('bn.bias')]
            name = name + 'bn.beta'
        param_dict['name'] = name
        param_dict['data'] = Tensor(parameter.numpy())
        new_params_list.append(param_dict)
    save_checkpoint(new_params_list, '{}_ms.ckpt'.format(path[:path.rfind('.pth')]))


pth2ckpt('/opt_data/xidian_wks/kasor/Fairmot/src/utils/dla34-ba72cf86.pth')
