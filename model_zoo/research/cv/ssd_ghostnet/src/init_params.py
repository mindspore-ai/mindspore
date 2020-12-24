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
"""Parameters utils"""

import numpy as np
from mindspore.common.initializer import initializer, TruncatedNormal


def init_net_param(network, initialize_mode='TruncatedNormal'):
    """Init the parameters in net."""
    params = network.trainable_params()
    for p in params:
        if 'beta' not in p.name and 'gamma' not in p.name and 'bias' not in p.name:
            np.random.seed(seed=1)
            if initialize_mode == 'TruncatedNormal':
                p.set_data(initializer(
                    TruncatedNormal(), p.data.shape, p.data.dtype))
            else:
                p.set_data(
                    initialize_mode, p.data.shape)


def load_backbone_params(network, param_dict):
    """Init the parameters from pre-train model, default is mobilenetv2."""
    for _, param in network.parameters_and_names():
        param_name = param.name.replace('network.backbone.', '')
        name_split = param_name.split('.')
        if 'features_1' in param_name:
            param_name = param_name.replace('features_1', 'features')
        if 'features_2' in param_name:
            param_name = '.'.join(
                ['features', str(int(name_split[1]) + 14)] + name_split[2:])
        if param_name in param_dict:
            param.set_parameter_data(param_dict[param_name].data)


def filter_checkpoint_parameter(param_dict):
    """remove useless parameters"""
    for key in list(param_dict.keys()):
        if 'multi_loc_layers' in key or 'multi_cls_layers' in key:
            del param_dict[key]
