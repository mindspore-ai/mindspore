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
"""utils script"""
from mindspore.train.serialization import load_param_into_net

def _load_param_into_net(model, params_dict):
    """
    load fp32 model parameters to quantization model.

    Args:
        model: quantization model
        params_dict: f32 param

    Returns:
        None
    """
    model_param = list(model.parameters_and_names())
    filter_keys = ['global_step', 'learning_rate', 'momentum', 'moments']
    filt_param_dict = list(filter(lambda x: x.split('.')[0] not in filter_keys, params_dict))
    if len(model_param) == len(filt_param_dict):
        load_param_into_net(model, params_dict)
        return
    iterable_dict = {
        'weight': iter([item for item in params_dict.items() if item[0].endswith('weight')]),
        'bias': iter([item for item in params_dict.items() if item[0].endswith('bias')]),
        'gamma': iter([item for item in params_dict.items() if item[0].endswith('gamma')]),
        'beta': iter([item for item in params_dict.items() if item[0].endswith('beta')]),
        'moving_mean': iter([item for item in params_dict.items() if item[0].endswith('moving_mean')]),
        'moving_variance': iter(
            [item for item in params_dict.items() if item[0].endswith('moving_variance')]),
        'minq': iter([item for item in params_dict.items() if item[0].endswith('minq')]),
        'maxq': iter([item for item in params_dict.items() if item[0].endswith('maxq')])
    }
    for name, param in model.parameters_and_names():
        key_name = name.split(".")[-1]
        if key_name not in iterable_dict.keys():
            continue
        value_param = next(iterable_dict[key_name], None)
        if value_param is not None:
            param.set_parameter_data(value_param[1].data)
            print(f'init model param {name} with checkpoint param {value_param[0]}')
