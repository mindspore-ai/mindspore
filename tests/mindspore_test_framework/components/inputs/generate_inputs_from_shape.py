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

"""Component that generate inputs for specified shape type."""

import numpy as np

from mindspore.common.tensor import Tensor
from ...components.icomponent import IDataComponent
from ...utils import keyword
from ...utils.config_util import get_input_config
from ...utils.other_util import shape2tensor


class GenerateFromShapeDC(IDataComponent):
    """
    Generate inputs from shape, desc_inputs must be configured, desc_bprop is optional.

    Examples:
        'desc_inputs': [
            [1, 16, 128, 64], # inputs could be shape/tensor/np.ndarray
        ]
        'desc_inputs': [
            ([1, 16, 128, 64], np.float32, 6), # (inputs, dtype, scale)
        ]
        'desc_bprop': [
            [1, 16, 128, 64], # inputs could be shape/tensor/np.ndarray
        ]
        'desc_bprop': [
            ([1, 16, 128, 64], np.float32, 6), # (inputs, dtype, scale)
        ]
    """

    def __call__(self):
        result = []
        for config in self.verification_set[keyword.inputs]:
            desc_inputs = config[keyword.desc_inputs]
            add_fake_input = config.get(keyword.add_fake_input, False)
            fake_input_type = config.get(keyword.fake_input_type, np.float32)

            inputs = []
            if not desc_inputs and add_fake_input:
                inputs = [Tensor(np.array([1.0]).astype(fake_input_type))]
            else:
                for d in desc_inputs:
                    s, dtype, scale = get_input_config(d)
                    inputs.append(shape2tensor(s, dtype, scale))
            config[keyword.desc_inputs] = inputs

            desc_bprop = config.get(keyword.desc_bprop, [])
            bprops = []
            if not desc_bprop and add_fake_input:
                bprops = [Tensor(np.array([1.0]))]
            else:
                for d in desc_bprop:
                    s, dtype, scale = get_input_config(d)
                    bprops.append(shape2tensor(s, dtype, scale))
            config[keyword.desc_bprop] = bprops

            result.append(config)
        return result
