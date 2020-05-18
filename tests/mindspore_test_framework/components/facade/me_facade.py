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

"""Component that do me facade adaptation to core verification config."""

import numpy as np

from ...components.icomponent import IFacadeComponent
from ...utils import keyword
from ...utils.facade_util import get_block_config, fill_block_config


class MeFacadeFC(IFacadeComponent):
    """
    Transform ME style config to mindspore_test_framework style.

    Examples:
        ('MatMul', {
        'block': P.MatMul(),
        'inputs': [[2, 2], [2, 2]],
        'desc_bprop': [[2, 2]],
        'expect': {
            'shape_type': [
                {
                    'type': np.float32,
                    'shape': (2, 2)
                }
            ],
            'compare': [
                matmul_np,
                matmul_tf
            ],
            'compare_gradient': [
                matmul_gradient_tf
            ],
        }
    })
    """

    def __call__(self):
        ret = get_block_config()
        for config in self.verification_set:
            tid = config[0]
            group = 'default'
            m = config[1]
            desc_inputs = m.get(keyword.desc_inputs, [])
            desc_bprop = m.get(keyword.desc_bprop, [])
            expect = m.get(keyword.desc_expect, [])
            block = m.get(keyword.block, None)
            desc_const = m.get(keyword.desc_const, [])
            const_first = m.get(keyword.const_first, False)
            add_fake_input = m.get(keyword.add_fake_input, False)
            fake_input_type = m.get(keyword.fake_input_type, np.float32)
            fill_block_config(ret, block, tid, group, desc_inputs, desc_bprop, expect,
                              desc_const, const_first, add_fake_input, fake_input_type)
        return ret
