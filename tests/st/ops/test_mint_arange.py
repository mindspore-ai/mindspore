# Copyright 2022-2024 Huawei Technologies Co., Ltd
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

import numpy as np
import pytest
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark

import mindspore as ms
from mindspore import mint, int32, int64, float32


@test_utils.run_with_cell
def arange_forward(start=0, end=None, step=1, *, dtype=None):
    return mint.arange(start, end, step, dtype=dtype)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_arange_forward(mode):
    """
    Feature: mint.arange
    Description: Verify the result of arange forward
    Expectation: success
    """
    ms.set_context(mode=mode)
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_config={'jit_level': 'O0'})
    cases = [
        {
            'args': (1, 6),
            'kwargs': {},
            'expected': np.array(range(1, 6)),
            'dtype': int64
        },
        {
            'args': (0, 5.5, 1.2),
            'kwargs': {},
            'expected': np.array([0, 1.2, 2.4, 3.6, 4.8]),
            'dtype': float32
        },
        {
            'args': (10.0,),
            'kwargs': {'dtype': int32},
            'expected': np.array(range(10)),
            'dtype': int32
        }
    ]

    for case in cases:
        res = arange_forward(*case['args'], **case['kwargs'])
        assert np.allclose(res.asnumpy(), case['expected'])
        assert res.dtype == case['dtype']


@arg_mark(plat_marks=['platform_ascend910b', 'platform_ascend'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
def test_forward_dynamic_shape():
    """
    Feature: mint.arange
    Description: Verify the result of arange forward with dynamic shape
    Expectation: success
    """
    inputs1 = [[1, 10, 2], [0, 6, 1]]
    inputs2 = [[5, 0.1, -1.2], [0, 5.5, 1.2]]
    TEST_OP(arange_forward, inputs1, 'arange', disable_mode=['GRAPH_MODE'], disable_grad=True, disable_yaml_check=True)
    TEST_OP(arange_forward, inputs2, 'arange', disable_mode=['GRAPH_MODE'], disable_grad=True, disable_yaml_check=True)
