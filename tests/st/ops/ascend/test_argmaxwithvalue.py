# Copyright 2023 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import jit
from mindspore.common import dtype as mstype


class ArgMaxWithValue(nn.Cell):
    def __init__(self, axis, keep_dims):
        super().__init__()
        self.axis = axis
        self.keep_dims = keep_dims

    @jit
    def construct(self, x):
        out = x.argmax_with_value(self.axis, self.keep_dims)
        return out


def get_output(x, axis, keep_dims):
    net = ArgMaxWithValue(axis, keep_dims)
    out = net(x)
    if out[0].dtype == mstype.bfloat16 and len(out) > 1:
        return [out[0].float().asnumpy(), out[1].asnumpy()]
    return [t.asnumpy() for t in out]


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_argmax_with_value(mode):
    """
    Feature: Test primitive argmaxwithvalue operator.
    Description: Operator argmaxwithvalue's input Tensors with bfloat16 type.
    Expectation: Assert result compare with expect value.
    """
    context.set_context(mode=mode, device_target="Ascend")
    input_ms = Tensor([[1.1, 2.0, 3.0], [3.0, 4.0, 2.0], [5.0, 6.0, 7.0]], mstype.bfloat16)
    axis = 1
    keep_dims = True
    rst_out = get_output(input_ms, axis, keep_dims)
    rst_expect = [np.array([[3.0], [4.0], [7.0]]).astype(np.float32), np.array([[2], [1], [2]]).astype(np.int32)]
    assert np.allclose(rst_expect[0], rst_out[0], 0.004, 0.004)
    assert np.allclose(rst_expect[1], rst_out[1], 0.004, 0.004)
