# Copyright 2022 Huawei Technologies Co., Ltd
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
# ===================================================
import pytest
import mindspore.context as context
import mindspore.ops.operations.math_ops as P
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype


class MulNoNanNet(nn.Cell):
    def __init__(self):
        super(MulNoNanNet, self).__init__()
        self.mulnonan = P.MulNoNan()

    def construct(self, x, y):
        return self.mulnonan(x, y)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_trainingg
@pytest.mark.env_onecard
def test_mulnonan_graph():
    '''
        Description: 1d fp32
        Expectation: success
    '''
    input_x = Tensor([-1.2447039, 0.49464428, -1.1997741, -0.29174238, 0.26505676, 0.06303899], dtype=mstype.float32)
    input_y = Tensor([-1.6108084, 2.9904065, 1.9903952, 0.26857498, -1.0689504, -1.4920218], dtype=mstype.float32)
    res_expect = Tensor([2.0049794, 1.4791875, -2.3880246, -0.07835471, -0.28333253, -0.09405555], dtype=mstype.float32)
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    mul = MulNoNanNet()
    res = mul(input_x, input_y)
    assert(res.asnumpy() == res_expect).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_trainingg
@pytest.mark.env_onecard
def test_mulnonan_pynative():
    '''
        Description: 1d fp32
        Expectation: success
    '''
    input_x = Tensor([-1.2447039, 0.49464428, -1.1997741, -0.29174238, 0.26505676, 0.06303899], dtype=mstype.float32)
    input_y = Tensor([-1.6108084, 2.9904065, 1.9903952, 0.26857498, -1.0689504, -1.4920218], dtype=mstype.float32)
    res_expect = Tensor([2.0049794, 1.4791875, -2.3880246, -0.07835471, -0.28333253, -0.09405555], dtype=mstype.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    mul = MulNoNanNet()
    res = mul(input_x, input_y)
    assert(res.asnumpy() == res_expect).all()
