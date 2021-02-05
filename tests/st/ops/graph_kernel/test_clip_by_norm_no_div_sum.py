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
import numpy as np
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class ClipByNormNoDivSum(nn.Cell):
    def __init__(self):
        super(ClipByNormNoDivSum, self).__init__()
        self.greater = P.Greater()
        self.select = P.Select()
        self.sqrt = P.Sqrt()
        self.maximum = P.Maximum()

    def construct(self, i0, i1, i2, i3):
        greater_res = self.greater(i0, i1)
        select_res0 = self.select(greater_res, i0, i2)
        sqrt_res = self.sqrt(select_res0)
        select_res1 = self.select(greater_res, sqrt_res, i0)
        res = self.maximum(select_res1, i3)
        return res


def get_output(x0, x1, x2, x3, enable_graph_kernel=False):
    if enable_graph_kernel:
        context.set_context(enable_graph_kernel=True)
    net = ClipByNormNoDivSum()
    output = net(x0, x1, x2, x3)
    return output


def test_clip_by_norm_no_div_sum(shape0, shape1, shape2, shape3, dtype):
    x0 = Tensor(np.random.normal(0, 1, shape0).astype(dtype))
    x1 = Tensor(np.zeros(shape1, dtype))
    x2 = Tensor(np.ones(shape2, dtype))
    x3 = Tensor(np.ones(shape3, dtype))

    expect = get_output(x0, x1, x2, x3, False)
    output = get_output(x0, x1, x2, x3, True)

    expect_np = expect.asnumpy().copy()
    output_np = output.asnumpy().copy()

    assert np.allclose(expect_np, output_np, 0.0001, 0.0001)


def test_clip_by_norm_no_div_sum_ascend():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_clip_by_norm_no_div_sum((1, 1), (1,), (1, 1), (1,), np.float32)
