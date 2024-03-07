# Copyright 2024 Huawei Technologies Co., Ltd
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
import os
import pytest
import mindspore.context as context
from mindspore import Tensor, nn
import mindspore as ms
import mindspore.ops.operations as P

ascend_grad_overflow = P.IsFinite()


def tensor_ascend_grad_overflow(grad):
    status = ascend_grad_overflow(grad)
    base = Tensor(1.0, dtype=ms.float32)
    output = base - status.all()
    output = P.Reshape()(output, ((1,)))
    return output

class ComplexNet(nn.Cell):
    def __init__(self):
        super(ComplexNet, self).__init__()
        self.greater = P.Greater()
        self.select = P.Select()
        self.gelu = P.GeLU()
        self.reduce_sum = P.ReduceSum(keep_dims=True)
        self.reduce_mean = P.ReduceMean()
        self.addn = P.AddN()

    def construct(self, x, y):
        a = x + y + 4
        b = x - y - 5
        c = self.gelu(x)
        d = self.reduce_sum(c, (0,))
        e = self.greater(a, b)
        f = self.select(e, a*a, b+4)
        a_overflow = tensor_ascend_grad_overflow(a)
        b_overflow = tensor_ascend_grad_overflow(b)
        d_overflow = tensor_ascend_grad_overflow(d)
        g = self.addn((a_overflow, b_overflow, d_overflow))
        return f, d, g


def get_output(net, i0, i1, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net_obj = net()
    output = net_obj(i0, i1)
    return output


def fuse(shape1, shape2, dtype):
    i0 = Tensor(np.random.uniform(1, 2, shape1).astype(dtype))
    i1 = Tensor(np.random.uniform(1, 2, shape2).astype(dtype))
    expect = get_output(ComplexNet, i0, i1, False)
    output = get_output(ComplexNet, i0, i1, True)
    if dtype == np.float32:
        eps = 1e-5
    elif dtype == np.float16:
        eps = 1e-3
    else:
        eps = 0
    assert np.allclose(expect[0].asnumpy(), output[0] .asnumpy(), eps, eps)
    assert np.allclose(expect[1].asnumpy(), output[1] .asnumpy(), eps, eps)
    assert np.allclose(expect[2].asnumpy(), output[2] .asnumpy(), 0, 0)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("shape1, shape2", [((32, 1024), (32, 1024)), ((44, 1, 47, 1), (1, 34, 1, 91))])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_easy_fuse_dvm(shape1, shape2, dtype):
    """
    Feature: easy test case for graph_kernel in Ascend.
    Description: ascend test case, use graph_kernel execute ops.
    Expectation: the result match with close graph_kernel result
    """
    os.environ["GRAPH_OP_RUN"] = "1"
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    fuse(shape1, shape2, dtype)
    del os.environ["GRAPH_OP_RUN"]
