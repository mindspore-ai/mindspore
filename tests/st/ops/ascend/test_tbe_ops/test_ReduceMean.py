# Copyright 2020-2023 Huawei Technologies Co., Ltd
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

import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import jit
from mindspore.ops import operations as P
from mindspore.ops.composite import GradOperation

context.set_context(device_target="Ascend")


class Net(nn.Cell):
    def __init__(self, keep_dims, axis):
        super(Net, self).__init__()
        self.reduce_mean = P.ReduceMean(keep_dims=keep_dims)
        self.axis = axis

    @jit
    def construct(self, inputs):
        return self.reduce_mean(inputs, self.axis)


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    @jit
    def construct(self, input_, grads):
        return self.grad(self.network)(input_, grads)


x1 = np.random.randn(64).astype(np.float32)


def test_net():
    """
    Feature: test reducemean function forward.
    Description: test reducemean op.
    Expectation: expect correct result.
    """
    keepdims = False
    axis = -1
    Reduce_mean = Net(keepdims, axis)
    output = Reduce_mean(Tensor(x1))
    print(x1)
    print(output.asnumpy())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_net_bf16():
    """
    Feature: test reducemean function bf16 forward.
    Description: test reducemean op bf16 forward.
    Expectation: expect correct result.
    """
    x = Tensor(np.array([[[2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2]],
                         [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
                         [[6, 6, 6, 6, 6, 6], [8, 8, 8, 8, 8, 8], [10, 10, 10, 10, 10, 10]]]), mindspore.bfloat16)
    keepdims = True
    axis = 0
    Reduce_mean = Net(keepdims, axis)
    output = Reduce_mean(x)
    except_out = np.array([[[4., 4., 4., 4., 4., 4.],
                            [5., 5., 5., 5., 5., 5.],
                            [6., 6., 6., 6., 6., 6.]]]).astype(np.float32)
    assert np.allclose(output.float().asnumpy(), except_out, 0.004, 0.004)
