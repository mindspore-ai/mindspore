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
# ============================================================================
import pytest
import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor, context
from mindspore.ops.functional import vmap
from mindspore.ops import functional as F
from mindspore.common.api import jit

context.set_context(mode=context.GRAPH_MODE, device_target='GPU')


class CeluTEST(nn.Cell):
    def __init__(self, alpha):
        super(CeluTEST, self).__init__()
        self.celu = P.CeLU(alpha)

    def construct(self, x):
        return self.celu(x)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("data_type", [np.float32, np.float16])
def test_celu_op(data_type):
    """
    Feature: Celu cpu kernel
    Description: test the celu alpha = 1.0.
    Expectation: match to np benchmark.
    """
    error = 1e-3
    celu = CeluTEST(1.)
    x = Tensor(np.array([-2.0, -1.0, 1.0, 2.0]).astype(data_type))
    expect = np.array([-0.8646, -0.6321, 1., 2.]).astype(data_type)
    context.set_context(mode=context.GRAPH_MODE)
    output = celu(x)
    print(output)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=error)
    context.set_context(mode=context.PYNATIVE_MODE)
    output = celu(x)
    print(output)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=error)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("data_type", [np.float32, np.float16])
def test_celu_func(data_type):
    """
    Feature: Celu cpu kernel
    Description: test the celu alpha = 1.0.
    Expectation: match to np benchmark.
    """
    error = 1e-3
    x = Tensor(np.array([-2.0, -1.0, 1.0, 2.0]).astype(data_type))
    expect = np.array([-0.86468184, -0.6321212, 1., 2.]).astype(data_type)
    context.set_context(mode=context.GRAPH_MODE)
    output = F.celu(x, 1.0)
    print(output)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=error)
    context.set_context(mode=context.PYNATIVE_MODE)
    output = F.celu(x, 1.0)
    print(output)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=error)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_celu_vmap():
    """
    Feature: celu gpu kernel.
    Description: test celu vmap feature.
    Expectation: Success.
    """
    error = 1e-3
    def cal_celu(x):
        return P.CeLU(1.0)(x)

    x = Tensor(np.array([[-2.0, -1.0, 1.0, 2.0], [-2.0, -1.0, 1.0, 2.0], [-2.0, -1.0, 1.0, 2.0],
                         [-2.0, -1.0, 1.0, 2.0], [-2.0, -1.0, 1.0, 2.0], [-2.0, -1.0, 1.0, 2.0],
                         [-2.0, -1.0, 1.0, 2.0], [-2.0, -1.0, 1.0, 2.0]]).astype(np.float32))
    expect = np.array([[-0.86468184, -0.6321212, 1., 2.], [-0.86468184, -0.6321212, 1., 2.],
                       [-0.86468184, -0.6321212, 1., 2.], [-0.86468184, -0.6321212, 1., 2.],
                       [-0.86468184, -0.6321212, 1., 2.], [-0.86468184, -0.6321212, 1., 2.],
                       [-0.86468184, -0.6321212, 1., 2.], [-0.86468184, -0.6321212, 1., 2.]]).astype(np.float32)

    vmap_celu = vmap(cal_celu, in_axes=(0), out_axes=0)

    output = vmap_celu(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=error)

    @jit
    def manually_batched(xs):
        output = []
        for i in range(xs.shape[0]):
            output.append(cal_celu(xs[i]))
        return F.stack(output)

    expect_m = manually_batched(x)
    np.testing.assert_allclose(output.asnumpy(), expect_m.asnumpy(), rtol=error)
