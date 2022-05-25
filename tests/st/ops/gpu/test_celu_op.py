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


context.set_context(mode=context.GRAPH_MODE, device_target='GPU')


class CeluTEST(nn.Cell):
    def __init__(self, alpha):
        super(CeluTEST, self).__init__()
        self.celu = P.CeLU(alpha)

    def construct(self, x):
        return self.celu(x)


@pytest.mark.level0
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
