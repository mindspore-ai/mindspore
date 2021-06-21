# Copyright 2021 Huawei Technologies Co., Ltd
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
import mindspore as ms
from mindspore import Tensor
from mindspore.common.api import ms_function
import mindspore.nn as nn
import mindspore.ops as P
import mindspore.context as context
from mindspore.common import Parameter, ParameterTuple


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pynative_ms_function():
    context.set_context(mode=context.PYNATIVE_MODE)

    class MsFunctionCell(nn.Cell):
        def __init__(self):
            super().__init__()
            self.param = Parameter(Tensor(1, ms.float32))

        @ms_function
        def construct(self, x):
            x = self.param * x
            return x

    class NetA(nn.Cell):
        def __init__(self):
            super().__init__()
            self.param = Parameter(Tensor(1, ms.float32))

        def construct(self, x):
            x = self.param * x
            x = x + x
            return x

    class NetB(nn.Cell):
        def __init__(self):
            super().__init__()
            self.ms_function_net = MsFunctionCell()

        def construct(self, x):
            x = self.ms_function_net(x)
            x = x + x
            return x

    net_a = NetA()
    params_a = ParameterTuple(net_a.trainable_params())
    net_b = NetB()
    params_b = ParameterTuple(net_b.trainable_params())
    input_data = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    # The first net run
    grad = P.GradOperation(get_all=True, get_by_list=True, sens_param=False)
    out_a = grad(net_a, params_a)(input_data)
    out_b = grad(net_b, params_b)(input_data)
    assert np.allclose(out_a[0][0].asnumpy(), out_b[0][0].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(out_a[1][0].asnumpy(), out_b[1][0].asnumpy(), 0.0001, 0.0001)
    # The second net run
    out_a = grad(net_a, params_a)(input_data)
    out_b = grad(net_b, params_b)(input_data)
    assert np.allclose(out_a[0][0].asnumpy(), out_b[0][0].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(out_a[1][0].asnumpy(), out_b[1][0].asnumpy(), 0.0001, 0.0001)
