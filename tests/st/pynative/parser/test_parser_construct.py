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
""" test_parser_construct """
import pytest
import numpy as np
from mindspore import context
from mindspore.nn import Cell
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.ops.composite import GradOperation
from mindspore.common.parameter import Parameter

def setup_module():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_parser_construct():
    class ParentNet(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()

        def construct(self, x):
            return self.relu(x)

    class UncleNet(Cell):
        def __init__(self):
            super(UncleNet, self).__init__()
            self.sigmoid = P.Sigmoid()

        def construct(self, x):
            return self.sigmoid(x)

    class Net(UncleNet, ParentNet):
        def __init__(self):
            super().__init__()
            super(UncleNet, self).__init__()

        def construct(self, x):
            return super(UncleNet, self).construct(x)

    input_np_x = np.ones([2, 3, 4, 5]).astype(np.float32)
    out_np = np.ones([2, 3, 4, 5]).astype(np.float32)

    input_me = Tensor(input_np_x)
    output_grad_me = Tensor(out_np)
    net = Net()
    out_me = net(input_me)

    net1 = Net()
    grad = GradOperation(sens_param=True)
    grad_op = grad(net1)
    grad_me = grad_op(input_me, output_grad_me)

    assert np.allclose(input_np_x, out_me.asnumpy(), 0.001, 0.001)
    assert np.allclose(input_np_x, grad_me.asnumpy(), 0.001, 0.001)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sit_parser_input_parameter():
    def tensor_add(x, y):
        add = P.Add()
        z = add(x, y)
        return  z
    x = Tensor(np.ones([2, 2]).astype(np.float32))
    x = Parameter(x, name="x")
    y = Tensor(np.ones([2, 2]).astype(np.float32))
    y = Parameter(y, name="y")
    grad = GradOperation(get_all=True, get_by_list=False, sens_param=False)

    with pytest.raises(TypeError):
        grad(tensor_add)(x, y)
