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
import pytest
from mindspore.ops.operations import _sequence_ops as seq
from mindspore import context
from mindspore.nn import Cell
from mindspore.common import mutable
from mindspore.ops.composite import GradOperation
from sequence_help import TupleFactory, context_prepare

context.set_context(mode=context.GRAPH_MODE)
context_prepare()


class Net(Cell):
    def __init__(self):
        super().__init__()
        self.func = seq.make_range()

    def construct(self, x, y, z):
        return self.func(x, y, z)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_seqence_make_range():
    """
    Feature: test sequence makerange op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """

    def func(x, y, z):
        return tuple(range(x, y, z))

    net_ms = Net()
    input_x = 1
    input_y = 1000
    input_z = 31
    fact = TupleFactory(net_ms, func, (input_x, input_y, input_z))
    fact.forward_cmp()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_seqence_make_range_grad():
    """
    Feature: test sequence makerange grad
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    net_ms = Net()
    input_x = mutable(10)
    input_y = mutable(100)
    input_z = mutable(3)
    dout = mutable((1, 1), True)
    grad_func = GradOperation(get_all=True, sens_param=True)(net_ms)
    print("grad out1 = ", grad_func(input_x, input_y, input_z, dout))
    input_x = 10
    input_y = 100
    input_z = 30
    dout = (1, 1, 1)
    grad_func = GradOperation(get_all=True, sens_param=True)(net_ms)
    print("grad out1 = ", grad_func(input_x, input_y, input_z, dout))
