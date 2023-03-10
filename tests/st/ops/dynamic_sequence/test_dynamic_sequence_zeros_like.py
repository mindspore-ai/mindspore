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
import numpy as np
from mindspore import context, Tensor, nn
from mindspore.common import mutable
from mindspore.ops.composite.multitype_ops.zeros_like_impl import zeros_like
from sequence_help import TupleFactory, context_prepare

context.set_context(mode=context.GRAPH_MODE)
context_prepare()


class Net(nn.Cell):
    def construct(self, x):
        return zeros_like(x)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_seqence_zeros_like():
    """
    Feature: test sequence zeroslike op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    def func(x):
        return tuple(np.zeros_like(x))
    net_ms = Net()
    input_x = (1, 2, 3, 4, 5)
    fact = TupleFactory(net_ms, func, (input_x,))
    fact.forward_cmp()

    input_x = (mutable(1), 2, 3, 4, mutable(2))
    fact = TupleFactory(net_ms, func, (input_x,))
    fact.forward_cmp()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_seqence_zeros_like_tupleoftensor():
    """
    Feature: test sequence zeroslike op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    net_ms = Net()
    input_x = mutable((Tensor([1, 3, 4]), Tensor([1, 5, 7])), True)
    output = net_ms(input_x)
    excepts = (Tensor([0, 0, 0]), Tensor([0, 0, 0]))
    assert np.array_equal(output[0].asnumpy(), excepts[0].asnumpy())
    assert np.array_equal(output[1].asnumpy(), excepts[1].asnumpy())
