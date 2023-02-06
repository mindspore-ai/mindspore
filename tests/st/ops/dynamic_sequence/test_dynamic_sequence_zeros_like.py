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
from mindspore.ops.operations import _sequence_ops as seq
from mindspore import context
from mindspore.nn import Cell
from tuple_help import TupleFactory

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_seqence_zeros_like():
    """
    Feature: test sequence zeroslike op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.func = seq.SequenceZerosLike()

        def construct(self, x):
            return self.func(x)

    def func(x):
        return tuple(np.zeros_like(x))
    net_ms = Net()
    input_x = (1, 2, 3, 4, 5)
    fact = TupleFactory(net_ms, func, (input_x,))
    fact.forward_cmp()
