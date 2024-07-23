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
import mindspore as ms
from mindspore import Tensor, jit, context, mutable
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_len_asnumpy():
    """
    Feature: JIT Fallback
    Description: Test len() in fallback runtime
    Expectation: No exception.
    """

    @jit
    def foo(x):
        a = [1, 2, 3, x, np.array([1, 2, 3, 4])]
        return len(a), len(x.asnumpy())

    out = foo(Tensor([1, 2, 3, 4]))
    assert out == (5, 4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_len_numpy_string():
    """
    Feature: Graph len syntax.
    Description: Graph syntax len support numpy ndarray.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = np.array([[1, 2, 3], [0, 0, 0]])
        return len(x), len("string")

    out = foo()
    assert out == (2, 6)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_len_mutable():
    """
    Feature: JIT Fallback
    Description: Test len() in fallback runtime
    Expectation: No exception
    """
    @jit
    def foo():
        return len(mutable(2))

    with pytest.raises(AttributeError) as e:
        foo()
    assert "object has no attribute '__len__'" in str(e.value)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_len_cust_class():
    """
    Feature: JIT Fallback
    Description: Test len() in fallback runtime
    Expectation: No exception
    """
    class GetattrClass():
        def __init__(self):
            self.attr1 = [1, 2, 3, 4]

    class GetattrClassNet(ms.nn.Cell):
        def __init__(self):
            super(GetattrClassNet, self).__init__()
            self.cls = GetattrClass()

        def construct(self):
            return len(self.cls.attr1)

    net = GetattrClassNet()
    out = net()
    assert out == 4


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_len_seq_cell():
    """
    Feature: JIT Fallback
    Description: Test len() in fallback runtime
    Expectation: No exception
    """
    class BasicBlock(ms.nn.Cell):
        def __init__(self):
            super(BasicBlock, self).__init__()
            self.model = ms.nn.SequentialCell([ms.nn.ReLU(), ms.nn.ReLU()])

        def construct(self):
            return len(self.model)

    block = BasicBlock()
    out = block()
    assert out == 2
