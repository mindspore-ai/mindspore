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
""" test graph fallback control flow."""
import pytest
import itertools
import numpy as np
from mindspore import Tensor, jit, context
from mindspore import dtype as mstype
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
def test_single_for_1():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_for():
        x = Tensor(7).astype("int32")
        y = Tensor(0).astype("int32")
        for _ in range(3):
            y += x
        return y
    res = control_flow_for()
    assert res == 21


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
def test_single_for_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_for():
        x = Tensor(7).astype("int32")
        y = Tensor(0).astype("int32")
        for _ in range(Tensor(3).astype("int32")):
            y += x
        return y

    with pytest.raises(TypeError, match="the 0th input should be a int scalar"):
        control_flow_for()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.01)
def test_single_for_zip():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_for():
        tuple_x = (Tensor(1).astype("int32"), Tensor(3).astype("int32"), Tensor(5).astype("int32"))
        sum_x = Tensor(0).astype("int32")
        for x in zip(tuple_x):
            sum_x += x
        return sum_x

    res = control_flow_for()
    assert res == 9


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.01)
def test_single_for_builtin_function_int():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_for():
        x = np.array(1.1)
        for _ in range(3):
            x = x + int(x)
        return Tensor(x, mstype.float32)
    res = control_flow_for()
    assert res == 8.1


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_single_for_iter_object():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for(x, y, z):
        a = 0
        m = (x, y, z)
        n = (x + 3, y + 3, z + 3)
        for i, j in itertools.product(m, n):
            a = a + i * j
        return a

    ret = control_flow_for(Tensor([1]), Tensor([2]), Tensor([3]))
    assert ret == 90
