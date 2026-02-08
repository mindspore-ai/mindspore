# Copyright 2023-2025 Huawei Technologies Co., Ltd
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
""" test_graph_fallback_runtime_enumerate.py """
import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor, jit, context
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
def test_enumerate_asnumpy():
    """
    Feature: JIT Fallback
    Description: Test enumerate() in fallback runtime
    Expectation:No exception
    """
    @jit(backend="ms_backend")
    def foo():
        index_sum = 0
        value_sum = 0
        value = Tensor(np.array([11, 22, 33, 44])).asnumpy()
        for i, j in enumerate(value):
            index_sum += i
            value_sum += j
        return index_sum, value_sum

    ret = foo()
    assert ret == (6, 110)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
def test_enumerate_string():
    """
    Feature: JIT Fallback
    Description: Test enumerate() in fallback runtime
    Expectation:No exception
    """
    @jit
    def foo():
        index_sum = 0
        str_res = ""
        str_value = "abcd"
        for i, j in enumerate(str_value):
            index_sum += i
            str_res += j
        return index_sum, str_res

    ret = foo()
    assert ret == (6, "abcd")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_enumerate_check_start():
    """
    Feature: JIT Fallback
    Description: Test enumerate() in fallback runtime
    Expectation:No exception
    """
    class Net(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = ms.nn.ReLU()
            self.start = [1, 2, 3]

        def construct(self, x):
            l = []
            l.append(x)
            l.append(x)
            for index, value in enumerate(l, start=self.start):
                if index == 1:
                    x = self.relu(value)
            return x

    with pytest.raises(TypeError) as error_info:
        net = Net()
        x = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
        out = net(x)
        print("out:", out)
    assert "'For 'enumerate', the 'start' should be a const int number, but got [1, 2, 3].'" in str(error_info.value)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_parser_construct_no_tensor_035():
    """
    Feature: JIT Fallback
    Description: Test enumerate() in fallback runtime
    Expectation:No exception
    """
    class Net(ms.nn.Cell):
        def construct(self, input_1, start):
            return enumerate(input_1, start)

    input_np_1 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_a = Tensor(input_np_1)
    net = Net()
    out_me = net(input_a, 2)
    out_np = enumerate(input_np_1, 2)
    for np_enum, ms_enum in zip(out_np, out_me):
        if isinstance(ms_enum[0], Tensor):
            assert np_enum[0] == ms_enum[0].asnumpy()
        else:
            assert np_enum[0] == ms_enum[0]
        assert np.allclose(np_enum[1], ms_enum[1].asnumpy(), 0, 0)
