# Copyright 2024 Huawei Technologies Co., Ltd
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
from mindspore.common import Tensor, Parameter, jit
from mindspore.nn import Cell
from mindspore import ops
import mindspore as ms
from tests.mark_utils import arg_mark

ms.context.set_context(mode=ms.GRAPH_MODE)
ms.context.set_context(jit_level='O0')


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.0)
def test_param_compile_cache():
    """
    Feature: Support compile cache when exist parameter in the top graph inputs.
    Description: Support compile cache.
    Expectation: Run success.
    """
    class AddNet(Cell):
        def __init__(self):
            super().__init__()
            self.op = ops.add

        def construct(self, weight):
            out = self.op(1, weight)
            return out

    weight1 = Parameter(Tensor([2]), name='weight1')
    weight2 = Parameter(Tensor([3]), name='weight2')
    net = AddNet()
    out1 = net(weight1)
    assert weight1.asnumpy() == 2
    assert weight2.asnumpy() == 3
    assert out1 == 3
    out2 = net(weight2)
    assert weight1.asnumpy() == 2
    assert weight2.asnumpy() == 3
    assert out2 == 4


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
def test_param_compile_cache_kwargs():
    """
    Feature: Support compile cache when exist parameter in the top graph inputs.
    Description: Support compile cache.
    Expectation: Run success.
    """
    class AddNet(Cell):
        def __init__(self):
            super().__init__()
            self.op = ops.add

        def construct(self, **kwargs):
            out = self.op(kwargs['a'], 3)
            return out

    input_a = Parameter(Tensor([4]), name='input_a')
    input_b = Parameter(Tensor([5]), name='input_b')
    net = AddNet()
    out1 = net(a=input_a)
    assert input_a.asnumpy() == 4
    assert input_b.asnumpy() == 5
    assert out1 == 7
    out2 = net(a=input_b)
    assert input_a.asnumpy() == 4
    assert input_b.asnumpy() == 5
    assert out2 == 8


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
def test_param_compile_cache_jit():
    """
    Feature: Support compile cache when exist parameter in the top graph inputs.
    Description: Support compile cache.
    Expectation: Run success.
    """
    @jit
    def func(param, **kwargs):
        return ops.add(param, kwargs['a'])

    param_1 = Parameter(Tensor([1]), name='param_1')
    param_2 = Parameter(Tensor([2]), name='param_2')
    param_3 = Parameter(Tensor([3]), name='param_3')
    res1 = func(param_1, a=param_1)
    res2 = func(param_2, a=param_1)
    assert res1 == 2
    assert res2 == 3
    assert param_1.asnumpy() == 1
    assert param_2.asnumpy() == 2
    res3 = func(param_1, a=param_2)
    res4 = func(param_1, a=param_3)
    assert res3 == 3
    assert res4 == 4
    assert param_1.asnumpy() == 1
    assert param_2.asnumpy() == 2
    assert param_3.asnumpy() == 3
