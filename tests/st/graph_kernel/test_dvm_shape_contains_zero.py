# Copyright 2025 Huawei Technologies Co., Ltd
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
import numpy as np
import mindspore.context as context
from mindspore import Tensor, nn
import mindspore as ms
import mindspore.ops as ops
from tests.mark_utils import arg_mark

class TestNet(nn.Cell):
    def __init__(self):
        super(TestNet, self).__init__()
        self.add = ops.Add()
        self.mul = ops.Mul()

    def construct(self, p0, p1, p2, p3):
        t0 = self.add(p0, p1)
        t1 = self.mul(p2, p3)
        t2 = self.add(t0, t1)
        return (t1, t2)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
def test_dvm_input_shape_contains_zero():
    """
    Feature: Fused kernel corner case test
    Description: test the behavior of fused kernel which has size of some input equals to zero in static shape situation
    Expectation: the result matches with expect
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(jit_level="O1")
    net = TestNet()
    # shape of input0 contains zero
    np_input_0 = np.random.randn(3, 0, 5).astype(np.float16)
    np_input_1 = np.random.randn(3, 1, 5).astype(np.float16)
    np_input_2 = np.random.randn(3, 1, 5).astype(np.float16)
    np_input_3 = np.random.randn(3, 1, 5).astype(np.float16)
    input_0 = Tensor(np_input_0, dtype=ms.float16)
    input_1 = Tensor(np_input_1, dtype=ms.float16)
    input_2 = Tensor(np_input_2, dtype=ms.float16)
    input_3 = Tensor(np_input_3, dtype=ms.float16)
    res1, res2 = net(input_0, input_1, input_2, input_3)
    np_res1 = np_input_2 * np_input_3
    np_res2 = np_input_0 + np_input_1 + np_res1
    assert np.allclose(res1.asnumpy(), np_res1, 1e-4, 1e-4)
    assert np.allclose(res2.asnumpy(), np_res2, 1e-4, 1e-4)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
def test_dvm_dynamic_shape_contains_zero():
    """
    Feature: Fused kernel corner case test
    Description: test the behavior of fused kernel which has size of some input equals to zero
    in dynamic shape situation
    Expectation: the result matches with expect
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(jit_level="O1")

    net = TestNet()
    dyn_input_0 = Tensor(shape=[3, None, 5], dtype=ms.float16)
    dyn_input_1 = Tensor(shape=[3, None, 5], dtype=ms.float16)
    dyn_input_2 = Tensor(shape=[3, None, 5], dtype=ms.float16)
    dyn_input_3 = Tensor(shape=[3, None, 5], dtype=ms.float16)
    net.set_inputs(dyn_input_0, dyn_input_1, dyn_input_2, dyn_input_3)

    np_input_0 = np.random.randn(3, 0, 5).astype(np.float16)
    np_input_1 = np.random.randn(3, 1, 5).astype(np.float16)
    np_input_2 = np.random.randn(3, 1, 5).astype(np.float16)
    np_input_3 = np.random.randn(3, 0, 5).astype(np.float16)

    input_0 = Tensor(np_input_0, dtype=ms.float16)
    input_1 = Tensor(np_input_1, dtype=ms.float16)
    input_2 = Tensor(np_input_2, dtype=ms.float16)
    input_3 = Tensor(np_input_3, dtype=ms.float16)

    res1, res2 = net(input_0, input_1, input_2, input_3)
    np_res1 = np_input_2 * np_input_3
    np_res2 = np_input_0 + np_input_1 + np_res1
    assert np.allclose(res1.asnumpy(), np_res1, 1e-4, 1e-4)
    assert np.allclose(res2.asnumpy(), np_res2, 1e-4, 1e-4)
