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
''' test context option '''
from mindspore import Tensor, jit
import mindspore as ms
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_initial_tensor_body_ref():
    """
    Feature: While specialize.
    Description: Test constant tensor arg when first entry of while and set to RefTensor in body.
    Expectation: No exception in infer process.
    """

    class Net(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.weight = ms.Parameter(Tensor([1]))

        def construct(self, a, b):
            y_param = Tensor([1])
            while a < b:
                y_param = self.weight
                a += 1
            out = y_param + b
            return out

    test_net = Net()
    ms.context.set_context(precompile_only=True, mode=ms.context.GRAPH_MODE)
    input_a = Tensor([2])
    input_b = Tensor([6])
    test_net(input_a, input_b)

    @jit(mode="PIJit")
    def func(a, b):
        return ms.ops.grad(test_net)(a, b)

    input_a = Tensor([2])
    input_b = Tensor([6])
    res = jit(mode="PIJit", fn=func)(input_a, input_b)
    except_res = jit(mode="PSJit", fn=func)(input_a, input_b)

    ms.context.set_context(precompile_only=False, mode=ms.context.PYNATIVE_MODE)
    assert res == except_res
