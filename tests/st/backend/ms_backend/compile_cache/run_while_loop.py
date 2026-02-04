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
"""
test compile cache when using WhileLoopEvaluator to handle ops.WhileLoop operation.
"""
import mindspore as ms
from mindspore import Tensor, jit, context, ops, nn


@ms.jit
def cond_func(init_cond_value):
    return init_cond_value[1] > 1


@jit
def while_function(init_value):
    input_tensor, init, add = init_value
    while_func_out = add(input_tensor, init)
    init = init - 1
    return [while_func_out, init, add]

@jit
class WhileLoopNet(nn.Cell):
    """WhileLoopNet
    Args:
        None

    Inputs:
        input (Tensor): Input tensor of any numeric type.

    Returns:
        Tensor, output tensor after while-loop processing.

    Examples:
        >>> net = WhileLoopNet()
        >>> x = Tensor(1, mstype.int32)
        >>> output = net(x)
    """
    def __init__(self):
        super().__init__()
        self.add = ops.Add()
        self.whileop = ops.WhileLoop()

    def construct(self, while_loop_input):
        whileop_res = self.whileop(cond_func, while_function, [while_loop_input, 3, self.add])
        return whileop_res[0]


if __name__ == "__main__":
    context.set_context(jit_config={"jit_level": "O1"})
    net = WhileLoopNet()
    res = net(Tensor([2]))
    assert res == 7
    print("RUNTIME_COMPILE", res, "RUNTIME_CACHE")
    print("RUNTIME_COMPILE", res.asnumpy().shape, "RUNTIME_CACHE")
