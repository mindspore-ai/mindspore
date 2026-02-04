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
test compile cache with kernel packet reducesum.
"""
import numpy as np
import mindspore as ms
from mindspore import ops, nn, Tensor, context


def helper(net, inputs_dyn, inputs, expect):
    """
    Test compile cache with kernel packet reducesum in O1 graph mode.

    Steps:
    1. Run in jit mode with jit level O1
    2. Print results for verification
    """
    net.set_inputs(*inputs_dyn)
    output = net(*inputs)
    print("RUNTIME_COMPILE", output[0], "RUNTIME_CACHE")
    print("RUNTIME_COMPILE", output[0].asnumpy().shape, "RUNTIME_CACHE")


@ms.jit
class ReduceSumNet(nn.Cell):
    """
    ReduceSumNet
    Args:
        None
    Inputs:
        x (Tensor): Input tensor of integer type.

    Returns:
        Tensor, output tensor

    Examples:
        >>> net = ReduceSumNet()
        >>> x = Tensor(np.array(1), mstype.int32)
        >>> output = net(x)
    """
    def __init__(self):
        super().__init__()
        self.add = ops.Add()
        self.shape = ops.Shape()
        self.reducesum = ops.ReduceSum(True, True)

    def construct(self, x):
        shape = self.shape(x)
        b = shape[1]
        y = self.reducesum(x, b)
        return y

def calc(x):
    return np.sum(x, x.shape[1], keepdims=True)


if __name__ == "__main__":
    context.set_context(jit_level="O1")
    x_dyn = Tensor(shape=[None, None], dtype=ms.float32)
    input_x = np.array([[2], [1]], dtype=np.float32)
    helper(ReduceSumNet(), (x_dyn,), (Tensor(input_x),), calc(input_x))
