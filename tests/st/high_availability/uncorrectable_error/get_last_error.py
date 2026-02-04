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
Test `aclrtGetLastError` will be called when enable UCE
"""
import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor


class Norm(nn.Cell):
    def __init__(self, axis=(), keep_dims=False):
        """Initialize Norm."""
        super().__init__()
        self.axis = axis
        self.keep_dims = keep_dims
        self.reduce_sum = ops.ReduceSum(True)
        self.sqrt = ops.Sqrt()
        self.squeeze = ops.Squeeze(self.axis)

    def construct(self, x):
        out = self.sqrt(self.reduce_sum(ops.square(x), self.axis))
        if not self.keep_dims:
            out = self.squeeze(out)
        return out

ms.set_context(mode=ms.GRAPH_MODE)

input_x = Tensor(np.ones([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32))
net = Norm()
output = net(input_x)
print(output.asnumpy())
