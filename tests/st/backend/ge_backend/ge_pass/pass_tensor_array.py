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
""" test ge frontend pass and op `TensorArray`"""
import numpy as np

import mindspore.context as context
from mindspore import nn
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import numpy as mnp
from mindspore.ops import constexpr
from mindspore.ops.operations import _tensor_array as P

context.set_context(mode=context.GRAPH_MODE)


@constexpr
def get_while_index():
    return Tensor(0, mstype.int64)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.tensor_array = P.TensorArray(dtype=mstype.float32, element_shape=(8,), name="test_ta",
                                          dynamic_size=False, size=25)
        self.tensor_array_write = P.TensorArrayWrite()
        self.tensor_array_gather = P.TensorArrayGather(
            dtype=mstype.float32, element_shape=(8,))
        self.tensor_0 = Tensor(0, mstype.int32)
        self.tensor_1 = Tensor(1, mstype.int32)

    def construct(self, x, cond):
        index_0 = get_while_index()
        tensor_array_handle = self.tensor_array()
        indices_range = mnp.arange(0, 3, 1, mstype.int32)

        while index_0 < cond:
            self.tensor_array_write(tensor_array_handle, index_0, x)
            index_0 += 1

        stack_res = self.tensor_array_gather(
            tensor_array_handle, indices_range)
        return stack_res


def ge_tensor_array(data):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    net = Net()
    x = Tensor(data)
    cond = Tensor(3, mstype.int32)
    out = net(x, cond)
    return out


def run_ge_tensor_array():
    """
    Feature: Test TensorArray in GE backend.
    Description: Test TensorArray in GE backend.
    Expectation: success.
    """
    data = np.arange(1, 9).astype(np.float32)
    out = ge_tensor_array(data)
    assert np.allclose(out.asnumpy(), np.array([[1., 2., 3., 4., 5., 6., 7., 8.],
                                                [1., 2., 3., 4., 5., 6., 7., 8.],
                                                [1., 2., 3., 4., 5., 6., 7., 8.]]).astype(np.float32))


if __name__ == "__main__":
    run_ge_tensor_array()
