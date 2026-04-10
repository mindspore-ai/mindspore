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
from tests.mark_utils import arg_mark
from mindspore import context
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

context.set_context(mode=context.PYNATIVE_MODE)


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.i = 1

    def construct(self, x):
        return x.item()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu'],
          level_mark='level0',
          card_mark='onecard')
def test_tensor_item():
    """
    Feature: tensor.item
    Description: Verify the result of item
    Expectation: success
    """
    net = Net()
    # ============== dtype of tensor is int8 ===========
    x = ms.Tensor(1, ms.int8)
    output = net(x)
    assert output == 1

    # ============== dtype of tensor is uint8 ===========
    x = ms.Tensor(1, ms.uint8)
    output = net(x)
    assert output == 1
    y = ops.abs(x)
    output = net(y)
    assert output == 1

    # ============== dtype of tensor is int16 ===========
    x = ms.Tensor(1, ms.int16)
    output = net(x)
    assert output == 1

    # ============== dtype of tensor is uint16 ===========
    x = ms.Tensor(1, ms.uint16)
    output = net(x)
    assert output == 1

    # ============== dtype of tensor is int32 ===========
    x = ms.Tensor(1, ms.int32)
    output = net(x)
    assert output == 1
    y = ops.abs(x)
    output = net(y)
    assert output == 1

    # ============== dtype of tensor is uint32 ===========
    x = ms.Tensor(1, ms.uint32)
    output = net(x)
    assert output == 1

    # ============== dtype of tensor is int64 ===========
    x = ms.Tensor(1, ms.int64)
    output = net(x)
    assert output == 1

    # ============== dtype of tensor is uint64 ===========
    x = ms.Tensor(1, ms.uint64)
    output = net(x)
    assert output == 1
    y = ops.sqrt(x)
    output = net(y)
    assert output == 1

    # ============== dtype of tensor is float32 ===========
    x = ms.Tensor(1.0, ms.float32)
    output = net(x)
    assert output == 1.0
    y = ops.abs(x)
    output = net(y)
    assert output == 1

    # ============== dtype of tensor is float64 ===========
    x = ms.Tensor(1.0, ms.float64)
    output = net(x)
    assert output == 1.0
    y = ops.abs(x)
    output = net(y)
    assert output == 1

    # ============== dtype of tensor is bfloat16 ===========
    x = ms.Tensor(1.0, ms.bfloat16)
    output = net(x)
    assert output == 1.0

    # ============== dtype of tensor is float16 ===========
    x = ms.Tensor(1.0, ms.float16)
    output = net(x)
    assert output == 1.0
    y = ops.abs(x)
    output = net(y)
    assert output == 1.0

    # ============== dtype of tensor is bool ===========
    x = ms.Tensor(True)
    output = net(x)
    assert output

    # ============== dtype of tensor is complex64 ===========
    x = ms.Tensor(1 + 1j, ms.complex64)
    output = net(x)
    assert output == complex(1 + 1j)
    y = ops.abs(x)
    output = net(y)
    assert output == y.asnumpy().item()

    # ============== dtype of tensor is complex128 ===========
    x = ms.Tensor(1 - 1j, ms.complex128)
    output = net(x)
    assert output == complex(1 - 1j)
