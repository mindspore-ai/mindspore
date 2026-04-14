# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Test skip_offload region functionality in PyNative mode."""
import mindspore as ms
from mindspore import nn, ops, Tensor, context
from mindspore.utils import skip_offload
from mindspore._c_expression import OffloadManager


class Net(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x, w):
        y1 = ops.matmul(x, w)

        @skip_offload
        def inner_block(y1):
            y2 = ops.relu(y1)
            return ops.mul(y2, 2.0)

        z = inner_block(y1)
        out = ops.add(z, 1.0)
        return out


def test_offload_skip_region():
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = Net()
    net.auto_offload()
    x = Tensor([[1.0, 1.0]], ms.float32)
    w = Tensor([[2.0], [3.0]], ms.float32)

    grad_fn = ms.value_and_grad(net, grad_position=(0, 1))
    out, grad = grad_fn(x, w)

    manager.analyse()

    out, grad = grad_fn(x, w)

    print("out:", out)
    print("grad:", grad)

    assert out.asnumpy()[0][0] == 11.0


if __name__ == "__main__":
    test_offload_skip_region()
