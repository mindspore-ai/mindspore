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
import numpy as np
from tests.mark_utils import arg_mark
import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor, mint
from mindspore import context


class args_Net(nn.Cell):
    def construct(self, x, other):
        return mint.remainder(x, other)


class kwargs_Net(nn.Cell):
    def construct(self, x, other):
        return mint.remainder(input=x, other=other)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.01)
def test_remainder_scalar_tensor_args_pynative():
    """
    Feature: mint.remainder()
    Description: Verify the result of tensor.remainder
    Expectation: success
    """
    context.set_context(mode=ms.PYNATIVE_MODE)
    args_net = args_Net()
    a = 2
    x = Tensor(np.array([-4.0, 5.0, 6.0]), mstype.float32)
    expected = np.array([-2.0, 2.0, 2.0], dtype=np.float32)
    output = args_net(a, x)
    assert np.allclose(output.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.01)
def test_remainder_scalar_tensor_kwargs_pynative():
    """
    Feature: mint.remainder()
    Description: Verify the result of tensor.remainder
    Expectation: success
    """
    context.set_context(mode=ms.PYNATIVE_MODE)
    kwargs_net = kwargs_Net()
    a = 2
    x = Tensor(np.array([-4.0, 5.0, 6.0]), mstype.float32)
    expected = np.array([-2.0, 2.0, 2.0], dtype=np.float32)
    output = kwargs_net(a, x)
    assert np.allclose(output.asnumpy(), expected)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
def test_remainder_scalar_tensor_args_graph():
    """
    Feature: mint.remainder()
    Description: Verify the result of tensor.remainder
    Expectation: success
    """
    context.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})
    args_net = args_Net()
    a = 2
    x = Tensor(np.array([-4.0, 5.0, 6.0]), mstype.float32)
    expected = np.array([-2.0, 2.0, 2.0], dtype=np.float32)
    output = args_net(a, x)
    assert np.allclose(output.asnumpy(), expected)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
def test_remainder_scalar_tensor_kwargs_graph():
    """
    Feature: mint.remainder()
    Description: Verify the result of tensor.remainder
    Expectation: success
    """
    context.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})
    kwargs_net = kwargs_Net()
    a = 2
    x = Tensor(np.array([-4.0, 5.0, 6.0]), mstype.float32)
    expected = np.array([-2.0, 2.0, 2.0], dtype=np.float32)
    output = kwargs_net(a, x)
    assert np.allclose(output.asnumpy(), expected)
