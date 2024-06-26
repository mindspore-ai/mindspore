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

import numpy as np
import pytest
from tests.mark_utils import arg_mark

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import dtype as mstype


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_contiguous_pynative():
    """
    Feature: countiguous
    Description: Verify the result of x
    Expectation: success
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    x = ms.Tensor([[1, 2, 3], [4, 5, 6]], dtype=ms.float32)
    y = ops.transpose(x, (1, 0))
    z = y.contiguous()
    assert not y.is_contiguous()
    assert z.is_contiguous()


class ContiguousNet(nn.Cell):
    def construct(self, x, w):
        output = ops.matmul(x, w)
        output = ops.transpose(output, (2, 1, 0))
        output = output[..., 1]
        output = output.contiguous()
        output = output * output
        return output


class WithoutContiguousNet(nn.Cell):
    def construct(self, x, w):
        output = ops.matmul(x, w)
        output = ops.transpose(output, (2, 1, 0))
        output = output[..., 1]
        output = output * output
        return output

@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_contiguous_grad(mode):
    """
    Feature: countiguous
    Description: Verify the result of grad
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor(np.random.randn(16, 32, 32), mstype.float32)
    weight = Tensor(np.random.randn(32, 16), mstype.float32)

    contiguous_net = ContiguousNet()
    no_contiguous_net = WithoutContiguousNet()
    grad = ops.GradOperation(get_all=True)

    contiguous_grad_output = grad(contiguous_net)(x, weight)
    no_contiguous_grad_output = grad(no_contiguous_net)(x, weight)

    assert np.allclose(contiguous_grad_output[0].asnumpy(), no_contiguous_grad_output[0].asnumpy())
    assert np.allclose(contiguous_grad_output[1].asnumpy(), no_contiguous_grad_output[1].asnumpy())
