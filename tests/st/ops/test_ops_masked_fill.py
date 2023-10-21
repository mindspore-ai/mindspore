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
import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor, ops, nn
from mindspore import value_and_grad


class Net(nn.Cell):
    def construct(self, x, mask, value):
        return ops.masked_fill(x, mask, value)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_masked_fill_grad_dtype(mode):
    """
    Feature: test the grad of ops.masked_fill
    Description: test the grad value and type of ops.masked_fill
    Expectation: success
    """
    ms.set_context(mode=mode)

    x = Tensor([1, 2, 3, 4], ms.float32)
    mask = Tensor([True, False, True, False], ms.bool_)
    value = Tensor(6, ms.float32)
    net = Net()
    grad_fn = value_and_grad(net, grad_position=(0, 1, 2))
    output, inputs_grad = grad_fn(x, mask, value)

    expect_out = np.array([6, 2, 6, 4], np.float32)
    expect_grad_x = np.array([0, 1, 0, 1], np.float32)
    expect_grad_mask = np.array([False, False, False, False], np.bool_)
    expect_grad_value = np.array(2, np.float32)

    assert output.asnumpy().dtype == expect_out.dtype
    assert inputs_grad[0].asnumpy().dtype == expect_grad_x.dtype
    assert inputs_grad[1].asnumpy().dtype == expect_grad_mask.dtype
    assert inputs_grad[2].asnumpy().dtype == expect_grad_value.dtype

    assert np.allclose(output.asnumpy(), expect_out)
    assert np.allclose(inputs_grad[0].asnumpy(), expect_grad_x)
    assert np.allclose(inputs_grad[1].asnumpy(), expect_grad_mask)
    assert np.allclose(inputs_grad[2].asnumpy(), expect_grad_value)
