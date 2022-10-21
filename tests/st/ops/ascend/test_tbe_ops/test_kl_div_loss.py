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
import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.functional as F
from mindspore import Tensor
from mindspore.common.api import jit
from mindspore.ops import operations as P
from mindspore.ops.functional import vmap


class TestKLDivLossNet(nn.Cell):
    def __init__(self, reduction):
        super(TestKLDivLossNet, self).__init__()
        self.kl_div_loss = P.KLDivLoss(reduction=reduction)

    def construct(self, x, target):
        return self.kl_div_loss(x, target)


def kl_div_loss_np(x, target, reduction):
    out = target * (np.log(target) - x)
    out = np.nan_to_num(out, nan=0.)

    if reduction == "none":
        return out
    if reduction == "batchmean":
        return np.sum(out) / out.shape[0]
    if reduction == "sum":
        return np.sum(out)
    raise RuntimeError("reduction should be one of ['none', 'batchmean', 'sum']")


def compare_with_numpy(x, target, reduction):
    x_ms = Tensor(x)
    target_ms = Tensor(target)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    out = TestKLDivLossNet(reduction)(x_ms, target_ms)
    expected = kl_div_loss_np(x, target, reduction)
    np.testing.assert_array_almost_equal(out.asnumpy(), expected)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    out = TestKLDivLossNet(reduction)(x_ms, target_ms)
    expected = kl_div_loss_np(x, target, reduction)
    np.testing.assert_array_almost_equal(out.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("reduction", ["none", "sum"])
@pytest.mark.parametrize("data_type", [np.float16, np.float32])
def test_kl_div_loss_scalar(reduction, data_type):
    """
    Feature: KLDivLoss operators.
    Description: test cases for KLDivLoss operator
    Expectation: the result match numpy implementation.
    """
    x = data_type(0.7)
    target = data_type(1.)
    compare_with_numpy(x, target, reduction)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("reduction", ["none", "sum", "batchmean"])
@pytest.mark.parametrize("data_type", [np.float16, np.float32])
def test_kl_div_loss_multi_dim(reduction, data_type):
    """
    Feature: KLDivLoss operators.
    Description: test cases for KLDivLoss operator
    Expectation: the result match numpy implementation.
    """
    x = np.array([[0.2, 0.7, 0.1], [-0.1, 3., 0.9]]).astype(data_type)
    target = np.array([[1., 0., 0.1], [0.6, -1., 4.]]).astype(data_type)
    compare_with_numpy(x, target, reduction)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("reduction", ["none", "sum", "batchmean"])
def test_kl_div_loss_vmap(reduction):
    """
    Feature: vmap of KLDivLoss operators.
    Description: test cases for vmap of KLDivLoss operator
    Expectation: the result matched.
    """
    def cal_kl_div_loss(x, target):
        return P.KLDivLoss(reduction)(x, target)

    @jit
    def manually_batched(xs, targets):
        output = []
        for i in range(xs.shape[-1]):
            inner_output = []
            for j in range(xs[:, :, i].shape[1]):
                inner_output.append(cal_kl_div_loss(xs[:, j, i], targets[:, j, i]))
            output.append(F.stack(inner_output))
        return F.stack(output)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    x = Tensor(np.random.rand(4, 4, 4).astype(np.float32))
    target = Tensor(np.random.rand(4, 4, 4).astype(np.float32))

    vmap_kl_div_loss = vmap(
        vmap(cal_kl_div_loss, in_axes=(1, 1), out_axes=0),
        in_axes=(-1, -1), out_axes=0,
    )

    outputs = vmap_kl_div_loss(x, target)
    expect = manually_batched(x, target)

    np.testing.assert_allclose(outputs.asnumpy(), expect.asnumpy(), rtol=1e-4, atol=1e-3)
