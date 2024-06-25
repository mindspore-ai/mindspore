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
# pylint: disable=unused-variable
import pytest
import numpy as np
from mindspore import Tensor, context
from mindspore import ops
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def batch_norm_grad_forward_func(dout, x, scale, mean, variance, reserve):
    batch_norm_grad = ops.auto_generate.BatchNormGrad(is_training=True,
                                                      epsilon=1e-5,
                                                      data_format="NCHW")
    out = batch_norm_grad(dout, x, scale, mean, variance, reserve)
    return out[0]


@test_utils.run_with_cell
def batch_norm_grad_backward_func(dout, x, scale, mean, variance, reserve):
    return ops.grad(batch_norm_grad_forward_func, 0)(dout, x, scale, mean,
                                                     variance, reserve)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_bn_grad_forward(mode):
    """
    Feature: Ops.
    Description: test BatchNormGrad.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)
    dout = Tensor(np.random.rand(10, 36, 12, 12).astype(np.float32))
    x = Tensor(np.random.rand(10, 36, 12, 12).astype(np.float32))
    scale = Tensor(np.random.rand(36).astype(np.float32))
    mean = Tensor(np.random.rand(36).astype(np.float32))
    variance = Tensor(np.random.rand(36).astype(np.float32))
    reserve = Tensor(np.random.rand(36).astype(np.float32))
    output = batch_norm_grad_forward_func(dout, x, scale, mean, variance,
                                          reserve)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_bn_grad_backward(mode):
    """
    Feature: Ops.
    Description: test BatchNormGradGrad.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)
    dout = Tensor(np.random.rand(10, 36, 12, 12).astype(np.float32))
    x = Tensor(np.random.rand(10, 36, 12, 12).astype(np.float32))
    scale = Tensor(np.random.rand(36).astype(np.float32))
    mean = Tensor(np.random.rand(36).astype(np.float32))
    variance = Tensor(np.random.rand(36).astype(np.float32))
    reserve = Tensor(np.random.rand(36).astype(np.float32))
    grads = batch_norm_grad_backward_func(dout, x, scale, mean, variance,
                                          reserve)
