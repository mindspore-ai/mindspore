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
from tests.mark_utils import arg_mark
# pylint: disable=unused-variable

import pytest
import numpy as np

from mindspore import Tensor, context
from mindspore import ops
from mindspore.ops.function.nn_func import batch_norm_ext
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP

@test_utils.run_with_cell
def batch_norm_forward_func(x, scale, bias, mean, var, training=False, momentum=0.1, eps=1e-5):
    out = batch_norm_ext(x, scale, bias, mean, var, training, momentum, eps)
    return out


@test_utils.run_with_cell
def batch_norm_backward_func(x, scale, bias, mean, var, is_train=False, momentum=0.1, eps=1e-5):
    return ops.grad(batch_norm_forward_func, 0)(x, scale, bias, mean, var, is_train, momentum, eps)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("is_training", [True, False])
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_batchnormext_normal(is_training, mode):
    """
    Feature: Ops.
    Description: test BatchNorm.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)
    x = Tensor((3 * np.ones(16)).reshape(2, 2, 1, 4).astype(np.float32))
    scale = Tensor(np.ones(2).astype(np.float32))
    bias = Tensor(np.ones(2).astype(np.float32))
    mean = Tensor(np.ones(2).astype(np.float32))
    variance = Tensor(np.ones(2).astype(np.float32))

    expect = None
    if is_training:
        expect = np.array([1.0000305])
    else:
        expect = np.array([2.99999])
    expect = expect.repeat(16).astype(np.float32).reshape(2, 2, 1, 4)

    output = batch_norm_forward_func(x, scale, bias, mean, variance,
                                     is_training)
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


    grad = batch_norm_backward_func(x, scale, bias, mean, variance,
                                    is_training)
    grad_expect = None
    if is_training:
        grad_expect = np.array([0.])
    else:
        grad_expect = np.array([0.999995])
    grad_expect = grad_expect.repeat(16).astype(np.float32).reshape(2, 2, 1, 4)

    assert np.allclose(grad.asnumpy(), grad_expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_bn_vmap(mode):
    """
    Feature: test vmap function.
    Description: test BatchNorm op vmap.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)
    shape = (2, 2, 2, 2, 2, 2)
    in_axes = (-1, -1, -1, -1, -1, None)
    x = np.ones(64).astype(np.float32)
    x = Tensor(x.reshape(shape))
    scale = Tensor(np.ones((2, 2, 2)).astype(np.float32))
    bias = Tensor(np.ones((2, 2, 2)).astype(np.float32))
    mean = Tensor(np.ones((2, 2, 2)).astype(np.float32))
    var = Tensor(np.ones((2, 2, 2)).astype(np.float32))
    nest_vmap = ops.vmap(ops.vmap(batch_norm_forward_func,
                                  in_axes=in_axes,
                                  out_axes=0),
                         in_axes=in_axes,
                         out_axes=0)
    out = nest_vmap(x, scale, bias, mean, var, False)

    expect = np.ones(64).astype(np.float32)
    expect = expect.reshape(shape)
    assert np.allclose(out.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bn_dyn():
    """
    Feature: Ops.
    Description: test BatchNorm dynamic shape and dynamic rank.
    Expectation: expect correct result.
    """
    input_x1 = np.random.randn(*(1, 2, 4, 4)).astype(np.float32)
    input_x2 = np.random.randn(*(2, 3, 3)).astype(np.float32)
    scale1 = Tensor(np.ones(2).astype(np.float32))
    bias1 = Tensor(np.ones(2).astype(np.float32))
    mean1 = Tensor(np.ones(2).astype(np.float32))
    variance1 = Tensor(np.ones(2).astype(np.float32))
    scale2 = Tensor(np.ones(3).astype(np.float32))
    bias2 = Tensor(np.ones(3).astype(np.float32))
    mean2 = Tensor(np.ones(3).astype(np.float32))
    variance2 = Tensor(np.ones(3).astype(np.float32))
    training1 = True
    training2 = False
    momentum1 = 0.01
    momentum2 = 0.001
    eps1 = 1e-5
    eps2 = 1e-7

    TEST_OP(batch_norm_forward_func,
            [[Tensor(input_x1), scale1, bias1, mean1, variance1, training1, momentum1, eps1],
             [Tensor(input_x2), scale2, bias2, mean2, variance2, training2, momentum2, eps2]],
            '', disable_input_check=True, disable_yaml_check=True, disable_mode=['GRAPH_MODE'])
