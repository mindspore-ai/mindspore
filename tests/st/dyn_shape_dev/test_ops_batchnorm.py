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
from mindspore import Tensor, context, Parameter
from mindspore import ops
import test_utils


@test_utils.run_with_cell
def batch_norm_forward_func(x, scale, bias, mean, var, is_train=False):
    out = ops.auto_generate.batch_norm_(x,
                                        scale,
                                        bias,
                                        mean,
                                        var,
                                        is_train,
                                        epsilon=1e-5,
                                        momentum=0.1,
                                        data_format="NCHW")
    return out[0]


@test_utils.run_with_cell
def batch_norm_backward_func(x, scale, bias, mean, var, is_train=False):
    return ops.grad(batch_norm_forward_func, 0)(x, scale, bias, mean, var,
                                                is_train)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("is_training", [True, False])
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_bn_forward(is_training, mode):
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

    if is_training:
        scale = Parameter(scale)
        bias = Parameter(bias)
        mean = Parameter(mean)
        variance = Parameter(variance)
    output = batch_norm_forward_func(x, scale, bias, mean, variance,
                                     is_training)
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("is_training", [True, False])
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_bn_backward(is_training, mode):
    """
    Feature: Ops.
    Description: test BatchNormGrad.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)
    x = Tensor((3 * np.ones(16)).reshape(2, 2, 1, 4).astype(np.float32))
    scale = Tensor(np.ones(2).astype(np.float32))
    bias = Tensor(np.ones(2).astype(np.float32))
    mean = Tensor(np.ones(2).astype(np.float32))
    variance = Tensor(np.ones(2).astype(np.float32))
    if is_training:
        scale = Parameter(scale)
        bias = Parameter(bias)
        mean = Parameter(mean)
        variance = Parameter(variance)
    grad = batch_norm_backward_func(x, scale, bias, mean, variance,
                                    is_training)

    expect = None
    if is_training:
        expect = np.array([0.])
    else:
        expect = np.array([0.999995])
    expect = expect.repeat(16).astype(np.float32).reshape(2, 2, 1, 4)

    assert np.allclose(grad.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("is_training", [True, False])
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_bn_backward_ascend(is_training, mode):
    """
    Feature: Ops.
    Description: test BatchNormGrad.
    Expectation: expect correct result.
    """
    if mode == context.PYNATIVE_MODE:
        # There are still some problems in ascend acl.
        return
    context.set_context(mode=mode)
    x = Tensor((3 * np.ones(16)).reshape(2, 2, 1, 4).astype(np.float32))
    scale = Tensor(np.ones(2).astype(np.float32))
    bias = Tensor(np.ones(2).astype(np.float32))
    mean = Tensor(np.ones(2).astype(np.float32))
    variance = Tensor(np.ones(2).astype(np.float32))
    if is_training:
        scale = Parameter(scale)
        bias = Parameter(bias)
        mean = Parameter(mean)
        variance = Parameter(variance)
    grad = batch_norm_backward_func(x, scale, bias, mean, variance,
                                    is_training)

    expect = None
    if is_training:
        expect = np.array([0.])
    else:
        expect = np.array([0.999995])
    expect = expect.repeat(16).astype(np.float32).reshape(2, 2, 1, 4)

    assert np.allclose(grad.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_bn_vmap(mode):
    """
    Feature: test vmap function.
    Description: test BatchNorm op vmap.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)
    shape = (2, 2, 2, 2)
    in_axes = (-1, -1, -1, -1, -1, None)
    x = np.ones(16).astype(np.float32)
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

    expect = np.ones(16).astype(np.float32)
    expect = expect.reshape(shape)
    assert np.allclose(out.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.skip(reason="There are still some problems in ascend.")
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_bn_vmap_ascend(mode):
    """
    Feature: test vmap function.
    Description: test BatchNorm op vmap.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)
    shape = (2, 2, 2, 2)
    in_axes = (-1, -1, -1, -1, -1, None)
    x = np.ones(16).astype(np.float32)
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

    expect = np.ones(16).astype(np.float32)
    expect = expect.reshape(shape)
    assert np.allclose(out.asnumpy(), expect, rtol=1e-4, atol=1e-4)
