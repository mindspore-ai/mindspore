# Copyright 2020 Huawei Technologies Co., Ltd
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
import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class Net(nn.Cell):
    def __init__(self, reduction="none"):
        super(Net, self).__init__()
        self.kl_div_loss = P.KLDivLoss(reduction)

    def construct(self, x, y):
        return self.kl_div_loss(x, y)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_mode_none_and_dtype_with_static_input(mode, dtype):
    """
    Feature: test none mode with different input dtype.
    Description: input with negative elements.
    Expectation: success.
    """
    context.set_context(mode=mode)
    np.random.seed(42)
    prediction = mindspore.Tensor(np.log(np.array([[0.3, 0.7], [0.5, 0.5]])).astype(dtype))
    target = mindspore.Tensor(np.array([[-1, 1], [1, -1]]).astype(dtype))
    net = Net("none")
    loss = net(Tensor(prediction), Tensor(target))
    expect = np.array([[0, 0.35667494], [0.69314718, 0]]).astype(dtype)
    assert np.allclose(loss.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_mode_mean_and_dtype_with_static_input(mode, dtype):
    """
    Feature: test mean mode with different input dtype.
    Description: input with negative elements.
    Expectation: success.
    """
    context.set_context(mode=mode)
    np.random.seed(42)
    prediction = mindspore.Tensor(np.log(np.array([[0.3, 0.7], [0.5, 0.5]])).astype(dtype))
    target = mindspore.Tensor(np.array([[-1, 1], [1, -1]]).astype(dtype))
    net = Net("mean")
    loss = net(Tensor(prediction), Tensor(target))
    expect = np.array([0.26245553]).astype(dtype)
    assert np.allclose(loss.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_mode_sum_and_dtype_with_static_input(mode, dtype):
    """
    Feature: test sum mode with different input dtype.
    Description: input with negative elements.
    Expectation: success.
    """
    context.set_context(mode=mode)
    np.random.seed(42)
    prediction = mindspore.Tensor(np.log(np.array([[0.3, 0.7], [0.5, 0.5]])).astype(dtype))
    target = mindspore.Tensor(np.array([[-1, 1], [1, -1]]).astype(dtype))
    net = Net("sum")
    loss = net(Tensor(prediction), Tensor(target))
    expect = np.array([1.04982212]).astype(dtype)
    assert np.allclose(loss.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_mode_batchmean_and_dtype_with_static_input(mode, dtype):
    """
    Feature: test batchmean mode with different input dtype.
    Description: input with negative elements.
    Expectation: success.
    """
    context.set_context(mode=mode)
    np.random.seed(42)
    prediction = mindspore.Tensor(np.log(np.array([[0.3, 0.7], [0.5, 0.5]])).astype(dtype))
    target = mindspore.Tensor(np.array([[-1, 1], [1, -1]]).astype(dtype))
    net = Net("batchmean")
    loss = net(Tensor(prediction), Tensor(target))
    expect = np.array([0.52491106]).astype(dtype)
    assert np.allclose(loss.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("dtype", [np.float32])
def test_mode_none_and_dtype_with_dynamic_input(mode, dtype):
    """
    Feature: test none mode with different input dtype.
    Description: input with negative elements.
    Expectation: success.
    """
    context.set_context(mode=mode)
    np.random.seed(42)
    prediction = mindspore.Tensor(np.log(np.array([[0.3, 0.7], [0.5, 0.5]])).astype(dtype))
    target = mindspore.Tensor(np.array([[-1, 1], [1, -1]]).astype(dtype))
    net = Net("none")
    dyn_prediction = Tensor(shape=[None, None], dtype=mindspore.float32)
    dyn_target = Tensor(shape=[None, None], dtype=mindspore.float32)
    net.set_inputs(dyn_prediction, dyn_target)
    loss = net(Tensor(prediction), Tensor(target))
    expect = np.array([[0, 0.35667494], [0.69314718, 0]]).astype(dtype)
    assert np.allclose(loss.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("dtype", [np.float32])
def test_mode_mean_and_dtype_with_dynamic_input(mode, dtype):
    """
    Feature: test mean mode with different input dtype.
    Description: input with negative elements.
    Expectation: success.
    """
    context.set_context(mode=mode)
    np.random.seed(42)
    prediction = mindspore.Tensor(np.log(np.array([[0.3, 0.7], [0.5, 0.5]])).astype(dtype))
    target = mindspore.Tensor(np.array([[-1, 1], [1, -1]]).astype(dtype))
    net = Net("mean")
    dyn_prediction = Tensor(shape=[None, None], dtype=mindspore.float32)
    dyn_target = Tensor(shape=[None, None], dtype=mindspore.float32)
    net.set_inputs(dyn_prediction, dyn_target)
    loss = net(Tensor(prediction), Tensor(target))
    expect = np.array([0.26245553]).astype(dtype)
    assert np.allclose(loss.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("dtype", [np.float32])
def test_mode_sum_and_dtype_with_dynamic_input(mode, dtype):
    """
    Feature: test sum mode with different input dtype.
    Description: input with negative elements.
    Expectation: success.
    """
    context.set_context(mode=mode)
    np.random.seed(42)
    prediction = mindspore.Tensor(np.log(np.array([[0.3, 0.7], [0.5, 0.5]])).astype(dtype))
    target = mindspore.Tensor(np.array([[-1, 1], [1, -1]]).astype(dtype))
    net = Net("sum")
    dyn_prediction = Tensor(shape=[None, None], dtype=mindspore.float32)
    dyn_target = Tensor(shape=[None, None], dtype=mindspore.float32)
    net.set_inputs(dyn_prediction, dyn_target)
    loss = net(Tensor(prediction), Tensor(target))
    expect = np.array([1.04982212]).astype(dtype)
    assert np.allclose(loss.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("dtype", [np.float32])
def test_mode_batchmean_and_dtype_with_dynamic_input(mode, dtype):
    """
    Feature: test batchmean mode with different input dtype.
    Description: input with negative elements.
    Expectation: success.
    """
    context.set_context(mode=mode)
    np.random.seed(42)
    prediction = mindspore.Tensor(np.log(np.array([[0.3, 0.7], [0.5, 0.5]])).astype(dtype))
    target = mindspore.Tensor(np.array([[-1, 1], [1, -1]]).astype(dtype))
    net = Net("batchmean")
    dyn_prediction = Tensor(shape=[None, None], dtype=mindspore.float32)
    dyn_target = Tensor(shape=[None, None], dtype=mindspore.float32)
    net.set_inputs(dyn_prediction, dyn_target)
    loss = net(Tensor(prediction), Tensor(target))
    expect = np.array([0.52491106]).astype(dtype)
    assert np.allclose(loss.asnumpy(), expect)
