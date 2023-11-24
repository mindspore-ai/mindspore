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
import torch
import pytest
import numpy as np
import mindspore
from mindspore import context, nn, Tensor
from mindspore.ops.operations.math_ops import Renorm
from mindspore.ops import functional as F

np.random.seed(5)


class NetAscend(nn.Cell):
    def __init__(self, p, dim, max_norm):
        super(NetAscend, self).__init__()
        self.p = p
        self.dim = dim
        self.max_norm = max_norm
        self.renorm = Renorm(p=self.p, dim=self.dim, maxnorm=self.max_norm)

    def construct(self, x):
        y = self.renorm(x)
        return y


class NetTorch:
    def __init__(self, array, p, dim, max_norm):
        self.tensor = array
        self.p = p
        self.dim = dim
        self.max_norm = max_norm

    def test_float16(self):
        tensor = torch.tensor(self.tensor, dtype=torch.float16)
        y = torch.renorm(tensor, self.p, self.dim, self.max_norm)
        return y.numpy()

    def test_float32(self):
        tensor = torch.tensor(self.tensor, dtype=torch.float32)
        y = torch.renorm(tensor, self.p, self.dim, self.max_norm)
        return y.numpy()


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_renorm_p1_fp32(mode):
    """
    Feature: test renorm
    Description: test renorm with input tensor's type float32, p=1
    Expectation: none.
    """
    context.set_context(mode=mode, device_target="Ascend")
    p = 1
    dim = 0
    max_norm = 0.5
    a = np.random.random([2, 3, 4, 5]).astype(np.float32)

    tensor = Tensor(a, mindspore.float32)
    ms_net = NetAscend(p, dim, max_norm)
    ms_out = ms_net(tensor)

    torch_net = NetTorch(a, p, dim, max_norm)
    torch_out = torch_net.test_float32()

    assert np.allclose(torch_out, ms_out.asnumpy(), 0.0001, 0.0001)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_renorm_p1_fp32_dyn(mode):
    """
    Feature: test renorm dynamic shape
    Description: test renorm with input tensor's type float32, p=1
    Expectation: none.
    """
    context.set_context(mode=mode, device_target="Ascend")
    p = 1
    dim = 0
    max_norm = 0.5
    a = np.random.random([2, 3, 4, 5]).astype(np.float32)

    tensor = Tensor(a, mindspore.float32)
    dyn_tensor = Tensor(shape=[None, 3, 4, 5], dtype=mindspore.float32)
    ms_net = NetAscend(p, dim, max_norm)

    ms_net.set_inputs(dyn_tensor)

    ms_out = ms_net(tensor)

    torch_net = NetTorch(a, p, dim, max_norm)
    torch_out = torch_net.test_float32()

    assert np.allclose(torch_out, ms_out.asnumpy(), 0.0001, 0.0001)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_renorm_p2_fp32(mode):
    """
    Feature: test renorm
    Description: test renorm with input tensor's type float32, p=2
    Expectation: none.
    """
    context.set_context(mode=mode, device_target="Ascend")
    p = 2
    dim = 0
    max_norm = 0.5
    a = np.random.random([2, 3, 4, 5]).astype(np.float32)

    tensor = Tensor(a, mindspore.float32)
    ms_net = NetAscend(p, dim, max_norm)
    ms_out = ms_net(tensor)

    torch_net = NetTorch(a, p, dim, max_norm)
    torch_out = torch_net.test_float32()

    assert np.allclose(torch_out, ms_out.asnumpy(), 0.0001, 0.0001)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_renorm_p3_fp32(mode):
    """
    Feature: test renorm
    Description: test renorm with input tensor's type float32, p=3
    Expectation: none.
    """
    context.set_context(mode=mode, device_target="Ascend")
    p = 3
    dim = 0
    max_norm = 0.5
    a = np.random.random([2, 3, 4, 5]).astype(np.float32)

    tensor = Tensor(a, mindspore.float32)
    ms_net = NetAscend(p, dim, max_norm)
    ms_out = ms_net(tensor)

    torch_net = NetTorch(a, p, dim, max_norm)
    torch_out = torch_net.test_float32()

    assert np.allclose(torch_out, ms_out.asnumpy(), 0.0001, 0.0001)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_renorm_p1_fp16(mode):
    """
    Feature: test renorm
    Description: test renorm with input tensor's type float16, p=1
    Expectation: none.
    """
    context.set_context(mode=mode, device_target="Ascend")
    p = 1
    dim = 1
    max_norm = 0.5
    a = np.random.random([2, 3, 4, 5]).astype(np.float16)

    tensor = Tensor(a, mindspore.float16)
    ms_net = NetAscend(p, dim, max_norm)
    ms_out = ms_net(tensor)

    torch_net = NetTorch(a, p, dim, max_norm)
    torch_out = torch_net.test_float16()
    assert np.allclose(torch_out, ms_out.asnumpy(), 0.0001, 0.0001)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_renorm_p2_fp16(mode):
    """
    Feature: test renorm
    Description: test renorm with input tensor's type float16, p=2
    Expectation: none.
    """
    context.set_context(mode=mode, device_target="Ascend")
    p = 2
    dim = 1
    max_norm = 0.5
    a = np.random.random([2, 3, 4, 5]).astype(np.float16)

    tensor = Tensor(a, mindspore.float16)
    ms_net = NetAscend(p, dim, max_norm)
    ms_out = ms_net(tensor)

    torch_net = NetTorch(a, p, dim, max_norm)
    torch_out = torch_net.test_float16()
    assert np.allclose(torch_out, ms_out.asnumpy(), 0.0001, 0.0001)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_renorm_p3_fp16(mode):
    """
    Feature: test renorm
    Description: test renorm with input tensor's type float16, p=3
    Expectation: none.
    """
    context.set_context(mode=mode, device_target="Ascend")
    p = 3
    dim = -2
    max_norm = 0.5
    a = np.random.random([2, 3, 4, 5]).astype(np.float16)

    tensor = Tensor(a, mindspore.float16)
    ms_net = NetAscend(p, dim, max_norm)
    ms_out = ms_net(tensor)

    torch_net = NetTorch(a, p, dim, max_norm)
    torch_out = torch_net.test_float16()
    assert np.allclose(torch_out, ms_out.asnumpy(), 0.0001, 0.0001)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_tensor_renorm(mode):
    """
    Feature: test tensor's renorm
    Description: test tensor's renorm with type float16, p=3
    Expectation: none.
    """
    context.set_context(mode=mode, device_target="Ascend")
    p = 3
    dim = -2
    max_norm = 0.5
    a = np.random.random([2, 3, 4, 5]).astype(np.float16)

    tensor = Tensor(a, mindspore.float16)
    ms_out = tensor.renorm(p, dim, max_norm)

    torch_net = NetTorch(a, p, dim, max_norm)
    torch_out = torch_net.test_float16()
    assert np.allclose(torch_out, ms_out.asnumpy(), 0.0001, 0.0001)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_renorm_functional(mode):
    """
    Feature: test functional renorm
    Description: test functional renorm with type float16, p=3
    Expectation: none.
    """
    context.set_context(mode=mode, device_target="Ascend")
    p = 3
    dim = -2
    max_norm = 0.5
    a = np.random.random([2, 3, 4, 5]).astype(np.float16)

    tensor = Tensor(a, mindspore.float16)
    ms_out = F.renorm(tensor, p, dim, max_norm)

    torch_net = NetTorch(a, p, dim, max_norm)
    torch_out = torch_net.test_float16()
    assert np.allclose(torch_out, ms_out.asnumpy(), 0.0001, 0.0001)
