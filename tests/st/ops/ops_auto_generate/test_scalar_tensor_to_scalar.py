# Copyright 2025 Huawei Technologies Co., Ltd
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

"""test feature: scalar tensor to scalar for yaml-registered ops"""
from tests.mark_utils import arg_mark
import numpy as np
import pytest
import mindspore as ms


class Argsort(ms.nn.Cell):
    def construct(self, x, dim):
        return ms.ops.auto_generate.argsort_ext(x, dim)


class TensorArgsort(ms.nn.Cell):
    def construct(self, x, dim):
        return x.argsort(dim=dim)


class DropoutExt(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.op = ms.ops.auto_generate.DropoutExt()

    def construct(self, x, p, seed, offset):
        return self.op(x, p, seed, offset)


class TensorAdd(ms.nn.Cell):
    def construct(self, x, y, alpha):
        return x.add(y, alpha=alpha)

class Add(ms.nn.Cell):
    def construct(self, x, y, alpha):
        return ms.ops.auto_generate.add_ext(x, y, alpha)


class Round(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.op = ms.ops.auto_generate.Round()

    def construct(self, x, decimals):
        return self.op(x, decimals)


class TensorRound(ms.nn.Cell):
    def construct(self, x, decimals):
        return x.round(decimals)


class Norm(ms.nn.Cell):
    def construct(self, x, p):
        return ms.mint.norm(x, p)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_scalar_tensor_to_int(mode):
    """
    Feature: default cast of scalar tensor to int
    Description: int input support 0-dim scalar tensor input by default
    Expectation: success
    """
    ms.set_context(mode=mode, jit_level="O0")

    net = Argsort()
    tensor_net = TensorArgsort()

    x = ms.Tensor([1, 2, 3])
    expected_output = np.array([0, 1, 2])
    for dtype in [ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8]:
        dim = ms.Tensor(0, dtype=dtype)
        output = net(x, dim)
        assert np.allclose(output.asnumpy(), expected_output, rtol=0, atol=0)
        output = tensor_net(x, dim)
        assert np.allclose(output.asnumpy(), expected_output, rtol=0, atol=0)

    # non-0-dim tensor is not supported by default
    dim = ms.Tensor([0])
    with pytest.raises((ValueError, TypeError)):
        net(x, dim)
    with pytest.raises((ValueError, TypeError)):
        tensor_net(x, dim)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_scalar_tensor_to_float(mode):
    """
    Feature: default cast of scalar tensor to float
    Description: float input support 0-dim scalar tensor input by default
    Expectation: success
    """
    ms.set_context(mode=mode, jit_level="O0")

    net = DropoutExt()

    x = ms.Tensor([1., 1., 1.], dtype=ms.float32)
    expected_output = ms.Tensor([2., 2., 0.], dtype=ms.float32)
    for dtype in [ms.float16, ms.float32, ms.float64]:
        p = ms.Tensor(0.5, dtype=dtype)
        seed = ms.Tensor(2, dtype=ms.int64)
        offset = ms.Tensor(2, dtype=ms.int64)
        output = net(x, p, seed, offset)
        assert np.allclose(output[0].asnumpy(),
                           expected_output, rtol=0, atol=0)

    # non-0-dim tensor is not supported by default
    p = ms.Tensor([0.5], dtype=ms.float32)
    with pytest.raises((ValueError, TypeError)):
        net(x, p)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_scalar_tensor_to_number(mode):
    """
    Feature: default cast of scalar tensor
    Description: number input support 0-dim scalar tensor input by default
    Expectation: success
    """
    ms.set_context(mode=mode, jit_level="O0")

    net = Add()
    tensor_net = TensorAdd()

    # test int scalar
    x = ms.Tensor(2)
    y = ms.Tensor(3)
    expected_output = 17
    for dtype in [ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8]:
        alpha = ms.Tensor(5, dtype=dtype)
        output = net(x, y, alpha)
        assert np.allclose(output.asnumpy(), expected_output, rtol=0, atol=0)
        output = tensor_net(x, y, alpha)
        assert np.allclose(output.asnumpy(), expected_output, rtol=0, atol=0)

    # test float scalar
    x = ms.Tensor(2.0)
    y = ms.Tensor(3.0)
    expected_output = 17.0
    for dtype in [ms.float16, ms.float32, ms.float64]:
        alpha = ms.Tensor(5, dtype=dtype)
        output = net(x, y, alpha)
        assert np.allclose(output.asnumpy(), expected_output, rtol=0, atol=0)
        output = tensor_net(x, y, alpha)
        assert np.allclose(output.asnumpy(), expected_output, rtol=0, atol=0)

    # non-0-dim tensor is not supported by default
    alpha = ms.Tensor([5], dtype=dtype)
    with pytest.raises((ValueError, TypeError)):
        net(x, y, alpha)
    with pytest.raises((ValueError, TypeError)):
        tensor_net(x, y, alpha)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_non_zero_dim_tensor_to_scalar(mode):
    """
    Feature: tensor type_cast to scalar
    Description: scalar input with type_cast tensor support single-element tensor
    Expectation: success
    """
    ms.set_context(mode=mode, jit_level="O0")

    net = Round()
    tensor_net = TensorRound()

    # test int scalar
    x = ms.Tensor([1., 2., 3.])
    expected_output = np.array([1., 2., 3.])
    for decimals in (0, ms.Tensor(0), ms.Tensor([0])):
        output = net(x, decimals)
        assert np.allclose(output.asnumpy(), expected_output, rtol=0, atol=0)
        output = tensor_net(x, decimals)
        assert np.allclose(output.asnumpy(), expected_output, rtol=0, atol=0)

    # element number > 1 not supported
    decimals = ms.Tensor([0, 0])
    with pytest.raises((ValueError, TypeError)):
        output = net(x, decimals)
    with pytest.raises((ValueError, TypeError)):
        output = tensor_net(x, decimals)
