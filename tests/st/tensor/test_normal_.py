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
# pylint: disable=unused-variable

import numpy as np
import mindspore as ms
from mindspore import ops, Generator, jit
from tests.mark_utils import arg_mark
import pytest


class Net(ms.nn.Cell):
    def construct(self, x, mean=0, std=1, generator=None):
        x = x * 1
        x.normal_(mean, std, generator=generator)
        return x


def generate_random_input(shape):
    return np.random.randn(*shape).astype(np.float32)


def normal_forward_func(x, mean=0, std=1, generator=None):
    return Net()(x, mean, std, generator)


@jit(backend="ms_backend")
def normal_backward_func(x):
    grad = ops.GradOperation(get_all=True)
    return grad(Net())(x)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
    mem_peak=2.00
)
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_normal_backward(mode):
    """
    Feature: pyboost function.
    Description: test function Tensor.normal_ backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')

    x = generate_random_input((2, 2, 3, 4))
    expect_x_grad = np.zeros_like(x, dtype=np.float32)
    output_x_grad = normal_backward_func(ms.Tensor(x))
    np.allclose(output_x_grad[0].asnumpy(), expect_x_grad)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
    mem_peak=2.00
)
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_normal_forward(mode):
    """
    Feature: pyboost function.
    Description: test function Tensor.normal_ forward.
    Expectation: expect correct result shape.
    """
    ms.context.set_context(mode=mode)
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')

    expect_shape = (10, 10)
    x = generate_random_input((10, 10))
    output = normal_forward_func(ms.Tensor(x))
    assert output.shape == expect_shape
    # Check whether the update is performed inplace.
    assert not ops.equal(ms.Tensor(x), output).all()

    expect_shape = (20, 20)
    mean = 2.0
    std = 1.0
    generator = Generator()
    x = generate_random_input((20, 20))
    output = normal_forward_func(ms.Tensor(x), mean=mean, std=std, generator=generator)
    assert output.shape == expect_shape

    # Test the `mean` and `std` input of different types.
    mean = 10
    std = 1
    output = normal_forward_func(ms.Tensor(x), mean=mean, std=std, generator=generator)
    assert output.shape == expect_shape
