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

import mindspore.context as context
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops.operations.nn_ops as nn_ops


class NetNthElement(nn.Cell):
    def __init__(self, reverse):
        super().__init__()
        self.nth_element = nn_ops.NthElement(reverse=reverse)

    def construct(self, x, k):
        return self.nth_element(x, k)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_nth_element_1d():
    """
    Feature: test nth_element to find the t-th item in 1D input
    Description: 1D x, 0D n
    Expectation: success
    """
    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")
        x = Tensor(np.array([1, 20, 5]).astype(np.int32))
        input_x = x.asnumpy()
        n = Tensor(1).astype("int32")
        input_n = n.asnumpy()
        net = NetNthElement(reverse=True)
        y = net(x, n)
        expect = np.sort(input_x, axis=-1)[..., ::-1][..., input_n].astype(np.int32)
        assert (y.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_nth_element_2d():
    """
    Feature: test nth_element to find the t-th item in 2D input
    Description: 2D x, 0D n
    Expectation: success
    """
    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")
        x = Tensor(np.array([[11., 20., 52.], [67., 18., 29.], [130., 24., 5.], [0.3, -0.4, -15.]]).astype(np.float32))
        input_x = x.asnumpy()
        n = Tensor(1).astype("int32")
        input_n = n.asnumpy()
        net = NetNthElement(reverse=True)
        y = net(x, n)
        expect = np.sort(input_x, axis=-1)[..., ::-1][..., input_n].astype(np.float32)
        assert (y.asnumpy() == expect).all()
