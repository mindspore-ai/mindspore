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
from mindspore import Tensor, ops
from mindspore.ops import operations as P


class SqueezeNet(nn.Cell):
    def __init__(self):
        super(SqueezeNet, self).__init__()
        self.squeeze = P.Squeeze()

    def construct(self, tensor):
        return self.squeeze(tensor)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("data_type",
                         [np.bool, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64,
                          np.uint64, np.float16, np.float32, np.float64, np.complex64, np.complex128])
def test_squeeze(data_type):
    """
    Feature: Test Squeeze GPU.
    Description: The input data type contains common valid types including bool
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    np.random.seed(0)
    x = np.random.randn(1, 16, 1, 1).astype(data_type)
    net = SqueezeNet()
    output = net(Tensor(x))
    assert np.all(output.asnumpy() == x.squeeze())


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_func():
    """
    Feature: Test Squeeze GPU.
    Description: Test functional api.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    np.random.seed(0)
    x = np.random.randn(1, 16, 1, 1).astype(np.int32)

    output = ops.squeeze(Tensor(x))
    assert np.all(output.asnumpy() == x.squeeze())

    output = ops.squeeze(Tensor(x), 0)
    assert np.all(output.asnumpy() == x.squeeze(0))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tensor():
    """
    Feature: Test Squeeze GPU.
    Description: Test Tensor api.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    np.random.seed(0)
    x = np.random.randn(1, 16, 1, 1).astype(np.int32)

    output = Tensor(x).squeeze()
    assert np.all(output.asnumpy() == x.squeeze())

    output = Tensor(x).squeeze(0)
    assert np.all(output.asnumpy() == x.squeeze(0))
