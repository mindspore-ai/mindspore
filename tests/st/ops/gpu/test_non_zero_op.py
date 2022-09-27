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
import mindspore.ops.operations as P
from mindspore import Tensor
from mindspore.ops.operations.array_ops import NonZero


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.ops = NonZero()

    def construct(self, x):
        return self.ops(x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('data_shape', [(10, 10), (3, 4, 5)])
@pytest.mark.parametrize('data_type',
                         [np.int8, np.int16, np.int32, np.int64, np.float16,
                          np.float32, np.float64, np.uint8, np.uint16])
def test_net(data_shape, data_type):
    """
    Feature: NonZero
    Description:  test cases for NonZero operator.
    Expectation: the result match numpy nonzero.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    np.random.seed(1)
    x = np.random.randint(low=-1, high=2, size=data_shape).astype(data_type)
    net = Net()
    ms_result = net(Tensor(x))
    np_result = np.transpose(np.nonzero(x))
    assert np.array_equal(ms_result, np_result)


class DynamicShapeNet(nn.Cell):
    def __init__(self, axis=0):
        super(DynamicShapeNet, self).__init__()
        self.unique = P.Unique()
        self.gather = P.Gather()
        self.nonzero = NonZero()
        self.axis = axis

    def construct(self, x, indices):
        unique_indices, _ = self.unique(indices)
        real_x = self.gather(x, unique_indices, self.axis)
        return real_x, self.nonzero(real_x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dyn_net():
    """
    Feature: NonZero
    Description:  test cases for NonZero operator in dynamic shape.
    Expectation: the result match numpy nonzero.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    np.random.seed(1)
    x = Tensor(np.random.randint(low=-1, high=2, size=(8, 8, 8, 8)).astype(np.float32))
    indices = Tensor(np.random.randint(0, 8, size=8))

    net = DynamicShapeNet()
    real_x, ms_result = net(x, indices)
    np_result = np.transpose(np.nonzero(real_x.asnumpy()))
    assert np.array_equal(ms_result, np_result)
