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
from mindspore import Tensor
import mindspore.ops.operations.array_ops as P


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.ops = P.NonZero()

    def construct(self, x):
        return self.ops(x)


def compare_with_numpy(x):
    net = Net()
    # Graph Mode
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    ms_result_graph = net(Tensor(x))
    # PyNative Mode
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    ms_result_pynative = net(Tensor(x))

    np_result = np.transpose(np.nonzero(x))
    return np.array_equal(ms_result_graph, np_result) and np.array_equal(ms_result_pynative, np_result)


@pytest.mark.level0
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
    np.random.seed(1)
    x = np.random.randint(low=-1, high=2, size=data_shape).astype(data_type)
    assert compare_with_numpy(x)
