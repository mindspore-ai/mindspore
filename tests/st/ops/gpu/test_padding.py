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
import pytest
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self, pad_dim_size):
        super(Net, self).__init__()
        self.padding = P.Padding(pad_dim_size)

    def construct(self, x):
        return self.padding(x)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('shape', [(2, 1), (2, 4, 1), (3, 4, 5, 1)])
@pytest.mark.parametrize('dtype', [np.uint32, np.float16, np.float32])
@pytest.mark.parametrize('pad_dim_size', [2, 4, 10])
def test_padding(mode, shape, dtype, pad_dim_size):
    """
    Feature: ALL To ALL
    Description: test cases for padding
    Expectation: the result match to numpy
    """
    context.set_context(mode=mode, device_target="GPU")
    prop = 100 if np.random.random() > 0.5 else -100
    x = (np.random.randn(*shape) * prop).astype(dtype)
    padding = Net(pad_dim_size)
    output = padding(Tensor(x))
    pad_width = [(0, 0) for _ in range(len(shape) - 1)]
    pad_width.append((0, pad_dim_size - 1))
    expect = np.pad(x, tuple(pad_width), 'constant', constant_values=0)
    np.testing.assert_array_almost_equal(output.asnumpy(), expect)
