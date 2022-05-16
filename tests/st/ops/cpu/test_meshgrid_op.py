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
from mindspore.ops import operations as P


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype',
                         [np.bool, np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64,
                          np.float16, np.float32, np.float64])
@pytest.mark.parametrize('indexing', ["xy", "ij"])
def test_meshgrid(dtype, indexing):
    """
    Feature: Meshgrid cpu kernel
    Description: test the rightness of Meshgrid cpu kernel
    Expectation: the output is same as np output
    """
    class NetMeshgrid(nn.Cell):
        def __init__(self):
            super(NetMeshgrid, self).__init__()
            self.meshgrid = P.Meshgrid(indexing)

        def construct(self, inputs):
            return self.meshgrid(inputs)

    meshgrid = NetMeshgrid()
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    x = np.random.uniform(low=0, high=10, size=3).astype(dtype)
    y = np.random.uniform(low=0, high=10, size=4).astype(dtype)
    np_output = np.meshgrid(x, y, indexing=indexing)
    output = meshgrid((Tensor(x), Tensor(y)))
    assert np.array_equal(output[0].asnumpy(), np_output[0])
    assert np.array_equal(output[1].asnumpy(), np_output[1])

    z = np.random.uniform(low=0, high=10, size=5).astype(dtype)
    np_output = np.meshgrid(x, y, z, indexing=indexing)
    output = meshgrid((Tensor(x), Tensor(y), Tensor(z)))
    assert np.array_equal(output[0].asnumpy(), np_output[0])
    assert np.array_equal(output[1].asnumpy(), np_output[1])
    assert np.array_equal(output[2].asnumpy(), np_output[2])
