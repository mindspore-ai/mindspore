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
from tests.mark_utils import arg_mark

import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F


class NetMeshgrid(nn.Cell):
    def __init__(self, indexing="xy"):
        super(NetMeshgrid, self).__init__()
        self.meshgrid = P.Meshgrid(indexing)

    def construct(self, inputs):
        return self.meshgrid(inputs)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('indexing', ["xy", "ij"])
def test_meshgrid(indexing):
    """
    Feature: Meshgrid ascend kernel
    Description: test the rightness of Meshgrid ascend kernel
    Expectation: the output is same as np output
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    meshgrid = NetMeshgrid(indexing)
    x = np.random.uniform(low=0, high=10, size=3).astype(np.float32)
    y = np.random.uniform(low=0, high=10, size=4).astype(np.float32)
    np_output = np.meshgrid(x, y, indexing=indexing)
    output = meshgrid((Tensor(x), Tensor(y)))
    assert np.array_equal(output[0].asnumpy(), np_output[0])
    assert np.array_equal(output[1].asnumpy(), np_output[1])

    # test functional interface
    output = F.meshgrid(Tensor(x), Tensor(y), indexing=indexing)
    assert np.array_equal(output[0].asnumpy(), np_output[0])
    assert np.array_equal(output[1].asnumpy(), np_output[1])
