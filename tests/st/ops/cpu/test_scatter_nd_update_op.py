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
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import Parameter
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_op1():
    class ScatterNdUpdate(nn.Cell):
        def __init__(self):
            super(ScatterNdUpdate, self).__init__()
            self.scatter_nd_update = P.ScatterNdUpdate()
            self.x = Parameter(Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mstype.float32), name="x")

        def construct(self, indices, update):
            return self.scatter_nd_update(self.x, indices, update)

    indices = Tensor(np.array([[0, 0], [1, 1]]), mstype.int32)
    update = Tensor(np.array([1.0, 2.2]), mstype.float32)

    scatter_nd_update = ScatterNdUpdate()
    scatter_nd_update(indices, update)
    print("x:\n", scatter_nd_update.x.data.asnumpy())
    expect = [[1.0, 0.3, 3.6], [0.4, 2.2, -3.2]]
    assert np.allclose(scatter_nd_update.x.data.asnumpy(), np.array(expect, np.float))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_op2():
    class ScatterNdUpdate(nn.Cell):
        def __init__(self):
            super(ScatterNdUpdate, self).__init__()
            self.scatter_nd_update = P.ScatterNdUpdate()
            self.x = Parameter(Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), mstype.float32), name="x")

        def construct(self, indices, update):
            return self.scatter_nd_update(self.x, indices, update)

    indices = Tensor(np.array([[4], [3], [1], [7]]), mstype.int32)
    update = Tensor(np.array([9, 10, 11, 12]), mstype.float32)

    scatter_nd_update = ScatterNdUpdate()
    scatter_nd_update(indices, update)
    print("x:\n", scatter_nd_update.x.data.asnumpy())
    expect = [1, 11, 3, 10, 9, 6, 7, 12]
    assert np.allclose(scatter_nd_update.x.data.asnumpy(), np.array(expect, dtype=float))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_op3():
    class ScatterNdUpdate(nn.Cell):
        def __init__(self):
            super(ScatterNdUpdate, self).__init__()
            self.scatter_nd_update = P.ScatterNdUpdate()
            self.x = Parameter(Tensor(np.zeros((4, 4, 4)), mstype.float32), name="x")

        def construct(self, indices, update):
            return self.scatter_nd_update(self.x, indices, update)

    indices = Tensor(np.array([[0], [2]]), mstype.int32)
    update = Tensor(np.array([[[5, 5, 5, 5], [6, 6, 6, 6],
                               [7, 7, 7, 7], [8, 8, 8, 8]],
                              [[5, 5, 5, 5], [6, 6, 6, 6],
                               [7, 7, 7, 7], [8, 8, 8, 8]]]), mstype.float32)

    scatter_nd_update = ScatterNdUpdate()
    scatter_nd_update(indices, update)
    print("x:\n", scatter_nd_update.x.data.asnumpy())
    expect = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
              [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
              [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
              [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
    assert np.allclose(scatter_nd_update.x.data.asnumpy(), np.array(expect, dtype=float))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_op4():
    class ScatterNdUpdate(nn.Cell):
        def __init__(self):
            super(ScatterNdUpdate, self).__init__()
            self.scatter_nd_update = P.ScatterNdUpdate()
            self.x = Parameter(Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mstype.float32), name="x")

        def construct(self, indices, update):
            return self.scatter_nd_update(self.x, indices, update)

    indices = Tensor(np.array([[0, 1]]), mstype.int32)
    update = Tensor(np.array([1.0]), mstype.float32)

    scatter_nd_update = ScatterNdUpdate()
    out = scatter_nd_update(indices, update)
    print("x:\n", out)
    assert np.allclose(out.asnumpy(), scatter_nd_update.x.data.asnumpy())
    expect = [[-0.1, 1.0, 3.6], [0.4, 0.5, -3.2]]
    assert np.allclose(out.asnumpy(), np.array(expect, np.float))
