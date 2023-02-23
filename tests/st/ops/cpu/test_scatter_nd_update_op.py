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
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_op1(dtype):
    """
    Feature: Op ScatterNdUpdate
    Description:  test ScatterNdUpdate
    Expectation: success
    """

    class ScatterNdUpdate(nn.Cell):
        def __init__(self):
            super(ScatterNdUpdate, self).__init__()
            self.scatter_nd_update = P.ScatterNdUpdate()
            self.x = Parameter(
                Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]], dtype=dtype)), name="x")

        def construct(self, indices, update):
            return self.scatter_nd_update(self.x, indices, update)

    indices = Tensor(np.array([[0, 0], [1, 1]]), mstype.int32)
    update = Tensor(np.array([1.0, 2.2], dtype=dtype))

    scatter_nd_update = ScatterNdUpdate()
    output = scatter_nd_update(indices, update)
    print("x:\n", output.asnumpy())
    expect = [[1.0, 0.3, 3.6], [0.4, 2.2, -3.2]]
    assert np.allclose(output.asnumpy(),
                       np.array(expect, dtype=dtype))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int32, np.int64])
def test_op2(dtype):
    """
    Feature: Op ScatterNdUpdate
    Description:  test ScatterNdUpdate
    Expectation: success
    """

    class ScatterNdUpdate(nn.Cell):
        def __init__(self):
            super(ScatterNdUpdate, self).__init__()
            self.scatter_nd_update = P.ScatterNdUpdate()
            self.x = Parameter(
                Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=dtype)), name="x")

        def construct(self, indices, update):
            return self.scatter_nd_update(self.x, indices, update)

    indices = Tensor(np.array([[4], [3], [1], [7]]), mstype.int32)
    update = Tensor(np.array([9, 10, 11, 12], dtype=dtype))

    scatter_nd_update = ScatterNdUpdate()
    output = scatter_nd_update(indices, update)
    print("x:\n", output.asnumpy())
    expect = [1, 11, 3, 10, 9, 6, 7, 12]
    assert np.allclose(output.asnumpy(),
                       np.array(expect, dtype=dtype))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int32, np.int64])
def test_op3(dtype):
    """
    Feature: Op ScatterNdUpdate
    Description:  test ScatterNdUpdate
    Expectation: success
    """

    class ScatterNdUpdate(nn.Cell):
        def __init__(self):
            super(ScatterNdUpdate, self).__init__()
            self.scatter_nd_update = P.ScatterNdUpdate()
            self.x = Parameter(Tensor(np.zeros((4, 4, 4)).astype(dtype)), name="x")

        def construct(self, indices, update):
            return self.scatter_nd_update(self.x, indices, update)

    indices = Tensor(np.array([[0], [2]]), mstype.int32)
    update = Tensor(np.array([[[5, 5, 5, 5], [6, 6, 6, 6],
                               [7, 7, 7, 7], [8, 8, 8, 8]],
                              [[5, 5, 5, 5], [6, 6, 6, 6],
                               [7, 7, 7, 7], [8, 8, 8, 8]]], dtype=dtype))

    scatter_nd_update = ScatterNdUpdate()
    output = scatter_nd_update(indices, update)
    print("x:\n", output.asnumpy())
    expect = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
              [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
              [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
              [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
    assert np.allclose(output.asnumpy(), np.array(expect, dtype=dtype))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_op4(dtype):
    """
    Feature: Op ScatterNdUpdate
    Description:  test ScatterNdUpdate
    Expectation: success
    """

    class ScatterNdUpdate(nn.Cell):
        def __init__(self):
            super(ScatterNdUpdate, self).__init__()
            self.scatter_nd_update = P.ScatterNdUpdate()
            self.x = Parameter(Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]], dtype=dtype)), name="x")

        def construct(self, indices, update):
            return self.scatter_nd_update(self.x, indices, update)

    indices = Tensor(np.array([[0, 1]]), mstype.int32)
    update = Tensor(np.array([1.0], dtype=dtype))

    scatter_nd_update = ScatterNdUpdate()
    out = scatter_nd_update(indices, update)
    print("x:\n", out)
    expect = [[-0.1, 1.0, 3.6], [0.4, 0.5, -3.2]]
    assert np.allclose(out.asnumpy(), np.array(expect, dtype=dtype))


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_op5(dtype):
    """
    Feature: Op ScatterNdUpdate
    Description:  test ScatterNdUpdate with index out of range
    Expectation: raise ValueError
    """

    class ScatterNdUpdate(nn.Cell):
        def __init__(self):
            super(ScatterNdUpdate, self).__init__()
            self.scatter_nd_update = P.ScatterNdUpdate()
            self.x = Parameter(Tensor(np.ones([1, 4, 1], dtype=dtype)), name="x")

        def construct(self, indices, update):
            return self.scatter_nd_update(self.x, indices, update)

    indices = Tensor(np.array([[0, 2], [3, 2], [1, 3]]), mstype.int32)
    update = Tensor(np.array([[1], [1], [1]], dtype=dtype))

    scatter_nd_update = ScatterNdUpdate()
    with pytest.raises(ValueError):
        scatter_nd_update(indices, update)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_scatter_nd_update_dyn_shape():
    """
    Feature: op dynamic shape
    Description: set input_shape None and input real tensor
    Expectation: success
    """

    class ScatterNdUpdate(nn.Cell):
        def __init__(self):
            super(ScatterNdUpdate, self).__init__()
            self.scatter_nd_update = P.ScatterNdUpdate()


        def construct(self, x, indices, update):
            return self.scatter_nd_update(x, indices, update)

    indices = Tensor(np.array([[0, 0], [1, 1]]), mstype.int32)
    update = Tensor(np.array([1.0, 2.2], dtype=np.float32))
    input_x = Parameter(Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]], dtype=np.float32)))
    indices_dyn = Tensor(shape=[None, None], dtype=indices.dtype)
    update_dyn = Tensor(shape=[None], dtype=update.dtype)
    scatter_nd_update = ScatterNdUpdate()
    scatter_nd_update.set_inputs(input_x, indices_dyn, update_dyn)
    output = scatter_nd_update(input_x, indices, update)
    expect_shape = (2, 3)
    assert output.asnumpy().shape == expect_shape
