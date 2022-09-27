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
from mindspore import Parameter
from mindspore.ops import functional as F
from mindspore.ops.operations.array_ops import TensorScatterElements

context.set_context(mode=context.GRAPH_MODE, device_target='GPU')


def scatter_element_np(input_x, indices, updates, axis, reduction="none"):
    result = input_x.asnumpy().copy()
    indices_np = indices.asnumpy().copy()
    updates_np = updates.asnumpy().copy()

    i_len = indices_np.shape[0]
    j_len = indices_np.shape[1]

    if axis < 0:
        axis += len(result.shape)

    for i in range(i_len):
        for j in range(j_len):
            if axis == 0:
                if reduction == "none":
                    result[indices_np[i][j]][j] = updates_np[i][j]
                if reduction == "add":
                    result[indices_np[i][j]][j] += updates_np[i][j]
            if axis == 1:
                if reduction == "none":
                    result[i][indices_np[i][j]] = updates_np[i][j]
                if reduction == "add":
                    result[i][indices_np[i][j]] += updates_np[i][j]

    return result


class TestTensorScatterElements(nn.Cell):
    def __init__(self, input_x, indices, updates, axis, reduction):
        super(TestTensorScatterElements, self).__init__()
        self.axis = axis
        self.reduction = reduction
        self.input_x = Parameter(input_x, name="input_x")
        self.indices = Parameter(indices, name="indices")
        self.updates = Parameter(updates, name="updates")
        self.scatter_elements = TensorScatterElements(
            self.axis, self.reduction)

    def construct(self):
        return self.scatter_elements(self.input_x, self.indices, self.updates)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int32])
@pytest.mark.parametrize('index_dtype', [np.int32, np.int64])
@pytest.mark.parametrize('axis', [0, 1, -1])
@pytest.mark.parametrize('reduction', ["none", "add"])
def test_scatter_elements(dtype, index_dtype, axis, reduction):
    """
    Feature: Op TensorScatterElements
    Description: Scatter update value according indices to output.
                output[indices[i][j]][j] = updates[i][j] if axis = 0, reduction="none"
                output[i][indices[i][j]] += updates[i][j] if axis = 1, reduction="add"
    Expectation: Ans is same as expected.
    """
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype))
    indices = Tensor(np.array([[-1, 0, 1], [0, 1, 2]], dtype=index_dtype))
    update = Tensor(np.array([[1, 2, 2], [4, 5, 8]], dtype=dtype))

    ms_output = TestTensorScatterElements(
        x, indices, update, axis, reduction)()
    np_output = scatter_element_np(x, indices, update, axis, reduction)
    print("ms_output:\n", ms_output.asnumpy())
    assert np.allclose(ms_output.asnumpy(), np_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float32])
@pytest.mark.parametrize('index_dtype', [np.int32])
@pytest.mark.parametrize('axis', [0])
@pytest.mark.parametrize('reduction', ["none", "add"])
def test_scatter_add_with_axis_func(dtype, index_dtype, axis, reduction):
    """
    Feature: test scatter_add_with_axis functional interface(scatter_add).
    Description: Scatter update value according indices to output.
                output[indices[i][j]][j] += updates[i][j] if axis = 0,
                output[i][indices[i][j]] += updates[i][j] if axis = 1.
    Expectation: Ans is same as expected.
    """
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype))
    indices = Tensor(np.array([[-1, 0, 1], [0, 1, 2]], dtype=index_dtype))
    update = Tensor(np.array([[1, 2, 2], [4, 5, 8]], dtype=dtype))

    #cause scatter_add will change the value of input, so we first calculate numpy output.
    np_output = scatter_element_np(x, indices, update, axis, reduction)
    ms_output = F.tensor_scatter_elements(x, indices, update, axis, reduction)
    print("np_output:\n", np_output)
    print("ms_output:\n", ms_output.asnumpy())
    assert np.allclose(ms_output.asnumpy(), np_output)
