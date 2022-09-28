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
import mindspore.nn as nn
from mindspore import Tensor, context, ops
from mindspore import dtype as mstype
from mindspore.ops.operations.array_ops import MatrixSetDiagV3


class MatrixSetDiagV3Net(nn.Cell):
    def __init__(self, align='RIGHT_LEFT'):
        super(MatrixSetDiagV3Net, self).__init__()
        self.matrix_set_diag_v3 = MatrixSetDiagV3(align=align)

    def construct(self, x, diagonal, k):
        return self.matrix_set_diag_v3(x, diagonal, k)


def get_dy_shape(real_shape):
    """
    Feature: generate a dynamic shape for mindspore dynamic shape
    Description: The shape set all shape none
    Expectation: match to test ops's input real shape.
    """
    part_shape_list = [None for _ in real_shape]
    return part_shape_list


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_matrix_set_diag_v3_function():
    """
    Feature: matrix_set_diag_v3 functional api.
    Description: Compatible with expect.
    Expectation: The result matches numpy.
    """
    context.set_context(device_target="GPU")
    input_x = Tensor(np.array([[[5, 5, 5, 5],
                                [5, 5, 5, 5],
                                [5, 5, 5, 5]],
                               [[5, 5, 5, 5],
                                [5, 5, 5, 5],
                                [5, 5, 5, 5]]]), mstype.float32)
    diagonal = Tensor(np.array([[1, 2, 3],
                                [4, 5, 6]]), mstype.float32)
    result = ops.matrix_set_diag(input_x, diagonal).asnumpy()
    expect = np.array([[[1, 5, 5, 5],
                        [5, 2, 5, 5],
                        [5, 5, 3, 5]],
                       [[4, 5, 5, 5],
                        [5, 5, 5, 5],
                        [5, 5, 6, 5]]], np.float32)
    np.testing.assert_allclose(result, expect)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_matrix_set_diag_v3_ops():
    """
    Feature: MatrixSetDiagV3 operator.
    Description: Compatible with expect.
    Expectation: The result matches numpy.
    """
    context.set_context(device_target="GPU", mode=context.GRAPH_MODE)
    input_x = Tensor(np.array([[[5, 5, 5, 5],
                                [5, 5, 5, 5],
                                [5, 5, 5, 5]],
                               [[5, 5, 5, 5],
                                [5, 5, 5, 5],
                                [5, 5, 5, 5]]]), mstype.float32)
    diagonal = Tensor(np.array([[1, 2, 3],
                                [4, 5, 6]]), mstype.float32)
    expect = np.array([[[5, 1, 5, 5],
                        [5, 5, 2, 5],
                        [5, 5, 5, 3]],
                       [[5, 4, 5, 5],
                        [5, 5, 5, 5],
                        [5, 5, 5, 6]]], np.float32)
    k = Tensor(1, mstype.int32)
    result = MatrixSetDiagV3Net()(input_x, diagonal, k).asnumpy()
    np.testing.assert_allclose(result, expect)
    context.set_context(device_target="GPU", mode=context.PYNATIVE_MODE)
    result = MatrixSetDiagV3Net()(input_x, diagonal, k).asnumpy()
    np.testing.assert_allclose(result, expect)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_matrix_set_diag_v3_ops_band():
    """
    Feature: MatrixSetDiagV3 operator.
    Description: Compatible with expect.
    Expectation: The result matches numpy.
    """
    context.set_context(device_target="GPU", mode=context.GRAPH_MODE)
    input_x = Tensor(np.array([[[5, 5, 5, 5],
                                [5, 5, 5, 5],
                                [5, 5, 5, 5]],
                               [[5, 5, 5, 5],
                                [5, 5, 5, 5],
                                [5, 5, 5, 5]]]), mstype.float32)

    k = Tensor((-1, 2), mstype.int32)
    diagonal = Tensor(np.array([[[0, 9, 1],
                                 [6, 5, 8],
                                 [1, 2, 3],
                                 [4, 5, 0]],
                                [[0, 1, 2],
                                 [5, 6, 4],
                                 [6, 1, 2],
                                 [3, 4, 0]]]), mstype.float32)
    result = MatrixSetDiagV3Net()(input_x, diagonal, k).asnumpy()
    expect = np.array([[[1, 6, 9, 5],
                        [4, 2, 5, 1],
                        [5, 5, 3, 8]],
                       [[6, 5, 1, 5],
                        [3, 1, 6, 2],
                        [5, 4, 2, 4]]], np.float32)
    np.testing.assert_allclose(result, expect)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_matrix_set_diag_v3_ops_align():
    """
    Feature: MatrixSetDiagV3 operator.
    Description: Compatible with expect.
    Expectation: The result matches numpy.
    """
    context.set_context(device_target="GPU", mode=context.GRAPH_MODE)
    align = "LEFT_RIGHT"
    input_x = Tensor(np.array([[[5, 5, 5, 5],
                                [5, 5, 5, 5],
                                [5, 5, 5, 5]],
                               [[5, 5, 5, 5],
                                [5, 5, 5, 5],
                                [5, 5, 5, 5]]]), mstype.float32)

    k = Tensor((-1, 2), mstype.int32)
    diagonal = Tensor(np.array([[[9, 1, 0],
                                 [6, 5, 8],
                                 [1, 2, 3],
                                 [0, 4, 5]],
                                [[1, 2, 0],
                                 [5, 6, 4],
                                 [6, 1, 2],
                                 [0, 3, 4]]]), mstype.float32)
    result = MatrixSetDiagV3Net(align=align)(input_x, diagonal, k).asnumpy()
    expect = np.array([[[1, 6, 9, 5],
                        [4, 2, 5, 1],
                        [5, 5, 3, 8]],
                       [[6, 5, 1, 5],
                        [3, 1, 6, 2],
                        [5, 4, 2, 4]]], np.float32)
    np.testing.assert_allclose(result, expect)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_matrix_set_diag_v3_vmap():
    """
    Feature: MatrixSetDiagV3 operator.
    Description: Compatible with expect.
    Expectation: The result matches numpy.
    """
    context.set_context(device_target="GPU")
    # vmap input x shape is [2,2,3,4]
    input_x = Tensor(np.array([[[[5, 5, 5, 5],
                                 [5, 5, 5, 5],
                                 [5, 5, 5, 5]],
                                [[5, 5, 5, 5],
                                 [5, 5, 5, 5],
                                 [5, 5, 5, 5]]],
                               [[[7, 7, 7, 7],
                                 [7, 7, 7, 7],
                                 [7, 7, 7, 7]],
                                [[7, 7, 7, 7],
                                 [7, 7, 7, 7],
                                 [7, 7, 7, 7]]]]), mstype.float32)
    # vmap diagonal shape is [2,2,3]
    diagonal = Tensor(np.array([[[1, 2, 3],
                                 [4, 5, 6]],
                                [[1, 2, 3],
                                 [4, 5, 6]]]), mstype.float32)
    # vmap output shape is  [2,2,3,4]
    expect = np.array([[[[5, 1, 5, 5],
                         [5, 5, 2, 5],
                         [5, 5, 5, 3]],
                        [[5, 4, 5, 5],
                         [5, 5, 5, 5],
                         [5, 5, 5, 6]]],
                       [[[7, 1, 7, 7],
                         [7, 7, 2, 7],
                         [7, 7, 7, 3]],
                        [[7, 4, 7, 7],
                         [7, 7, 5, 7],
                         [7, 7, 7, 6]]]], np.float32)
    k = Tensor(1, mstype.int32)
    matrix_set_diag_net = MatrixSetDiagV3Net()
    result = ops.vmap(matrix_set_diag_net, (0, 0, None))(input_x, diagonal, k).asnumpy()
    np.testing.assert_allclose(result, expect)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_matrix_set_diag_v3_dynamic_shape():
    """
    Feature: MatrixSetDiagV3 dynamic shape operator.
    Description: Compatible with expect.
    Expectation: The result matches numpy.
    """
    context.set_context(device_target="GPU", mode=context.GRAPH_MODE)
    input_x = Tensor(np.array([[[5, 5, 5, 5],
                                [5, 5, 5, 5],
                                [5, 5, 5, 5]],
                               [[5, 5, 5, 5],
                                [5, 5, 5, 5],
                                [5, 5, 5, 5]]]), mstype.float32)
    diagonal = Tensor(np.array([[1, 2, 3],
                                [4, 5, 6]]), mstype.float32)
    expect = np.array([[[5, 1, 5, 5],
                        [5, 5, 2, 5],
                        [5, 5, 5, 3]],
                       [[5, 4, 5, 5],
                        [5, 5, 5, 5],
                        [5, 5, 5, 6]]], np.float32)
    k = Tensor(1, mstype.int32)
    matrix_set_diag_net = MatrixSetDiagV3Net()
    dy_input_x = Tensor(shape=get_dy_shape(input_x.shape), dtype=input_x.dtype)
    dy_diagonal = Tensor(shape=get_dy_shape(diagonal.shape), dtype=diagonal.dtype)
    matrix_set_diag_net.set_inputs(dy_input_x, dy_diagonal, k)
    result = matrix_set_diag_net(input_x, diagonal, k).asnumpy()
    np.testing.assert_allclose(result, expect)
    context.set_context(device_target="GPU", mode=context.PYNATIVE_MODE)
    matrix_set_diag_net.set_inputs(dy_input_x, dy_diagonal, k)
    result = matrix_set_diag_net(input_x, diagonal, k).asnumpy()
    np.testing.assert_allclose(result, expect)
