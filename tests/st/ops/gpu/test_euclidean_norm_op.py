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
from tests.mark_utils import arg_mark

import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import math_ops as P


class Net(nn.Cell):
    def __init__(self, keep_dims=False):
        super(Net, self).__init__()
        self.euclideannorm = P.EuclideanNorm(keep_dims)

    def construct(self, x, axes):
        return self.euclideannorm(x, axes)


#euclideannorm op will be deleted soon since ops.norm has same functionality.
#@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("data_type1", [
    np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float16, np.float32,
    np.float64, np.complex64, np.complex128
])
@pytest.mark.parametrize("data_type2", [np.int32, np.int64])
def test_euclideannorm_graph(data_type1, data_type2):
    """
    Feature: EuclideanNorm
    Description: Test of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    loss = 1e-6
    input_x = Tensor(
        np.array([[[4, 0, 9, 5, 5], [9, 9, 5, 6, 5], [8, 6, 9, 3, 8], [7, 0, 3, 5, 9]],
                  [[4, 1, 0, 0, 3], [4, 9, 5, 2, 5], [6, 4, 3, 3, 6], [8, 4, 8, 2, 5]],
                  [[0, 5, 6, 6, 7], [8, 4, 9, 6, 9], [4, 6, 9, 1, 0], [1, 4, 8, 2, 7]]]).astype(data_type1))
    axes = Tensor(np.array([0]).astype(data_type2))
    net = Net()
    output = net(input_x, axes).asnumpy()
    expect = np.array([[5.65685425, 5.09901951, 10.81665383, 7.81024968, 9.11043358],
                       [12.68857754, 13.34166406, 11.44552314, 8.71779789, 11.44552314],
                       [10.77032961, 9.38083152, 13.07669683, 4.35889894, 10],
                       [10.67707825, 5.65685425, 11.70469991, 5.74456265, 12.4498996]]).astype(data_type1)
    assert np.allclose(output, expect, rtol=loss, atol=loss)


#@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("data_type1", [
    np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float16, np.float32,
    np.float64, np.complex64, np.complex128
])
@pytest.mark.parametrize("data_type2", [np.int32, np.int64])
def test_euclideannorm_pynative(data_type1, data_type2):
    """
    Feature: EuclideanNorm
    Description: Test of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    loss = 1e-6
    input_x = Tensor(
        np.array([[[4, 0, 9, 5, 5], [9, 9, 5, 6, 5], [8, 6, 9, 3, 8], [7, 0, 3, 5, 9]],
                  [[4, 1, 0, 0, 3], [4, 9, 5, 2, 5], [6, 4, 3, 3, 6], [8, 4, 8, 2, 5]],
                  [[0, 5, 6, 6, 7], [8, 4, 9, 6, 9], [4, 6, 9, 1, 0], [1, 4, 8, 2, 7]]]).astype(data_type1))
    axes = Tensor(np.array([1, 2]).astype(data_type2))
    net = Net()
    output = net(input_x, axes).asnumpy()
    expect = np.array([28.51315486, 21.3541565, 26.30589288]).astype(data_type1)
    assert np.allclose(output, expect, rtol=loss, atol=loss)


#@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("data_type1", [np.complex64, np.complex128])
@pytest.mark.parametrize("data_type2", [np.int32, np.int64])
def test_euclideannorm_complex_keep_dims(data_type1, data_type2):
    """
    Feature: EuclideanNorm
    Description: Test of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    loss = 1e-6
    input_x = Tensor(
        np.array([[[[-4.5, -1.5], [7.0, 6.0]], [[2.5, 0.5], [3.0, 9.0]]],
                  [[[-4.5, -1.5], [7.0, 6.0]], [[2.5, 0.5], [3.0, 9.0]]]]).astype(data_type1))
    axes = Tensor(np.array([0, 2]).astype(data_type2))
    keep_dims = True
    net = Net(keep_dims)
    output = net(input_x, axes).asnumpy()
    expect = np.array([[[[11.7686023 + 0.j, 8.74642784 + 0.j]], [[5.52268051 + 0.j,
                                                                  12.74754878 + 0.j]]]]).astype(data_type1)
    assert np.allclose(output, expect, rtol=loss, atol=loss)


#@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_euclideannorm_same_axes_with_input():
    """
    Feature: EuclideanNorm
    Description: Test of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    loss = 1e-6
    input_x = Tensor(
        np.array([[[2, 0, 4], [2, 5, 2], [9, 8, 6]], [[2, 7, 7], [3, 6, 8], [7, 7, 2]],
                  [[6, 0, 7], [1, 1, 7], [8, 5, 3]]]).astype(np.float32))
    axes = Tensor(np.array([0, 1, 2]).astype(np.int64))
    net = Net()
    output = net(input_x, axes).asnumpy()
    expect = np.array(27.946377).astype(np.float32)
    assert np.allclose(output, expect, rtol=loss, atol=loss)
