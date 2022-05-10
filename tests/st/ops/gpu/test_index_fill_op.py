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
import mindspore.common.dtype as mstype
import mindspore.ops.operations.array_ops as P


class IndexFillNet(nn.Cell):
    def __init__(self):
        super(IndexFillNet, self).__init__()
        self.index_fill = P.IndexFill()

    def construct(self, x, dim, index, value):
        out = self.index_fill(x, dim, index, value)
        return out


def compare_with_numpy(x, dim, index, value):
    # Graph Mode
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    ms_x = Tensor(x)
    ms_dim = dim
    ms_index = Tensor(index)
    ms_value = value
    ms_result_graph = IndexFillNet()(ms_x, ms_dim, ms_index, ms_value).asnumpy()
    # PyNative Mode
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    ms_result_pynative = IndexFillNet()(ms_x, ms_dim, ms_index, ms_value).asnumpy()

    # Numpy
    np_result = x.copy()
    if dim == 0:
        np_result[index] = value
    elif dim == 1:
        np_result[:, index] = value
    elif dim == 2:
        np_result[:, :, index] = value
    else:
        raise ValueError("dim must be 0, 1 or 2")

    return np.allclose(ms_result_graph, np_result) and np.allclose(ms_result_pynative, np_result)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('data_type', [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64])
def test_index_fill_data_type(data_type):
    """
    Feature: IndexFill
    Description:  test cases for IndexFill operator with multiple data types.
    Expectation: the result match numpy.
    """
    dim_type = np.int32
    dim = Tensor(np.array(1, dtype=dim_type))
    value = Tensor(np.array(-10, dtype=data_type))
    x_np = np.random.random(size=(5, 5, 5)).astype(data_type)
    index_np = np.random.randint(low=0, high=5, size=4).astype(np.int32)
    assert compare_with_numpy(x_np, dim, index_np, value)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dim_type', [np.int32, np.int64])
def test_index_fill_dim_type(dim_type):
    """
    Feature: IndexFill
    Description:  test cases for IndexFill operator with multiple index types.
    Expectation: the result match numpy.
    """
    data_type = np.float32
    dim = Tensor(np.array(2, dtype=dim_type))
    value = Tensor(np.array(-10, dtype=data_type))
    x_np = np.random.randint(20, size=(5, 5, 5)).astype(data_type)
    index_np = np.random.randint(low=0, high=5, size=4).astype(np.int32)
    assert compare_with_numpy(x_np, dim, index_np, value)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('value, data_type', [(10, np.int64), (10., np.float32)])
def test_index_fill_scalar(value, data_type):
    """
    Feature: IndexFill
    Description:  test cases for IndexFill operator with scalar input.
    Expectation: the result match numpy.
    """
    dim = 0
    x_np = np.random.randint(20, size=(5, 5, 5)).astype(data_type)
    index_np = np.random.randint(low=0, high=5, size=4).astype(np.int32)
    assert compare_with_numpy(x_np, dim, index_np, value)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dim', [0, 1])
@pytest.mark.parametrize('data_type', [np.int32])
def test_index_fill_error(dim, data_type):
    """
    Feature: IndexFill
    Description:  test cases for IndexFill operator that is out of bound.
    Expectation: raise RuntimeError.
    """
    ms_x = Tensor([[1, 2], [3, 4]]).astype(data_type)
    ms_index = Tensor([2]).astype(np.int32)
    ms_dim = Tensor(dim, dtype=mstype.int32)
    ms_value = Tensor(20, dtype=ms_x.dtype)

    with pytest.raises(RuntimeError):
        IndexFillNet()(ms_x, ms_dim, ms_index, ms_value)
