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
from mindspore.ops import PrimitiveWithInfer, prim_attr_register


class IndexFill(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x', 'dim', 'index', 'value'], outputs=['y'])

    def __infer__(self, x, dim, index, value):
        return {
            "shape": x["shape"],
            "dtype": x["dtype"],
            "value": None
        }


class IndexFillNet(nn.Cell):
    def __init__(self):
        super(IndexFillNet, self).__init__()
        self.index_fill = IndexFill()

    def construct(self, x, dim, index, value):
        out = self.index_fill(x, dim, index, value)
        return out


def compare_with_numpy(x, dim, index, value):
    # Graph Mode
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    ms_x = Tensor(x)
    ms_index = Tensor(index)
    ms_dim = Tensor(dim, dtype=mstype.int32)
    ms_value = Tensor(value, dtype=ms_x.dtype)
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
@pytest.mark.parametrize('dim', [0, 1, 2])
@pytest.mark.parametrize('data_type', [np.float16, np.float32, np.float64])
def test_index_fill_float(dim, data_type):
    """
    Feature: IndexFill
    Description:  test cases for IndexFill operator with float.
    Expectation: the result match numpy.
    """
    fill_value = -50.
    x_np = np.random.random(size=(5, 5, 5)).astype(data_type)
    index_np = np.random.randint(low=0, high=5, size=4).astype(np.int32)
    assert compare_with_numpy(x_np, dim, index_np, fill_value)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dim', [0, 1, 2])
@pytest.mark.parametrize('data_type', [np.int32, np.int64])
def test_index_fill_int(dim, data_type):
    """
    Feature: IndexFill
    Description:  test cases for IndexFill operator with int.
    Expectation: the result match numpy.
    """
    fill_value = 20
    x_np = np.random.randint(20, size=(5, 5, 5)).astype(data_type)
    index_np = np.random.randint(low=0, high=5, size=4).astype(np.int32)
    assert compare_with_numpy(x_np, dim, index_np, fill_value)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dim', [0, 1])
@pytest.mark.parametrize('data_type', [np.int32])
def test_index_fill_error(dim, data_type):
    """
    Feature: IndexFill
    Description:  test cases for IndexFill operator with int.
    Expectation: raise RuntimeError.
    """
    ms_x = Tensor([[1, 2], [3, 4]]).astype(data_type)
    ms_index = Tensor([2]).astype(data_type)
    ms_dim = Tensor(dim, dtype=mstype.int32)
    ms_value = Tensor(20, dtype=ms_x.dtype)

    with pytest.raises(RuntimeError):
        IndexFillNet()(ms_x, ms_dim, ms_index, ms_value)
