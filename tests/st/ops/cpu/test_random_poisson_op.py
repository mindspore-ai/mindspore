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
import mindspore as ms
from mindspore import nn
from mindspore.ops.function import random_func as R
from mindspore.ops import operations as P
from mindspore import Tensor
from mindspore.common.api import _pynative_executor


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("dtype", [ms.float64, ms.float32, ms.float16, ms.int64, ms.int32])
@pytest.mark.parametrize("shape_dtype", [ms.int64, ms.int32])
@pytest.mark.parametrize("rate_dtype", [ms.float64, ms.float32, ms.float16, ms.int64, ms.int32])
def test_poisson_function(dtype, shape_dtype, rate_dtype):
    """
    Feature: Poisson functional interface
    Description: Test output shape of the poisson functional interface.
    Expectation: Output shape is correct.
    """

    # rate is a scalar Tensor
    shape = Tensor(np.array([3, 5]), shape_dtype)
    rate = Tensor(0.5, rate_dtype)
    output = R.random_poisson(shape, rate, seed=1, dtype=dtype)
    assert output.shape == (3, 5)
    assert output.dtype == dtype

    # rate is a 2-D Tensor
    shape = Tensor(np.array([3, 2]), shape_dtype)
    rate = Tensor(np.array([[5.0, 10.0], [5.0, 1.0]]), rate_dtype)
    output = R.random_poisson(shape, rate, seed=5, dtype=dtype)
    assert output.shape == (3, 2, 2, 2)
    assert output.dtype == dtype


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
def test_poisson_function_shape_type_error():
    """
    Feature: Poisson functional interface
    Description: Feed tuple type `shape` into poisson functional interface.
    Expectation: Except TypeError.
    """

    shape = (3, 5)
    rate = Tensor(np.array([0.5]), ms.dtype.float32)
    try:
        R.random_poisson(shape, rate, seed=1)
        _pynative_executor.sync()
    except TypeError:
        return
    assert False


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
def test_poisson_function_shape_dim_error():
    """
    Feature: Poisson functional interface
    Description: Feed 2-D Tensor type `shape` into poisson functional interface.
    Expectation: Except TypeError.
    """

    shape = Tensor(np.array([[1, 2], [3, 5]]), ms.dtype.int32)
    rate = Tensor(np.array([0.5]), ms.dtype.float32)
    try:
        R.random_poisson(shape, rate, seed=1)
        _pynative_executor.sync()
    except ValueError:
        return
    assert False


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
def test_poisson_function_shape_dtype_error():
    """
    Feature: Poisson functional interface
    Description: Feed Tensor[Float32] type `shape` into poisson functional interface.
    Expectation: Except TypeError.
    """

    shape = Tensor(np.array([3, 5]), ms.dtype.float32)
    rate = Tensor(np.array([0.5]), ms.dtype.float32)
    try:
        R.random_poisson(shape, rate, seed=1)
        _pynative_executor.sync()
    except TypeError:
        return
    assert False


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
def test_poisson_function_shape_value_error():
    """
    Feature: Poisson functional interface
    Description: Feed Tensor[Float32] type `shape` with negative element into poisson functional interface.
    Expectation: Except ValueError.
    """

    shape = Tensor(np.array([-3, 5]), ms.dtype.int32)
    rate = Tensor(np.array([0.5]), ms.dtype.float32)
    try:
        R.random_poisson(shape, rate, seed=1)
        _pynative_executor.sync()
    except ValueError:
        return
    assert False


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
def test_poisson_function_rate_type_error():
    """
    Feature: Poisson functional interface
    Description: Feed list type `rate` into poisson functional interface.
    Expectation: Except TypeError.
    """

    shape = Tensor(np.array([3, 5]), ms.dtype.int32)
    rate = [0.5]
    try:
        R.random_poisson(shape, rate, seed=1)
        _pynative_executor.sync()
    except TypeError:
        return
    assert False


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
def test_poisson_function_rate_dtype_error():
    """
    Feature: Poisson functional interface
    Description: Feed Tensor[Bool] type `rate` into poisson functional interface.
    Expectation: Except TypeError.
    """

    shape = Tensor(np.array([3, 5]), ms.dtype.int32)
    rate = Tensor(np.array([0.5]), ms.dtype.bool_)
    try:
        R.random_poisson(shape, rate, seed=1)
        _pynative_executor.sync()
    except TypeError:
        return
    assert False


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
def test_poisson_function_seed_type_error():
    """
    Feature: Poisson functional interface
    Description: Feed float type `seed` into poisson functional interface.
    Expectation: Except TypeError.
    """

    shape = Tensor(np.array([3, 5]), ms.dtype.int32)
    rate = Tensor(np.array([0.5]), ms.dtype.float32)
    try:
        R.random_poisson(shape, rate, seed=0.5)
        _pynative_executor.sync()
    except TypeError:
        return
    assert False


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
def test_poisson_function_out_dtype_error():
    """
    Feature: Poisson functional interface
    Description: Feed Bool as `dtype` into poisson functional interface.
    Expectation: Except TypeError.
    """

    shape = Tensor(np.array([3, 5]), ms.dtype.int32)
    rate = Tensor(np.array([0.5]), ms.dtype.float32)
    try:
        R.random_poisson(shape, rate, seed=1, dtype=ms.dtype.bool_)
        _pynative_executor.sync()
    except TypeError:
        return
    assert False


class PoissonNet(nn.Cell):
    """ Network for test dynamic shape feature of poisson functional op. """
    def __init__(self, out_dtype, is_rate_scalar=False):
        super().__init__()
        self.odtype = out_dtype
        self.unique = P.Unique()
        self.gather = P.Gather()
        self.is_rate_scalar = is_rate_scalar

    def construct(self, x, y, indices):
        shape, _ = self.unique(x)
        rate = y
        idx, _ = self.unique(indices)
        if not self.is_rate_scalar:
            rate = self.gather(rate, idx, 0)
        return R.random_poisson(shape, rate, seed=1, dtype=self.odtype)


class PoissonDSFactory:
    """ Factory class for test dynamic shape feature of poisson functional op. """
    def __init__(self, max_dims, rate_dims):
        self.rate_random_range = 8
        self.odtype = ms.float32
        self.shape_dtype = ms.int32
        self.rate_dtype = ms.float32
        self.is_rate_scalar = rate_dims == 0
        # shape tensor is a 1-D tensor, uniqueed from shape_map.
        self.shape_map = np.random.randint(1, max_dims, 30, dtype=np.int32)

        if self.is_rate_scalar:
            self.rate_map = np.abs(np.random.randn(1))[0]
        else:
            # rate_shape: [rate_random_range, xx, ..., xx], rank of rate_shape = rate_dims
            rate_map_shape = np.random.randint(1, max_dims, rate_dims - 1, dtype=np.int32)
            rate_map_shape = np.append(np.array([self.rate_random_range]), rate_map_shape, axis=0)
            # rate tensor will be gathered from rate_map.
            self.rate_map = np.abs(np.random.randn(*rate_map_shape))
        # indices array is used to gather rate_map to rate tensor.
        indices_shape = np.random.randint(1, self.rate_random_range, 1, dtype=np.int32)[0]
        self.indices = np.random.randint(1, self.rate_random_range, indices_shape, dtype=np.int32)

    @staticmethod
    def _np_unranked_unique(nparr):
        """ Get unique elements in `nparr` without ranked. """
        n_unique = len(np.unique(nparr))
        ranked_unique = np.zeros([n_unique], dtype=nparr.dtype)
        i = 0
        for x in nparr:
            if x not in ranked_unique:
                ranked_unique[i] = x
                i += 1
        return ranked_unique

    def forward_compare(self):
        """ Compare result of mindspore and numpy """
        ms_shape, ms_dtype = self._forward_mindspore()
        np_shape = self._forward_numpy()
        assert len(ms_shape) == len(np_shape)
        for index, dim in enumerate(np_shape):
            assert ms_shape[index] == dim
        assert ms_dtype == self.odtype

    def _forward_numpy(self):
        """ Get result of numpy """
        shape = PoissonDSFactory._np_unranked_unique(self.shape_map)
        if self.is_rate_scalar:
            return shape
        indices = PoissonDSFactory._np_unranked_unique(self.indices)
        rate = self.rate_map[indices]
        rate_shape = rate.shape
        out_shape = np.append(shape, rate_shape, axis=0)
        return out_shape

    def _forward_mindspore(self):
        """ Get result of mindspore """
        shape_map_tensor = Tensor(self.shape_map, dtype=self.shape_dtype)
        rate_tensor = Tensor(self.rate_map, dtype=self.rate_dtype)
        indices_tensor = Tensor(self.indices, dtype=ms.dtype.int32)
        net = PoissonNet(self.odtype, self.is_rate_scalar)
        output = net(shape_map_tensor, rate_tensor, indices_tensor)
        return output.shape, output.dtype


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("max_dims", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("rate_dims", [0, 1, 2, 3, 4, 5, 6])
def test_poisson_function_dynamic_shape(max_dims, rate_dims):
    """
    Feature: Dynamic shape of functional interface RandomPoisson.
    Description:
      1. Initialize a 1-D Tensor as input `shape` whose data type fixed to int32, whose data and shape are random.
      2. Initialize a Tensor as input `rate` whose data type fixed to float32, whose data and shape are random.
      3. Compare shape of output from MindSpore and Numpy.
    Expectation: Output of MindSpore RandomPoisson equal to numpy.
    """

    factory = PoissonDSFactory(max_dims, rate_dims)
    factory.forward_compare()
