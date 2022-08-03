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

    shape = Tensor(np.array([3, 5]), shape_dtype)
    rate = Tensor(np.array([0.5]), rate_dtype)
    output = R.random_poisson(shape, rate, seed=1, dtype=dtype)
    assert output.shape == (3, 5, 1)
    assert output.dtype == dtype

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
    except TypeError:
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
    except TypeError:
        return
    assert False


class PoissonNet(nn.Cell):
    """ Network for test dynamic shape feature of poisson functional op. """
    def __init__(self, out_dtype, axis=0):
        super().__init__()
        self.odtype = out_dtype
        self.unique = P.Unique()
        self.gather = P.Gather()
        self.axis = axis

    def construct(self, x, y, indices):
        shape, _ = self.unique(x)
        idx, _ = self.unique(indices)
        rate = self.gather(y, idx, self.axis)
        return R.random_poisson(shape, rate, seed=1, dtype=self.odtype)


class PoissonDSFactory:
    """ Factory class for test dynamic shape feature of poisson functional op. """
    def __init__(self, rate_dims, out_dtype, shape_dtype, rate_dtype):
        self.rate_dims = rate_dims
        self.rate_random_range = 8
        self.odtype = out_dtype
        self.shape_dtype = shape_dtype
        self.rate_dtype = rate_dtype
        # shape tensor is a 1-D tensor, unique from shape_map.
        self.shape_map = np.random.randint(1, 6, 30, dtype=np.int32)
        # rate_map will be gathered as rate tensor.
        rate_map_shape = np.random.randint(1, 6, self.rate_dims - 1, dtype=np.int32)
        rate_map_shape = np.append(np.array([self.rate_random_range]), rate_map_shape, axis=0)
        self.rate_map = np.random.randn(*rate_map_shape)
        # indices array is used to gather rate_map to rate tensor.
        self.indices = np.random.randint(1, self.rate_random_range, 4, dtype=np.int32)

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
        indices = PoissonDSFactory._np_unranked_unique(self.indices)
        rate = self.rate_map[indices]
        rate_shape = rate.shape
        out_shape = np.append(shape, rate_shape, axis=0)
        return out_shape

    def _forward_mindspore(self):
        """ Get result of mindspore """
        shape_map_tensor = Tensor(self.shape_map, dtype=self.shape_dtype)
        rate_map_tensor = Tensor(self.rate_map, dtype=self.rate_dtype)
        indices_tensor = Tensor(self.indices, dtype=ms.dtype.int32)
        net = PoissonNet(self.odtype)
        output = net(shape_map_tensor, rate_map_tensor, indices_tensor)
        return output.shape, output.dtype


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("rate_dims", [1, 2, 3, 4, 5, 6])
def test_poisson_function_dynamic_shape(rate_dims):
    """
    Feature: Poisson functional interface
    Description: Test dynamic shape feature of the poisson functional interface with 1D-8D rate.
    Expectation: Output of mindspore poisson equal to numpy.
    """

    factory = PoissonDSFactory(rate_dims, ms.float32, ms.int32, ms.float32)
    factory.forward_compare()
