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
""" test nn.Dense """
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Tensor


# pylint: disable=E1123


def test_dense_defaultbias_noactivation():
    weight = Tensor(np.array([[0.1, 0.3, 0.4], [0.1, 0.3, 0.4]], dtype=np.float32))
    dense = nn.Dense(3, 2, weight)
    assert dense.activation is None

    input_data = Tensor(np.random.randint(0, 255, [1, 3]).astype(np.float32))
    output = dense(input_data)
    output_np = output.asnumpy()
    assert isinstance(output_np[0][0], (np.float32, np.float64))


def test_dense_defaultweight():
    bias = Tensor(np.array([0.5, 0.3], dtype=np.float32))
    dense = nn.Dense(3, 2, bias_init=bias)
    # batch_size 1 && 3-channel RGB
    input_data = Tensor(np.random.randint(0, 255, [1, 3]).astype(np.float32))
    output = dense(input_data)
    output_np = output.asnumpy()
    assert isinstance(output_np[0][0], (np.float32, np.float64))


def test_dense_bias():
    weight = Tensor(np.array([[0.1, 0.3, 0.6], [0.4, 0.5, 0.2]], dtype=np.float32))
    bias = Tensor(np.array([0.5, 0.3], dtype=np.float32))
    dense = nn.Dense(3, 2, weight, bias)

    input_data = Tensor(np.random.randint(0, 255, [2, 3]).astype(np.float32))
    output = dense(input_data)
    output_np = output.asnumpy()
    assert isinstance(output_np[0][0], (np.float32, np.float64))


def test_dense_nobias():
    weight = Tensor(np.array([[0.1, 0.3, 0.6], [0.4, 0.5, 0.2]], dtype=np.float32))
    dense = nn.Dense(3, 2, weight, has_bias=False)

    input_data = Tensor(np.random.randint(0, 255, [2, 3]).astype(np.float32))
    output = dense(input_data)
    output_np = output.asnumpy()
    assert isinstance(output_np[0][0], (np.float32, np.float64))


def test_dense_none():
    dense = nn.Dense(3, 4, None, None)
    input_data = Tensor(np.random.randint(0, 255, [2, 3]).astype(np.float32))
    dense(input_data)


def test_dense_str_activation():
    dense = nn.Dense(1, 1, activation='relu')
    assert isinstance(dense.activation, nn.ReLU)

    input_data = Tensor(np.random.randint(0, 255, [1, 1]).astype(np.float32))
    output = dense(input_data)
    output_np = output.asnumpy()
    assert isinstance(output_np[0][0], np.float32)


def test_dense_weight_error():
    dim_error = Tensor(np.array([[[0.1], [0.3], [0.6]], [[0.4], [0.5], [0.2]]]))
    with pytest.raises(ValueError):
        nn.Dense(3, 2, dim_error)

    shape_error = Tensor(np.array([[0.1, 0.3, 0.6], [0.4, 0.5, 0.2]]))
    with pytest.raises(ValueError):
        nn.Dense(2, 2, shape_error)
    with pytest.raises(ValueError):
        nn.Dense(3, 3, shape_error)


def test_dense_bias_error():
    dim_error = Tensor(np.array([[0.5, 0.3]]))
    with pytest.raises(ValueError):
        nn.Dense(3, 2, bias_init=dim_error)

    shape_error = Tensor(np.array([0.5, 0.3, 0.4]))
    with pytest.raises(ValueError):
        nn.Dense(3, 2, bias_init=shape_error)


def test_dense_dtype_error():
    with pytest.raises(TypeError):
        nn.Dense(3, 2, dtype=3)


def test_dense_channels_error():
    with pytest.raises(ValueError):
        nn.Dense(3, 0)

    with pytest.raises(ValueError):
        nn.Dense(-1, 2)
