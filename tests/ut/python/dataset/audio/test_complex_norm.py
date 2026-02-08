# Copyright 2021-2025 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""Test ComplexNorm."""

import numpy as np
from numpy import random

import mindspore.dataset as ds
from mindspore.dataset import audio


def test_complex_norm():
    """
    Feature: ComplexNorm
    Description: Test ComplexNorm in pipeline mode
    Expectation: Output is equal to the expected value
    """
    def gen():
        data = np.array([[1.0, 1.0], [2.0, 3.0], [4.0, 4.0]])
        yield (np.array(data, dtype=np.float32),)

    dataset = ds.GeneratorDataset(source=gen, column_names=["multi_dim_data"])

    dataset = dataset.map(operations=audio.ComplexNorm(2),
                          input_columns=["multi_dim_data"])

    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert i["multi_dim_data"].shape == (3,)
        expected = np.array([2., 13., 32.])
        assert np.array_equal(i["multi_dim_data"], expected)


def test_complex_norm_eager():
    """
    Feature: ComplexNorm
    Description: Test ComplexNorm in eager mode
    Expectation: Output is equal to the expected value
    """

    input_t = np.array([[1.0, 1.0], [2.0, 3.0], [4.0, 4.0]])
    output_t = audio.ComplexNorm()(input_t)
    assert output_t.shape == (3,)
    expected = np.array(
        [1.4142135623730951, 3.605551275463989, 5.656854249492381])
    assert np.array_equal(output_t, expected)


def test_complex_norm_uncallable():
    """
    Feature: ComplexNorm
    Description: Test ComplexNorm that is not callable due to invalid input power
    Expectation: Error is raised as expected
    """
    try:
        input_t = random.rand(2, 4, 3, 2)
        output_t = audio.ComplexNorm(-3.)(input_t)
        assert output_t.shape == (2, 4, 3)
    except ValueError as e:
        assert 'Input power is not within the required interval of [0, 16777216].' in str(
            e)


if __name__ == "__main__":
    test_complex_norm()
    test_complex_norm_eager()
    test_complex_norm_uncallable()
