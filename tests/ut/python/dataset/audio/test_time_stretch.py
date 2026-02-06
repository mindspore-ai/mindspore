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
"""Test TimeStretch."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from . import count_unequal_element

CHANNEL_NUM = 2
FREQ = 1025
FRAME_NUM = 300
COMPLEX = 2


def gen(shape):
    np.random.seed(0)
    data = np.random.random(shape)
    yield (np.array(data, dtype=np.float32),)


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan):
        count_unequal_element(data_expected, data_me, rtol, atol)


def test_time_stretch_pipeline():
    """
    Feature: TimeStretch
    Description: Test TimeStretch in pipeline mode
    Expectation: Output's shape is the same as expected output's shape
    """
    generator = gen([CHANNEL_NUM, FREQ, FRAME_NUM, COMPLEX])
    data1 = ds.GeneratorDataset(
        source=generator, column_names=["multi_dimensional_data"]
    )

    transforms = [audio.TimeStretch(512, FREQ, 1.3)]
    data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])

    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        output = item["multi_dimensional_data"]
    assert output.shape == (CHANNEL_NUM, FREQ, np.ceil(FRAME_NUM / 1.3), COMPLEX)


def test_time_stretch_pipeline_invalid_param():
    """
    Feature: TimeStretch
    Description: Test TimeStretch in pipeline mode with invalid parameter
    Expectation: Error is raised as expected
    """
    generator = gen([CHANNEL_NUM, FREQ, FRAME_NUM, COMPLEX])
    data1 = ds.GeneratorDataset(
        source=generator, column_names=["multi_dimensional_data"]
    )

    with pytest.raises(
        ValueError,
        match=r"Input fixed_rate is not within the required interval of \(0, 16777216\].",
    ):
        transforms = [audio.TimeStretch(512, FREQ, -1.3)]
        data1 = data1.map(
            operations=transforms, input_columns=["multi_dimensional_data"]
        )

        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            output = item["multi_dimensional_data"]
        assert output.shape == (CHANNEL_NUM, FREQ, np.ceil(FRAME_NUM / 1.3), COMPLEX)


def test_time_stretch_eager():
    """
    Feature: TimeStretch
    Description: Test TimeStretch in eager mode with customized parameter values
    Expectation: Output's shape is the same as expected output's shape
    """
    spectrogram = next(gen([CHANNEL_NUM, FREQ, FRAME_NUM, COMPLEX]))[0]
    output = audio.TimeStretch(512, FREQ, 1.3)(spectrogram)
    assert output.shape == (CHANNEL_NUM, FREQ, np.ceil(FRAME_NUM / 1.3), COMPLEX)


def test_percision_time_stretch_eager():
    """
    Feature: TimeStretch
    Description: Test TimeStretch in eager mode by comparing precision
    Expectation: Output is the same as expected output
    """
    spectrogram = np.array(
        [
            [
                [
                    [1.0402449369430542, 0.3807601034641266],
                    [-1.120057225227356, -0.12819576263427734],
                    [1.4303032159805298, -0.08839055150747299],
                ],
                [
                    [1.4198592901229858, 0.6900091767311096],
                    [-1.8593409061431885, 0.16363371908664703],
                    [-2.3349387645721436, -1.4366451501846313],
                ],
            ],
            [
                [
                    [-0.7083967328071594, 0.9325454831123352],
                    [-1.9133838415145874, 0.011225821450352669],
                    [1.477278232574463, -1.0551637411117554],
                ],
                [
                    [-0.6668586134910583, -0.23143270611763],
                    [-2.4390718936920166, 0.17638640105724335],
                    [-0.4795735776424408, 0.1345423310995102],
                ],
            ],
        ]
    ).astype(np.float64)
    out_expect = np.array(
        [
            [
                [
                    [1.0402449369430542, 0.3807601034641266],
                    [-1.302264928817749, -0.1490504890680313],
                ],
                [
                    [1.4198592901229858, 0.6900091767311096],
                    [-2.382312774658203, 0.2096325159072876],
                ],
            ],
            [
                [
                    [-0.7083966732025146, 0.9325454831123352],
                    [-1.8545820713043213, 0.010880803689360619],
                ],
                [
                    [-0.6668586134910583, -0.23143276572227478],
                    [-1.2737033367156982, 0.09211209416389465],
                ],
            ],
        ]
    ).astype(np.float64)
    output = audio.TimeStretch(64, 2, 1.6)(spectrogram)

    allclose_nparray(output, out_expect, 0.001, 0.001)


def test_time_stretch_transform():
    """
    Feature: TimeStretch
    Description: Test TimeStretch with various valid input parameters and data types
    Expectation: The operation completes successfully
    """

    spectrum = np.random.randn(64, 40, 2)
    time_stretch = audio.TimeStretch(hop_length=512, n_freq=64, fixed_rate=1.3)
    spec_out = time_stretch(spectrum)
    assert spec_out.shape == (64, 31, 2)

    spectrum = np.random.randn(64, 40, 20, 2)
    time_stretch = audio.TimeStretch(hop_length=512, n_freq=64, fixed_rate=0.3)
    spec_out = time_stretch(spectrum)
    assert spec_out.shape == (64, 40, 67, 2)

    # Test eager,data type
    spectrum = np.random.randn(64, 40, 2).astype(np.float16)
    time_stretch = audio.TimeStretch(hop_length=512, n_freq=64, fixed_rate=0.3)
    spec_out = time_stretch(spectrum)
    assert spec_out.shape == (64, 134, 2)

    spectrum = np.random.randn(64, 40, 2).astype(np.float32)
    time_stretch = audio.TimeStretch(hop_length=512, n_freq=64, fixed_rate=0.3)
    spec_out = time_stretch(spectrum)
    assert spec_out.shape == (64, 134, 2)

    spectrum = np.random.randn(64, 40, 2).astype(np.float64)
    time_stretch = audio.TimeStretch(hop_length=512, n_freq=64, fixed_rate=0.3)
    spec_out = time_stretch(spectrum)
    assert spec_out.shape == (64, 134, 2)

    spectrum = np.random.randn(64, 40, 2).astype(np.uint8)
    time_stretch = audio.TimeStretch(hop_length=512, n_freq=64, fixed_rate=0.3)
    spec_out = time_stretch(spectrum)
    assert spec_out.shape == (64, 134, 2)

    spectrum = np.random.randn(64, 40, 2).astype(np.int32)
    time_stretch = audio.TimeStretch(hop_length=512, n_freq=64, fixed_rate=0.3)
    spec_out = time_stretch(spectrum)
    assert spec_out.shape == (64, 134, 2)

    # Test eager,parameter type
    spectrum = np.random.randn(201, 40, 2).astype(np.float32)
    time_stretch = audio.TimeStretch(hop_length=None, n_freq=201, fixed_rate=1.3)
    out_default_implicit = time_stretch(spectrum)
    assert out_default_implicit.shape == (201, 31, 2)

    # check default param value for n_freq
    time_stretch = audio.TimeStretch(hop_length=200, n_freq=201, fixed_rate=None)
    out_defalut_explicit = time_stretch(spectrum)
    assert out_defalut_explicit.shape == (1, 201, 40, 2)

    # test time stretch normal pipeline

    data = np.random.random([10, 64, 40, 2]).astype(np.float32)
    dataset = ds.NumpySlicesDataset(data=data, column_names=["multi_dimensional_data"])
    transforms = audio.TimeStretch(512, 64, 0.3)

    dataset = dataset.map(
        input_columns=["multi_dimensional_data"], operations=transforms
    )
    for sample in dataset.create_dict_iterator(output_numpy=True):
        image_aug = sample["multi_dimensional_data"]
        assert image_aug.shape == (64, 134, 2)

    # test time stretch pipeline

    data = np.random.random([10, 20, 30, 2]).astype(np.float32)
    dataset = ds.NumpySlicesDataset(data=data, column_names=["multi_dimensional_data"])
    transforms = audio.TimeStretch(100, 20, 0.5)
    dataset = dataset.map(
        input_columns=["multi_dimensional_data"], operations=transforms
    )
    for sample in dataset.create_dict_iterator(output_numpy=True):
        image = sample["multi_dimensional_data"]
        assert image.shape == (20, 60, 2)

    # diff shape
    data = np.random.random([10, 20, 30, 10, 2]).astype(np.float32)
    dataset = ds.NumpySlicesDataset(data=data, column_names=["multi_dimensional_data"])
    transforms = audio.TimeStretch(100, 30, 1.2)
    dataset = dataset.map(
        input_columns=["multi_dimensional_data"], operations=transforms
    )
    for sample in dataset.create_dict_iterator(output_numpy=True):
        image = sample["multi_dimensional_data"]
        assert image.shape == (20, 30, 9, 2)

    # diff shape
    data = np.random.random([10, 10, 20, 30, 40, 2]).astype(np.float32)
    dataset = ds.NumpySlicesDataset(data=data, column_names=["multi_dimensional_data"])
    transforms = audio.TimeStretch(100, 30, 1.5)
    dataset = dataset.map(
        input_columns=["multi_dimensional_data"], operations=transforms
    )
    for sample in dataset.create_dict_iterator(output_numpy=True):
        image = sample["multi_dimensional_data"]
        assert image.shape == (10, 20, 30, 27, 2)

    # diff dtype
    data = np.random.random([10, 20, 30, 2]).astype(np.float64)
    dataset = ds.NumpySlicesDataset(data=data, column_names=["multi_dimensional_data"])
    transforms = audio.TimeStretch(100, 20, 0.5)
    dataset = dataset.map(
        input_columns=["multi_dimensional_data"], operations=transforms
    )
    for sample in dataset.create_dict_iterator(output_numpy=True):
        image = sample["multi_dimensional_data"]
        assert image.shape == (20, 60, 2)

    # diff dtype
    data = np.random.random([10, 20, 30, 2]).astype(np.float16)
    dataset = ds.NumpySlicesDataset(data=data, column_names=["multi_dimensional_data"])
    transforms = audio.TimeStretch(100, 20, 0.5)
    dataset = dataset.map(
        input_columns=["multi_dimensional_data"], operations=transforms
    )
    for sample in dataset.create_dict_iterator(output_numpy=True):
        image = sample["multi_dimensional_data"]
        assert image.shape == (20, 60, 2)

    # diff dtype
    data = np.random.random([10, 20, 30, 2]).astype(np.int_)
    dataset = ds.NumpySlicesDataset(data=data, column_names=["multi_dimensional_data"])
    transforms = audio.TimeStretch(100, 20, 0.5)
    dataset = dataset.map(
        input_columns=["multi_dimensional_data"], operations=transforms
    )
    for sample in dataset.create_dict_iterator(output_numpy=True):
        image = sample["multi_dimensional_data"]
        assert image.shape == (20, 60, 2)


def test_time_stretch_param_check():
    """
    Feature: TimeStretch
    Description: Test TimeStretch with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """

    spectrum = np.random.randn(64, 40, 3)
    time_stretch = audio.TimeStretch(hop_length=512, n_freq=64, fixed_rate=1.3)
    with pytest.raises(
        RuntimeError,
        match="TimeStretch: the shape of input tensor does not match the requirement of "
        "operator. Expecting tensor in shape of <..., freq, num_frame, complex=2>.",
    ):
        time_stretch(spectrum)
    spectrum = np.random.randn(64, 40)
    time_stretch = audio.TimeStretch(hop_length=512, n_freq=64, fixed_rate=1.3)
    with pytest.raises(
        RuntimeError,
        match="TimeStretch: the shape of input tensor does not match the requirement of "
        "operator. Expecting tensor in shape of <..., freq, num_frame, complex=2>.",
    ):
        time_stretch(spectrum)
    spectrum = np.random.randn(2)
    time_stretch = audio.TimeStretch(hop_length=100, n_freq=64, fixed_rate=0.3)
    with pytest.raises(
        RuntimeError,
        match="TimeStretch: the shape of input tensor does not match the requirement of "
        "operator. Expecting tensor in shape of <..., freq, num_frame, complex=2>.",
    ):
        time_stretch(spectrum)

    # Test eager,data type
    spectrum = np.random.randn(64, 40, 2).tolist()
    time_stretch = audio.TimeStretch(hop_length=512, n_freq=64, fixed_rate=0.3)
    with pytest.raises(
        TypeError, match="Input should be NumPy audio, got <class 'list'>."
    ):
        time_stretch(spectrum)

    def invalid_param(hop_length, n_freq, fixed_rate, error, error_msg):
        """Test invalid parameters."""
        with pytest.raises(error) as error_info:
            audio.TimeStretch(hop_length, n_freq, fixed_rate)
        assert error_msg in str(error_info.value)

    # Test eager,parameter type
    # Check range of input param,invalid hop_length
    invalid_param(
        -100,
        200,
        1.2,
        ValueError,
        "Input hop_length is not within the required interval of [1, 2147483647].",
    )
    # Check range of input param,invalid n_freq
    invalid_param(
        100,
        -100,
        2,
        ValueError,
        "Input n_freq is not within the required interval of [1, 2147483647].",
    )
    # Check range of input param,invalid fix_rate
    invalid_param(
        100,
        100,
        -2,
        ValueError,
        "Input fixed_rate is not within the required interval of (0, 16777216].",
    )

    # Check input param data type.invalid hop_length
    invalid_param(
        "True",
        200,
        1.2,
        TypeError,
        "Argument hop_length with value True is not of type [<class 'int'>], but got <class 'str'>.",
    )
    # Check input param data type.invalid n_freq
    invalid_param(
        100,
        "2",
        10,
        TypeError,
        "Argument n_freq with value 2 is not of type [<class 'int'>], but got <class 'str'>.",
    )
    # Check input param data type.invalid fix_rate
    invalid_param(
        100,
        100,
        "10",
        TypeError,
        "Argument fixed_rate with value 10 "
        "is not of type [<class 'int'>, <class 'float'>], but got <class 'str'>.",
    )


if __name__ == "__main__":
    test_time_stretch_pipeline()
    test_time_stretch_pipeline_invalid_param()
    test_time_stretch_eager()
    test_percision_time_stretch_eager()
    test_time_stretch_transform()
    test_time_stretch_param_check()
