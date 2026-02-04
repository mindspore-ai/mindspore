# Copyright 2022-2025 Huawei Technologies Co., Ltd
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
"""Test Filtfilt."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from . import count_unequal_element


def test_filtfilt_eager():
    """
    Feature: Filtfilt
    Description: Test Filtfilt in eager mode under normal test case
    Expectation: Output is equal to the expected output
    """
    # construct input
    waveform = np.array([[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]], dtype=np.float64)
    a_coeffs = [0.1, 0.2, 0.3]
    b_coeffs = [0.4, 0.5, 0.6]
    clanmp_input = True

    filtfilt = audio.Filtfilt(a_coeffs, b_coeffs, clamp=clanmp_input)
    our_waveform = filtfilt(waveform)

    # use np flip
    forward_filtered = audio.LFilter(a_coeffs, b_coeffs, clamp=False)
    backward_filtered = audio.LFilter(a_coeffs, b_coeffs, clamp=clanmp_input)

    # use np flip
    forward_filtered_waveform = forward_filtered(waveform)
    backward_filtered_waveform = backward_filtered(
        np.flip(forward_filtered_waveform, -1)
    )
    expect_waveform = np.flip(backward_filtered_waveform, -1)
    count_unequal_element(our_waveform, expect_waveform, 0.0001, 0.0001)


def test_filtfilt_pipeline():
    """
    Feature: Filtfilt
    Description: Test Filtfilt in pipeline mode under normal test case
    Expectation: Output is equal to the expected output
    """
    # construct input
    waveform = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.5, 0.6, 0.7]], dtype=np.float64)
    a_coeffs = [0.1, 0.2, 0.3]
    b_coeffs = [0.4, 0.5, 0.6]

    expect_waveform = np.array([[1, 0.2, -1, 1], [1, 0.5, -1, 1]], dtype=np.float64)

    data = (waveform, waveform.shape)
    dataset = ds.NumpySlicesDataset(data, ["channel", "sample"], shuffle=False)
    filtfilt = audio.Filtfilt(a_coeffs, b_coeffs, clamp=True)
    # Filtered waveform by lfilter
    dataset = dataset.map(
        input_columns=["channel"], operations=filtfilt, num_parallel_workers=8
    )
    i = 0
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(expect_waveform[i, :], data["channel"], 0.0001, 0.0001)
        i += 1


def test_filtfilt_invalid_input_all():
    """
    Feature: Filtfilt
    Description: Test Filtfilt with invalid input
    Expectation: Correct error is raised as expected
    """
    waveform = np.random.rand(2, 1000)

    def test_invalid_input(a_coeffs, b_coeffs, clamp, error, error_msg):
        with pytest.raises(error) as error_info:
            audio.Filtfilt(a_coeffs, b_coeffs, clamp)(waveform)
        assert error_msg in str(error_info.value)

    test_invalid_input(
        ["0.1", "0.2", "0.3"],
        [0.1, 0.2, 0.3],
        True,
        TypeError,
        "Argument a_coeffs[0] with value 0.1 is not of type [<class 'float'>, <class 'int'>], but got <class 'str'>.",
    )
    test_invalid_input(
        [234322354352353453651, 0.2, 0.3],
        [0.1, 0.2, 0.3],
        True,
        ValueError,
        "Input a_coeffs[0] is not within the required interval of [-16777216, 16777216].",
    )
    test_invalid_input(
        [0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3],
        "True",
        TypeError,
        "Argument clamp with value True is not of type [<class 'bool'>], but got <class 'str'>.",
    )


def test_filtfilt_transform():
    """
    Feature: Filtfilt
    Description: Test Filtfilt with various valid input parameters and data types
    Expectation: The operation completes successfully and output values match expectations
    """
    # Test eager filtfilt float64
    # Original waveform
    waveform = np.array(
        [
            [0.823655, 0.204918, 0.333587],
            [0.593376, 0.991154, 0.248223],
            [0.300798, 0.905465, 0.759823],
            [0.539423, 0.284265, 0.563423],
            [0.636365, 0.222653, 0.228853],
        ],
        dtype=np.float64,
    )
    # Expect waveform
    expect_waveform = np.array(
        [
            [0.77337015, 0.1536079, 0.28125167],
            [0.5254886, 0.92187756, 0.17755732],
            [0.22736466, 0.83051515, 0.6833563],
            [0.48783386, 0.23161554, 0.5097128],
            [0.59621745, 0.18168716, 0.18706888],
        ],
        dtype=np.float32,
    )

    filtfilt = audio.Filtfilt(
        a_coeffs=[1.0201494451137518, -1.9991880801438362, 0.9798505548862483],
        b_coeffs=[0.999797020035959, -1.999594040071918, 0.999797020035959],
    )
    # Filtered waveform by filtfilt
    output = filtfilt(waveform)
    count_unequal_element(expect_waveform, output, 0.000001, 0.000001)

    # test filtfilt normal3
    waveform = np.array(
        [
            [0.8236, 0.2049, 0.3335],
            [0.5933, 0.9911, 0.2482],
            [0.3007, 0.9054, 0.7598],
            [0.5394, 0.2842, 0.5634],
            [0.6363, 0.2226, 0.2288],
        ]
    )
    filtfilt = audio.Filtfilt(a_coeffs=[1.0, 0.1, 0.2], b_coeffs=[0.1, 0.2, 0.3])
    dataset = ds.NumpySlicesDataset(waveform, ["audio"], shuffle=False)
    dataset = dataset.map(input_columns=["audio"], operations=filtfilt)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # test filtfilt normal2
    waveform = np.array(
        [
            [0.8236, 0.2049, 0.3335],
            [0.5933, 0.9911, 0.2482],
            [0.3007, 0.9054, 0.7598],
            [0.5394, 0.2842, 0.5634],
            [0.6363, 0.2226, 0.2288],
        ]
    )
    filtfilt = audio.Filtfilt(a_coeffs=[1.0, 0.1], b_coeffs=[0.1, 0.2])
    dataset = ds.NumpySlicesDataset(waveform, ["audio"], shuffle=False)
    dataset = dataset.map(input_columns=["audio"], operations=filtfilt)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # test filtfilt tuple
    waveform = np.array(
        (
            (0.8236, 0.2049, 0.3335),
            (0.5933, 0.9911, 0.2482),
            (0.3007, 0.9054, 0.7598),
            (0.5394, 0.2842, 0.5634),
            (0.6363, 0.2226, 0.2288),
        )
    )
    filtfilt = audio.Filtfilt(a_coeffs=[1.0, 0.1], b_coeffs=[0.1, 0.2])
    dataset = ds.NumpySlicesDataset(waveform, ["audio"], shuffle=False)
    dataset = dataset.map(input_columns=["audio"], operations=filtfilt)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # test filtfilt Multiple coeffs
    waveform = np.array(
        (
            (0.8236, 0.2049, 0.3335),
            (0.5933, 0.9911, 0.2482),
            (0.3007, 0.9054, 0.7598),
            (0.5394, 0.2842, 0.5634),
            (0.6363, 0.2226, 0.2288),
            (0.8236, 0.2049, 0.3335),
            (0.5933, 0.9911, 0.2482),
            (0.3007, 0.9054, 0.7598),
            (0.5394, 0.2842, 0.5634),
            (0.6363, 0.2226, 0.2288),
            (0.8236, 0.2049, 0.3335),
            (0.5933, 0.9911, 0.2482),
            (0.3007, 0.9054, 0.7598),
            (0.5394, 0.2842, 0.5634),
            (0.6363, 0.2226, 0.2288),
            (0.8236, 0.2049, 0.3335),
            (0.5933, 0.9911, 0.2482),
            (0.3007, 0.9054, 0.7598),
            (0.5394, 0.2842, 0.5634),
            (0.6363, 0.2226, 0.2288),
        )
    )
    filtfilt = audio.Filtfilt(
        a_coeffs=[
            1.0,
            0.1,
            0.3,
            3.0,
            0.5,
            5.0,
            0.6,
            6.0,
            1.0,
            0.1,
            0.3,
            3.0,
            0.5,
            5.0,
            0.6,
            6.0,
        ],
        b_coeffs=[
            0.1,
            0.2,
            0.3,
            3.0,
            0.5,
            5.0,
            0.6,
            6.0,
            0.1,
            0.2,
            0.3,
            3.0,
            0.5,
            5.0,
            0.6,
            6.0,
        ],
    )
    dataset = ds.NumpySlicesDataset(waveform, ["audio"], shuffle=False)
    dataset = dataset.map(input_columns=["audio"], operations=filtfilt)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # test filtfilt one coeffs
    waveform = np.array(
        (
            (0.8236, 0.2049, 0.3335),
            (0.5933, 0.9911, 0.2482),
            (0.3007, 0.9054, 0.7598),
            (0.5394, 0.2842, 0.5634),
            (0.6363, 0.2226, 0.2288),
            (0.8236, 0.2049, 0.3335),
            (0.5933, 0.9911, 0.2482),
            (0.3007, 0.9054, 0.7598),
            (0.5394, 0.2842, 0.5634),
            (0.6363, 0.2226, 0.2288),
            (0.8236, 0.2049, 0.3335),
            (0.5933, 0.9911, 0.2482),
            (0.3007, 0.9054, 0.7598),
            (0.5394, 0.2842, 0.5634),
            (0.6363, 0.2226, 0.2288),
            (0.8236, 0.2049, 0.3335),
            (0.5933, 0.9911, 0.2482),
            (0.3007, 0.9054, 0.7598),
            (0.5394, 0.2842, 0.5634),
            (0.6363, 0.2226, 0.2288),
        )
    )
    filtfilt = audio.Filtfilt(a_coeffs=[1.0], b_coeffs=[0.1])
    dataset = ds.NumpySlicesDataset(waveform, ["audio"], shuffle=False)
    dataset = dataset.map(input_columns=["audio"], operations=filtfilt)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test eager filtfilt float32
    # Original waveform
    waveform = np.array(
        [
            [0.823655, 0.204918, 0.333587],
            [0.593376, 0.991154, 0.248223],
            [0.300798, 0.905465, 0.759823],
            [0.539423, 0.284265, 0.563423],
            [0.636365, 0.222653, 0.228853],
        ],
        dtype=np.float32,
    )
    # Expect waveform
    expect_waveform = np.array(
        [
            [0.77337015, 0.1536079, 0.28125167],
            [0.5254886, 0.92187756, 0.17755732],
            [0.22736466, 0.83051515, 0.6833563],
            [0.48783386, 0.23161554, 0.5097128],
            [0.59621745, 0.18168716, 0.18706888],
        ],
        dtype=np.float32,
    )

    filtfilt = audio.Filtfilt(
        a_coeffs=[1.0201494451137518, -1.9991880801438362, 0.9798505548862483],
        b_coeffs=[0.999797020035959, -1.999594040071918, 0.999797020035959],
    )
    # Filtered waveform by filtfilt
    output = filtfilt(waveform)
    count_unequal_element(expect_waveform, output, 0.000001, 0.000001)

    # Test eager filtfilt float16
    # Original waveform
    dtype = np.float64
    waveform = np.array(
        [
            [0.823655, 0.204918, 0.333587],
            [0.593376, 0.991154, 0.248223],
            [0.300798, 0.905465, 0.759823],
            [0.539423, 0.284265, 0.563423],
            [0.636365, 0.222653, 0.228853],
        ],
        dtype=dtype,
    )
    # Expect waveform
    expect_waveform = np.array(
        [
            [0.77337009, 0.15360786, 0.28125156],
            [0.52548867, 0.92187747, 0.17755712],
            [0.22736443, 0.83051509, 0.68335632],
            [0.48783389, 0.23161545, 0.50971279],
            [0.59621743, 0.18168716, 0.18706884],
        ],
        dtype=np.float32,
    )

    filtfilt = audio.Filtfilt(
        a_coeffs=[1.0201494451137518, -1.9991880801438362, 0.9798505548862483],
        b_coeffs=[0.999797020035959, -1.999594040071918, 0.999797020035959],
    )
    # Filtered waveform by filtfilt
    output = filtfilt(waveform)
    count_unequal_element(expect_waveform.astype(np.float16), output, 0.01, 0.01)

    # Test eager filtfilt error parameter {}
    filtfilt = audio.Filtfilt(
        a_coeffs=[1.0201494451137518, -1.9991880801438362, 0.9798505548862483],
        b_coeffs=[0.999797020035959, -1.999594040071918, 0.999797020035959],
    )
    waveform = {}
    with pytest.raises(TypeError, match="Input should be NumPy audio"):
        filtfilt(waveform)

    # Test eager numpy_array parameter 02
    filtfilt = audio.Filtfilt(
        a_coeffs=[1.0201494451137518, -1.9991880801438362, 0.9798505548862483],
        b_coeffs=[0.999797020035959, -1.999594040071918, 0.999797020035959],
    )
    waveform = np.random.randint(100, size=(5, 6)).astype("float")
    filtfilt(waveform)

    # Test eager numpy_array parameter 05
    filtfilt = audio.Filtfilt(
        a_coeffs=[1.0201494451137518, -1.9991880801438362, 0.9798505548862483],
        b_coeffs=[0.999797020035959, -1.999594040071918, 0.999797020035959],
    )
    waveform = np.random.randint(100, size=(1, 12, 5, 3, 4)).astype("float")
    filtfilt(waveform)

    # Test eager numpy_array parameter 06
    filtfilt = audio.Filtfilt(
        a_coeffs=[1.0201494451137518, -1.9991880801438362, 0.9798505548862483],
        b_coeffs=[0.999797020035959, -1.999594040071918, 0.999797020035959],
    )
    waveform = np.random.rand(5, 6)
    filtfilt(waveform)

    # Test filtfilt coeff 005
    waveform = np.array(
        [
            [0.823655, 0.204918, 0.333587],
            [0.593376, 0.991154, 0.248223],
            [0.300798, 0.905465, 0.759823],
            [0.539423, 0.284265, 0.563423],
            [0.636365, 0.222653, 0.228853],
        ],
        dtype=np.float32,
    )
    # Expect waveform

    a_coeffs = [-16777216, -1.9991880801438362, 0.9798505548862483]
    b_coeffs = [0.999797020035959, 16777216, 0.999797020035959]
    filtfilt = audio.Filtfilt(a_coeffs=a_coeffs, b_coeffs=b_coeffs)
    output = filtfilt(waveform)
    expected_result = np.array(
        [
            [8.2366e-01, 2.0492e-01, 1.2212e-08],
            [5.9338e-01, 9.9115e-01, 5.9065e-08],
            [3.0080e-01, 9.0547e-01, 5.3959e-08],
            [5.3942e-01, 2.8427e-01, 1.6940e-08],
            [6.3636e-01, 2.2265e-01, 1.3268e-08],
        ]
    )
    count_unequal_element(expected_result, output, 0.01, 0.01)


def test_filtfilt_param_check():
    """
    Feature: Filtfilt
    Description: Test Filtfilt with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """
    # Test error size filtfilt
    with pytest.raises(
        ValueError, match="The size of a_coeffs should be the same as that of b_coeffs."
    ):
        audio.Filtfilt(a_coeffs=[1.0, 0.1, 0.3], b_coeffs=[0.1, 0.2])

    # Test filtfilt coeff 005_2
    with pytest.raises(
        ValueError,
        match=r"is not within the required interval of \[-16777216, 16777216\]",
    ):
        audio.Filtfilt(
            a_coeffs=[-16777217, -1.9991880801438362, 0.9798505548862483],
            b_coeffs=[0.999797020035959, 16777217, 0.999797020035959],
        )

    filtfilt = audio.Filtfilt(
        a_coeffs=[1.0201494451137518, -1.9991880801438362, 0.9798505548862483],
        b_coeffs=[0.999797020035959, -1.999594040071918, 0.999797020035959],
    )

    # Test eager filtfilt error parameter bool
    with pytest.raises(TypeError, match="Input should be NumPy audio"):
        filtfilt(True)

    # Test eager filtfilt error parameter []
    with pytest.raises(TypeError, match="Input should be NumPy audio"):
        filtfilt([])

    # Test eager filtfilt error parameter ()
    with pytest.raises(TypeError, match="Input should be NumPy audio"):
        filtfilt(())

    # Test eager filtfilt error parameter one
    with pytest.raises(TypeError, match="Input should be NumPy audio"):
        filtfilt(1)

    # Test eager filtfilt error parameter float
    with pytest.raises(TypeError, match="Input should be NumPy audio"):
        filtfilt(1.221321)

    # Test eager error type parameter 08
    waveform = np.array(["a", "b", "c"])
    with pytest.raises(
        RuntimeError,
        match="the data type of input tensor does not match the requirement of operator.",
    ):
        filtfilt(np.array(["a", "b", "c"]))

    # Test eager error type parameter 01
    waveform = np.random.rand(5, 6).astype(np.int64)
    with pytest.raises(
        RuntimeError,
        match=r"the data type of input tensor does not match the requirement "
        r"of operator. Expecting tensor in type of \[float, double\]. "
        r"But got type int64.",
    ):
        filtfilt(waveform)

    # Test eager error type parameter 04
    waveform = np.random.randint(100, size=(5, 6)).astype(np.uint16)
    with pytest.raises(
        RuntimeError,
        match=r"the data type of input tensor does not match the requirement of operator. "
        r"Expecting tensor in type of \[float, double\]. But got type uint16.",
    ):
        filtfilt(waveform)

    # Test eager error type parameter 03
    waveform = np.random.randint(100, size=(1, 12, 5, 3)).astype(np.uint32)
    with pytest.raises(
        RuntimeError,
        match=r"the data type of input tensor does not match the requirement of operator. "
        r"Expecting tensor in type of \[float, double\]. But got type uint32.",
    ):
        filtfilt(waveform)

    # Test eager error type parameter 07
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.complex_)
    with pytest.raises(TypeError, match="cannot be converted into tensor."):
        filtfilt(waveform)

    # Test filtfilt eager invalid_input_all
    waveform = np.random.rand(2, 1000)
    b_coeffs = [0.1, 0.2, 0.3]

    def test_invalid_input(a_coeffs, clamp, error_type, error_msg):
        with pytest.raises(error_type) as error_info:
            audio.Filtfilt(a_coeffs, b_coeffs, clamp)(waveform)
        assert error_msg in str(error_info.value)

    test_invalid_input(
        1,
        True,
        TypeError,
        "Argument a_coeffs with value 1 is not of type [<class 'list'>, <class 'tuple'>], but got <class 'int'>.",
    )
    test_invalid_input(
        ["0.1", "0.2", "0.3"],
        True,
        TypeError,
        "Argument a_coeffs[0] with value 0.1 is not of type [<class 'float'>, <class 'int'>], but got <class 'str'>.",
    )
    test_invalid_input(
        [234322354352353453651, 0.2, 0.3],
        True,
        ValueError,
        "Input a_coeffs[0] is not within the required interval of [-16777216, 16777216].",
    )
    test_invalid_input(
        [0.1, 0.2, 0.3],
        "True",
        TypeError,
        "Argument clamp with value True is not of type [<class 'bool'>], but got <class 'str'>.",
    )


if __name__ == "__main__":
    test_filtfilt_eager()
    test_filtfilt_pipeline()
    test_filtfilt_invalid_input_all()
    test_filtfilt_transform()
    test_filtfilt_param_check()
