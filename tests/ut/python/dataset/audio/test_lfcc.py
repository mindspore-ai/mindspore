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
"""Test LFCC."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from mindspore.dataset.audio import WindowType, BorderType, NormMode
from . import count_unequal_element


def test_lfcc_pipeline():
    """
    Feature: Test pipeline mode normal testcase: lfcc
    Description: Input audio signal to test pipeline
    Expectation: Output is equal to the expected output
    """
    waveform = [
        [
            [
                1,
                1,
                2,
                2,
                3,
                3,
                4,
                4,
                5,
                5,
                4,
                4,
                3,
                3,
                2,
                2,
                1,
                1,
                0,
                0,
                1,
                1,
                2,
                2,
                3,
                3,
                4,
                4,
                5,
                5,
            ]
        ]
    ]
    dataset = ds.NumpySlicesDataset(waveform, column_names=["audio"], shuffle=False)
    out = audio.LFCC(
        sample_rate=16000,
        n_filter=128,
        n_lfcc=4,
        f_min=0.0,
        f_max=10000.0,
        dct_type=2,
        norm=NormMode.ORTHO,
        log_lf=True,
        speckwargs={
            "n_fft": 16,
            "win_length": 16,
            "hop_length": 8,
            "pad": 0,
            "window": WindowType.HANN,
            "power": 2.0,
            "normalized": False,
            "center": True,
            "pad_mode": BorderType.REFLECT,
            "onesided": True,
        },
    )
    dataset = dataset.map(
        operations=out, input_columns=["audio"], output_columns=["LFCC"]
    )
    result = np.array(
        [
            [
                [-137.9132, -137.0732, -137.2996, -140.0339],
                [4.1616, 5.3870, 4.2134, 4.9916],
                [-3.4581, -4.1653, -3.9544, -0.3347],
                [2.0614, 2.7895, 2.7281, 0.7957],
            ]
        ]
    )
    for data1 in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(data1["LFCC"], result, 0.0001, 0.0001)


def test_lfcc_eager():
    """
    Feature: Test eager mode normal testcase: lfcc
    Description: Input audio signal to test eager
    Expectation: Output is equal to the expected output
    """
    waveform = np.array(
        [
            [
                [
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    4,
                    4,
                    3,
                    3,
                    2,
                    2,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                ]
            ]
        ]
    )
    out = audio.LFCC(
        sample_rate=16000,
        n_filter=128,
        n_lfcc=4,
        f_min=0.0,
        f_max=10000.0,
        dct_type=2,
        norm=NormMode.ORTHO,
        log_lf=False,
        speckwargs={
            "n_fft": 16,
            "win_length": 16,
            "hop_length": 8,
            "pad": 0,
            "window": WindowType.HANN,
            "power": 2.0,
            "normalized": False,
            "center": True,
            "pad_mode": BorderType.REFLECT,
            "onesided": True,
        },
    )(waveform)
    result = np.array(
        [
            [
                [
                    [-5.5005e02, -5.4640e02, -5.4739e02, -5.5840e02],
                    [1.6892e01, 2.2214e01, 1.7117e01, 2.0112e01],
                    [-1.2942e01, -1.6014e01, -1.5098e01, -3.6112e-01],
                    [8.1695e00, 1.1332e01, 1.1065e01, 3.6742e00],
                ]
            ]
        ]
    )
    count_unequal_element(out, result, 0.0001, 0.0001)


def test_lfcc_invalid_input():
    """
    Feature: LFCC
    Description: Test operation with invalid input.
    Expectation: Throw exception as expected.
    """
    try:
        audio.LFCC(sample_rate=-1)
    except ValueError as error:
        assert (
            "Input sample_rate is not within the required interval of [0, 2147483647]."
            in str(error)
        )
    try:
        audio.LFCC(sample_rate=1.1)
    except TypeError as error:
        assert (
            "Argument sample_rate with value 1.1 is not of type [<class 'int'>]"
            in str(error)
        )
    try:
        audio.LFCC(n_filter=-1)
    except ValueError as error:
        assert (
            "Input n_filter is not within the required interval of [1, 2147483647]."
            in str(error)
        )
    try:
        audio.LFCC(n_filter=1.1)
    except TypeError as error:
        assert "Argument n_filter with value 1.1 is not of type [<class 'int'>]" in str(
            error
        )
    try:
        audio.LFCC(n_lfcc=-1)
    except ValueError as error:
        assert (
            "Input n_lfcc is not within the required interval of [1, 2147483647]."
            in str(error)
        )
    try:
        audio.LFCC(n_lfcc=1.1)
    except TypeError as error:
        assert "Argument n_lfcc with value 1.1 is not of type [<class 'int'>]" in str(
            error
        )
    try:
        audio.LFCC(log_lf=-1)
    except TypeError as error:
        assert "Argument log_lf with value -1 is not of type [<class 'bool'>]" in str(
            error
        )
    try:
        audio.LFCC(norm="Karl Marx")
    except TypeError as error:
        assert (
            "Argument norm with value Karl Marx is not of type [<enum 'NormMode'>]"
            in str(error)
        )
    try:
        audio.LFCC(dct_type=-1)
    except ValueError as error:
        assert "dct_type must be 2, but got : -1." in str(error)
    try:
        audio.LFCC(f_min=10000)
    except ValueError as error:
        assert (
            "sample_rate // 2 should be greater than f_min when f_max is set to None"
            in str(error)
        )
    try:
        audio.LFCC(f_min=False)
    except TypeError as error:
        assert (
            "Argument f_min with value False is not of type (<class 'int'>, <class 'float'>)"
            in str(error)
        )
    try:
        audio.LFCC(f_min=2, f_max=1)
    except ValueError as error:
        assert "f_max should be greater than or equal to f_min" in str(error)
    try:
        audio.LFCC(f_max=False)
    except TypeError as error:
        assert (
            "Argument f_max with value False is not of type (<class 'int'>, <class 'float'>)"
            in str(error)
        )
    try:
        audio.LFCC(speckwargs=False)
    except TypeError as error:
        assert (
            "Argument speckwargs with value False is not of type [<class 'dict'>]"
            in str(error)
        )
    try:
        audio.LFCC(
            speckwargs={
                "n_fft": 400,
                "win_length": 16,
                "hop_length": 8,
                "pad": 0,
                "window": "WindowType.HANN",
                "power": 2.0,
                "normalized": False,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
            }
        )
    except TypeError as error:
        assert (
            "Argument window with value WindowType.HANN is not of type [<enum 'WindowType'>]"
            in str(error)
        )
    try:
        audio.LFCC(
            speckwargs={
                "n_fft": 400,
                "win_length": 16,
                "hop_length": 8,
                "pad": 0,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": False,
                "center": True,
                "pad_mode": "BorderType.REFLECT",
                "onesided": True,
            }
        )
    except TypeError as error:
        assert (
            "Argument pad_mode with value BorderType.REFLECT is not of type [<enum 'BorderType'>]"
            in str(error)
        )
    try:
        audio.LFCC(
            speckwargs={
                "n_fft": 400,
                "win_length": 16,
                "hop_length": 8,
                "pad": -1,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": False,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
            }
        )
    except ValueError as error:
        assert (
            "Input pad is not within the required interval of [0, 2147483647]"
            in str(error)
        )
    try:
        audio.LFCC(
            speckwargs={
                "n_fft": 400,
                "win_length": 16,
                "hop_length": 8,
                "pad": 1.1,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": False,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
            }
        )
    except TypeError as error:
        assert "Argument pad with value 1.1 is not of type [<class 'int'>]" in str(
            error
        )
    try:
        audio.LFCC(
            speckwargs={
                "n_fft": 400,
                "win_length": 16,
                "hop_length": 8,
                "pad": 0,
                "window": WindowType.HANN,
                "power": -1.0,
                "normalized": False,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
            }
        )
    except ValueError as error:
        assert (
            "Input power is not within the required interval of [0, 16777216]"
            in str(error)
        )
    try:
        audio.LFCC(
            speckwargs={
                "n_fft": 400,
                "win_length": 16,
                "hop_length": 8,
                "pad": 0,
                "window": WindowType.HANN,
                "power": 2,
                "normalized": False,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
            }
        )
    except TypeError as error:
        assert "Argument power with value 2 is not of type [<class 'float'>]" in str(
            error
        )
    try:
        audio.LFCC(
            speckwargs={
                "n_fft": 40,
                "win_length": 41,
                "hop_length": 8,
                "pad": 0,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": False,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
            }
        )
    except ValueError as error:
        assert "win_length must be less than or equal to n_fft" in str(error)
    try:
        audio.LFCC(
            speckwargs={
                "n_fft": 16,
                "win_length": 16,
                "hop_length": 8,
                "pad": 0,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": False,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
            }
        )
    except ValueError as error:
        assert "n_fft should be greater than or equal to n_lfcc" in str(error)


def test_lfcc_transform():
    """
    Feature: LFCC
    Description: Test LFCC with various valid input parameters and data types
    Expectation: The operation completes successfully
    """
    waveform = np.random.randn(10, 30)

    # Test LFCC operator pipeline mode
    dataset = ds.NumpySlicesDataset(waveform, column_names=["audio"], shuffle=False)
    transform = audio.LFCC(
        sample_rate=16000,
        n_filter=128,
        n_lfcc=4,
        f_min=0.0,
        f_max=10000.0,
        dct_type=2,
        norm=NormMode.ORTHO,
        log_lf=False,
        speckwargs={
            "n_fft": 16,
            "win_length": 16,
            "hop_length": 8,
            "pad": 0,
            "window": WindowType.HANN,
            "power": 2.0,
            "normalized": False,
            "center": True,
            "pad_mode": BorderType.REFLECT,
            "onesided": True,
        },
    )
    dataset = dataset.map(operations=transform)
    for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        pass

    # Test LFCC operator parameter sample_rate as int
    audio.LFCC(sample_rate=3200)(waveform)

    # Test LFCC operator parameter dct_type as 2
    audio.LFCC(dct_type=2)(waveform)

    # Test LFCC operator parameter dct_type as 1
    with pytest.raises(ValueError, match=r"dct_type must be 2, but got : 1"):
        audio.LFCC(dct_type=1)(waveform)

    # Test LFCC operator parameter norm as NormMode.ORTHO type
    audio.LFCC(norm=NormMode.ORTHO)(waveform)

    # Test LFCC operator parameter norm as NormMode.NONE type
    audio.LFCC(norm=NormMode.NONE)(waveform)

    # Test LFCC operator parameter log_lf as False
    audio.LFCC(log_lf=False)(waveform)

    # Test LFCC operator parameter log_lf as True
    audio.LFCC(log_lf=True)(waveform)

    # Test LFCC operator parameter speckwargs as None
    audio.LFCC(speckwargs=None)(waveform)

    # Test LFCC operator parameter speckwargs as dict
    audio.LFCC(
        speckwargs={
            "n_fft": 64,
            "win_length": 64,
            "hop_length": 32,
            "pad": 0,
            "window": WindowType.HANN,
            "power": 2.0,
            "normalized": False,
            "center": True,
            "pad_mode": BorderType.REFLECT,
            "onesided": True,
        }
    )(waveform)

    # Test LFCC operator parameter n_filter as 1
    audio.LFCC(n_filter=1)(waveform)

    # Test LFCC operator parameter n_lfcc as 1
    audio.LFCC(n_lfcc=1)(waveform)

    # Test LFCC operator parameter f_min as int
    audio.LFCC(f_min=1)(waveform)

    # Test LFCC operator parameter f_max as int
    audio.LFCC(f_max=1)(waveform)

    # Test LFCC operator parameter f_max as None
    audio.LFCC(f_max=None)(waveform)

    # Test LFCC operator parameter f_max as 0
    audio.LFCC(f_max=0)(waveform)

    # LFCC operator eager mode test with one-dimensional input data
    waveform = np.random.randn(1, 2, 3, 201)
    audio.LFCC()(waveform)


def test_lfcc_param_check():
    """
    Feature: Lfcc
    Description: Test Lfcc with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """
    waveform = np.random.randn(10, 30)

    # Test LFCC operator parameter sample_rate as non-int type
    with pytest.raises(TypeError):
        audio.LFCC(sample_rate="a")

    # Test LFCC operator parameter sample_rate as -1
    with pytest.raises(
        ValueError,
        match=r"Input sample_rate is not within the required interval of \[0, 2147483647\]",
    ):
        audio.LFCC(sample_rate=-1)

    # Test LFCC operator parameter sample_rate as 2147483648
    with pytest.raises(
        ValueError,
        match=r"Input sample_rate is not within the required interval of \[0, 2147483647\]",
    ):
        audio.LFCC(sample_rate=-1)

    # Test LFCC operator parameter sample_rate as 0
    with pytest.raises(
        ValueError,
        match=r"sample_rate // 2 should be greater than f_min when f_max is set to None.",
    ):
        audio.LFCC(sample_rate=0)

    # Test LFCC operator parameter n_filter as non-int type
    with pytest.raises(TypeError):
        audio.LFCC(n_filter="n_filter")

    # Test LFCC operator parameter n_filter as -1
    with pytest.raises(
        ValueError,
        match=r"Input n_filter is not within the required interval of \[1, 2147483647\]",
    ):
        audio.LFCC(n_filter=-1)

    # Test LFCC operator parameter n_lfcc as non-int type
    with pytest.raises(TypeError):
        audio.LFCC(n_lfcc="n_lfcc")

    # Test LFCC operator parameter n_lfcc as -1
    with pytest.raises(
        ValueError,
        match=r"Input n_lfcc is not within the required interval of \[1, 2147483647\]",
    ):
        audio.LFCC(n_lfcc=-1)

    # Test LFCC operator parameter n_lfcc as 2147483647
    with pytest.raises(
        RuntimeError,
        match=r"LFCC: n_fft should be greater than or equal to n_lfcc, "
        r"but got n_lfcc: 2147483647 and n_fft: 400",
    ):
        audio.LFCC(n_lfcc=2147483647)(waveform)

    # Test LFCC operator parameter n_lfcc greater than parameter speckwargs[n_fft]
    with pytest.raises(
        RuntimeError,
        match=r"LFCC: n_fft should be greater than or equal to n_lfcc, "
        r"but got n_lfcc: 401 and n_fft: 400",
    ):
        audio.LFCC(n_lfcc=401)(waveform)

    # Test LFCC operator parameter f_min as non-float type
    with pytest.raises(TypeError):
        audio.LFCC(f_min="f_min")

    # `f_min` is greater than sample_rate // 2 when f_max is set to None.
    with pytest.raises(
        ValueError,
        match=r"sample_rate // 2 should be greater than f_min when f_max is set to None.",
    ):
        audio.LFCC(sample_rate=16000, f_min=8001)

    # `f_min` is equal sample_rate // 2 when f_max is set to None.
    with pytest.raises(
        ValueError,
        match=r"sample_rate // 2 should be greater than f_min when f_max is set to None.",
    ):
        audio.LFCC(sample_rate=16000, f_min=8000)

    # Test LFCC operator parameter f_max as non-float type
    with pytest.raises(TypeError):
        audio.LFCC(f_max="f_max")

    # Test LFCC operator parameter f_max as negative number
    with pytest.raises(
        ValueError,
        match=r"Input f_max is not within the required interval of \[0, 16777216\]",
    ):
        audio.LFCC(f_max=-1)

    # Test LFCC operator parameter f_min as negative number
    with pytest.raises(
        ValueError,
        match=r"Input f_min is not within the required interval of \[0, 16777216\]",
    ):
        audio.LFCC(f_min=-1)

    # Test LFCC operator parameter f_max less than or equal to f_min
    with pytest.raises(
        ValueError,
        match=r"f_max should be greater than or equal to f_min, "
        r"but got f_min: 101 and f_max: 100",
    ):
        audio.LFCC(f_max=100, f_min=101)

    # Test LFCC operator parameter dct_type as non-int type
    with pytest.raises(ValueError):
        audio.LFCC(dct_type="dct_type")

    # Test LFCC operator parameter norm as non-NormMode type
    with pytest.raises(TypeError):
        audio.LFCC(norm="norm")

    # Test LFCC operator parameter log_lf as non-bool type
    with pytest.raises(TypeError):
        audio.LFCC(log_lf="log_lf")

    # Test LFCC operator parameter speckwargs as non-dict type
    with pytest.raises(TypeError):
        audio.LFCC(speckwargs="speckwargs")

    # Test LFCC operator parameter speckwargs[n_fft] less than coefficients
    with pytest.raises(
        ValueError,
        match=r"n_fft should be greater than or equal to n_lfcc, "
        r"but got n_fft: 64 and n_lfcc: 100.",
    ):
        audio.LFCC(
            n_lfcc=100,
            speckwargs={
                "n_fft": 64,
                "win_length": 64,
                "hop_length": 32,
                "pad": 0,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": False,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
            },
        )


if __name__ == "__main__":
    test_lfcc_pipeline()
    test_lfcc_eager()
    test_lfcc_invalid_input()
    test_lfcc_transform()
    test_lfcc_param_check()
