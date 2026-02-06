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
"""Test SlidingWindowCmn."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from . import count_unequal_element


def test_sliding_window_cmn_eager():
    """
    Feature: Test the basic function in eager mode.
    Description: Mindspore eager mode normal testcase:sliding_window_cmn.
    Expectation: Compile done without error.
    """
    # Original waveform
    waveform = np.array(
        [
            [[0.0000, 0.1000, 0.2000], [0.3000, 0.4000, 0.5000]],
            [[0.6000, 0.7000, 0.8000], [0.9000, 1.0000, 1.1000]],
        ],
        dtype=np.float64,
    )
    # Expect waveform
    expect_waveform = np.array(
        [
            [[-0.1500, -0.1500, -0.1500], [0.1500, 0.1500, 0.1500]],
            [[-0.1500, -0.1500, -0.1500], [0.1500, 0.1500, 0.1500]],
        ],
        dtype=np.float64,
    )
    sliding_window_cmn = audio.SlidingWindowCmn(500, 200, False, False)
    # Filtered waveform by sliding_window_cmn
    output = sliding_window_cmn(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)

    # Original waveform
    waveform = np.array(
        [
            [0.0050, 0.0306, 0.6146, 0.7620, 0.6369],
            [0.9525, 0.0362, 0.6721, 0.6867, 0.8466],
        ],
        dtype=np.float32,
    )
    # Expect waveform
    expect_waveform = np.array(
        [
            [-1.0000, -1.0000, -1.0000, 1.0000, -1.0000],
            [1.0000, 1.0000, 1.0000, -1.0000, 1.0000],
        ],
        dtype=np.float32,
    )
    sliding_window_cmn = audio.SlidingWindowCmn(600, 100, False, True)
    # Filtered waveform by sliding_window_cmn
    output = sliding_window_cmn(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)

    # Original waveform
    waveform = np.array(
        [
            [
                [
                    0.3764,
                    0.4168,
                    0.0635,
                    0.7082,
                    0.4596,
                    0.3457,
                    0.8438,
                    0.8860,
                    0.9151,
                    0.5746,
                    0.6630,
                    0.0260,
                    0.2631,
                    0.7410,
                    0.5627,
                    0.6749,
                    0.7099,
                    0.1120,
                    0.4794,
                    0.2778,
                ],
                [
                    0.4157,
                    0.2246,
                    0.2488,
                    0.2686,
                    0.0562,
                    0.4422,
                    0.9407,
                    0.0756,
                    0.5737,
                    0.7501,
                    0.3122,
                    0.7982,
                    0.3034,
                    0.1880,
                    0.2298,
                    0.0961,
                    0.7439,
                    0.9947,
                    0.8156,
                    0.2907,
                ],
            ]
        ],
        dtype=np.float64,
    )
    # Expect waveform
    expect_waveform = np.array(
        [
            [
                [
                    -1.0000,
                    1.0000,
                    -1.0000,
                    1.0000,
                    1.0000,
                    -1.0000,
                    -1.0000,
                    1.0000,
                    1.0000,
                    -1.0000,
                    1.0000,
                    -1.0000,
                    -1.0000,
                    1.0000,
                    1.0000,
                    1.0000,
                    -1.0000,
                    -1.0000,
                    -1.0000,
                    -1.0000,
                ],
                [
                    1.0000,
                    -1.0000,
                    1.0000,
                    -1.0000,
                    -1.0000,
                    1.0000,
                    1.0000,
                    -1.0000,
                    -1.0000,
                    1.0000,
                    -1.0000,
                    1.0000,
                    1.0000,
                    -1.0000,
                    -1.0000,
                    -1.0000,
                    1.0000,
                    1.0000,
                    1.0000,
                    1.0000,
                ],
            ]
        ],
        dtype=np.float64,
    )
    sliding_window_cmn = audio.SlidingWindowCmn(3, 0, True, True)
    # Filtered waveform by sliding_window_cmn
    output = sliding_window_cmn(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_sliding_window_cmn_pipeline():
    """
    Feature: Test the basic function in pipeline mode.
    Description: Mindspore pipeline mode normal testcase:sliding_window_cmn.
    Expectation: Compile done without error.
    """
    # Original waveform
    waveform = np.array([[[3.2, 2.1, 1.3], [6.2, 5.3, 6]]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array(
        [[[-1.0000, -1.0000, -1.0000], [1.0000, 1.0000, 1.0000]]], dtype=np.float64
    )
    dataset = ds.NumpySlicesDataset(waveform, ["audio"], shuffle=False)
    sliding_window_cmn = audio.SlidingWindowCmn(600, 100, False, True)
    # Filtered waveform by sliding_window_cmn
    dataset = dataset.map(
        input_columns=["audio"], operations=sliding_window_cmn, num_parallel_workers=8
    )
    i = 0
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(expect_waveform[i, :], item["audio"], 0.0001, 0.0001)
        i += 1


def test_sliding_window_cmn_invalid_input():
    """
    Feature: Test the validate function with invalid parameters.
    Description: Mindspore invalid parameters testcase:sliding_window_cmn.
    Expectation: Compile done without error.
    """

    def test_invalid_input(
        cmn_window, min_cmn_window, center, norm_vars, error, error_msg
    ):
        with pytest.raises(error) as error_info:
            audio.SlidingWindowCmn(cmn_window, min_cmn_window, center, norm_vars)
        assert error_msg in str(error_info.value)

    test_invalid_input(
        "600",
        100,
        False,
        False,
        TypeError,
        "Argument cmn_window with value 600 is not of type [<class 'int'>],"
        " but got <class 'str'>.",
    )
    test_invalid_input(
        441324343243242342345300,
        100,
        False,
        False,
        ValueError,
        "Input cmn_window is not within the required interval of [0, 2147483647].",
    )
    test_invalid_input(
        600,
        "100",
        False,
        False,
        TypeError,
        "Argument min_cmn_window with value 100 is not of type [<class 'int'>],"
        " but got <class 'str'>.",
    )
    test_invalid_input(
        600,
        441324343243242342345300,
        False,
        False,
        ValueError,
        "Input min_cmn_window is not within the required interval of [0, 2147483647].",
    )
    test_invalid_input(
        600,
        100,
        "False",
        False,
        TypeError,
        "Argument center with value False is not of type [<class 'bool'>],"
        " but got <class 'str'>.",
    )
    test_invalid_input(
        600,
        100,
        False,
        "False",
        TypeError,
        "Argument norm_vars with value False is not of type [<class 'bool'>],"
        " but got <class 'str'>.",
    )


def test_sliding_window_cmn_transform():
    """
    Feature: SlidingWindowCmn
    Description: Test SlidingWindowCmn with various valid input parameters and data types
    Expectation: The operation completes successfully
    """
    waveform = np.array(
        [
            [
                -1.6798493924858628,
                -1.7071343394078222,
                -0.20683098446927464,
                -1.1690232518341401,
            ],
            [
                1.3589180927666873,
                -2.1876893787489657,
                -1.3641272270764149,
                0.8794233783133073,
            ],
        ]
    )

    # test SlidingWindowCmn normal with (400, 200, False, False)
    dataset = ds.NumpySlicesDataset([waveform], ["audio"], shuffle=False)
    sliding_window_cmn = audio.SlidingWindowCmn(400, 200, False, False)
    dataset = dataset.map(input_columns=["audio"], operations=sliding_window_cmn)
    expect = np.array(
        [
            [
                -1.519383742626275,
                0.24027751967057176,
                0.5786481213035701,
                -1.0242233150737237,
            ],
            [
                1.519383742626275,
                -0.24027751967057176,
                -0.5786481213035701,
                1.0242233150737237,
            ],
        ]
    )
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert np.allclose(data["audio"], expect, 0.01, 0.01)

    # Test SlidingWindowCmn normal with (400, 200, True, True)
    dataset = ds.NumpySlicesDataset([waveform], ["audio"], shuffle=False)
    sliding_window_cmn = audio.SlidingWindowCmn(400, 200, True, True)
    dataset = dataset.map(input_columns=["audio"], operations=sliding_window_cmn)
    expect = np.array(
        [
            [-1.0, 0.9999999999999956, 0.9999999999999999, -1.0],
            [1.0, -0.9999999999999956, -0.9999999999999999, 1.0],
        ]
    )
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert np.allclose(data["audio"], expect, 0.01, 0.01)

    # Test SlidingWindowCmn normal with (400, 200, False, True)
    dataset = ds.NumpySlicesDataset([waveform], ["audio"], shuffle=False)
    sliding_window_cmn = audio.SlidingWindowCmn(400, 200, False, True)
    dataset = dataset.map(input_columns=["audio"], operations=sliding_window_cmn)
    expect = np.array(
        [
            [-1.0, 0.9999999999999956, 0.9999999999999999, -1.0],
            [1.0, -0.9999999999999956, -0.9999999999999999, 1.0],
        ]
    )
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert np.allclose(data["audio"], expect, 0.01, 0.01)

    # Test SlidingWindowCmn normal with (400, 200, True, False)
    dataset = ds.NumpySlicesDataset([waveform], ["audio"], shuffle=False)
    sliding_window_cmn = audio.SlidingWindowCmn(400, 200, True, False)
    dataset = dataset.map(input_columns=["audio"], operations=sliding_window_cmn)
    expect = np.array(
        [
            [
                -1.519383742626275,
                0.24027751967057176,
                0.5786481213035701,
                -1.0242233150737237,
            ],
            [
                1.519383742626275,
                -0.24027751967057176,
                -0.5786481213035701,
                1.0242233150737237,
            ],
        ]
    )
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert np.allclose(data["audio"], expect, 0.01, 0.01)

    # Test SlidingWindowCmn eager int8
    waveform = waveform.astype(np.int8).reshape(1, -1)
    sliding_window_cmn = audio.SlidingWindowCmn()
    output = sliding_window_cmn(waveform)
    assert len(output.shape) == 1

    # Test SlidingWindowCmn eager uint8
    waveform = waveform.astype(np.uint8).reshape(1, -1)
    sliding_window_cmn = audio.SlidingWindowCmn()
    output = sliding_window_cmn(waveform)
    assert len(output.shape) == 1

    # Test SlidingWindowCmn eager int32
    waveform = waveform.astype(np.int32).reshape(1, -1)
    sliding_window_cmn = audio.SlidingWindowCmn()
    output = sliding_window_cmn(waveform)
    assert len(output.shape) == 1

    # Test SlidingWindowCmn eager uint32
    waveform = waveform.astype(np.uint32).reshape(1, -1)
    sliding_window_cmn = audio.SlidingWindowCmn()
    output = sliding_window_cmn(waveform)
    assert len(output.shape) == 1

    # Test SlidingWindowCmn eager float16
    waveform = waveform.astype(np.float16).reshape(1, -1)
    sliding_window_cmn = audio.SlidingWindowCmn()
    output = sliding_window_cmn(waveform)
    assert len(output.shape) == 1

    # Test SlidingWindowCmn eager float32
    waveform = waveform.astype(np.float32).reshape(1, -1)
    sliding_window_cmn = audio.SlidingWindowCmn()
    output = sliding_window_cmn(waveform)
    assert len(output.shape) == 1

    # Test SlidingWindowCmn eager float64
    waveform = waveform.astype(np.float64).reshape(1, -1)
    sliding_window_cmn = audio.SlidingWindowCmn()
    output = sliding_window_cmn(waveform)
    assert len(output.shape) == 1


def test_sliding_window_cmn_param_check():
    """
    Feature: SlidingWindowCmn
    Description: Test SlidingWindowCmn with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """

    waveform = np.random.random((2,))
    with pytest.raises(
        RuntimeError,
        match="SlidingWindowCmn: the shape of input tensor does not match the requirement "
        "of operator. Expecting tensor in shape of <..., freq, time>.",
    ):
        audio.SlidingWindowCmn(
            cmn_window=400, min_cmn_window=100, center=False, norm_vars=False
        )(waveform)

    # Test SlidingWindowCmn eager string
    waveform = np.array([["a", "b"], ["c", "d"]], np.str_)
    with pytest.raises(
        RuntimeError,
        match="SlidingWindowCmn: the data type of input tensor does not match the "
        "requirement of operator. "
        "Expecting tensor in type of \\[int, float, double\\].",
    ):
        audio.SlidingWindowCmn()(waveform)

    # Test SlidingWindowCmn input tensor is Null
    with pytest.raises(RuntimeError, match="Input Tensor is not valid."):
        audio.SlidingWindowCmn()()

    # Test SlidingWindowCmn error sliding_window_cmn with cmn_window = -1
    with pytest.raises(
        ValueError,
        match=r"Input cmn_window is not within the required interval of \[0, 2147483647\].",
    ):
        audio.SlidingWindowCmn(
            cmn_window=-1, min_cmn_window=100, center=False, norm_vars=False
        )

    # Test SlidingWindowCmn error sliding_window_cmn with cmn_window = [600]
    with pytest.raises(
        TypeError,
        match=r"Argument cmn_window with value \[600\] is not of type \[<class 'int'>\], "
        r"but got <class 'list'>",
    ):
        audio.SlidingWindowCmn(
            cmn_window=[600], min_cmn_window=100, center=False, norm_vars=False
        )

    # Test error sliding_window_cmn with cmn_window = "600"
    with pytest.raises(
        TypeError,
        match=r"Argument cmn_window with value 600 is not of type \[<class 'int'>\], "
        r"but got <class 'str'>.",
    ):
        audio.SlidingWindowCmn(
            cmn_window="600", min_cmn_window=100, center=False, norm_vars=False
        )

    # Test error sliding_window_cmn with min_cmn_window = -1
    with pytest.raises(
        ValueError,
        match=r"Input min_cmn_window is not within "
        r"the required interval of \[0, 2147483647\].",
    ):
        audio.SlidingWindowCmn(
            cmn_window=400, min_cmn_window=-1, center=False, norm_vars=False
        )

    # Test error sliding_window_cmn with min_cmn_window = [100]
    with pytest.raises(
        TypeError,
        match=r"Argument min_cmn_window with value \[100\] is not of type \[<class 'int'>\],"
        r" but got <class 'list'>.",
    ):
        audio.SlidingWindowCmn(
            cmn_window=400, min_cmn_window=[100], center=False, norm_vars=False
        )

    # Test error sliding_window_cmn with min_cmn_window = "100"
    with pytest.raises(
        TypeError,
        match=r"Argument min_cmn_window with value 100 is not of type \[<class 'int'>\],"
        r" but got <class 'str'>.",
    ):
        audio.SlidingWindowCmn(
            cmn_window=400, min_cmn_window="100", center=False, norm_vars=False
        )


if __name__ == "__main__":
    test_sliding_window_cmn_eager()
    test_sliding_window_cmn_pipeline()
    test_sliding_window_cmn_invalid_input()
    test_sliding_window_cmn_transform()
    test_sliding_window_cmn_param_check()
