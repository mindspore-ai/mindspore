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
"""Test Vol."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from . import count_unequal_element


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    """Precision calculation formula"""
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan):
        count_unequal_element(data_expected, data_me, rtol, atol)


def test_vol_eager():
    """
    Feature: Vol
    Description: Test Vol in eager mode under normal test case
    Expectation: Output is equal to the expected output
    """

    data = np.array(
        [
            [0.3667, 0.5295, 0.2949, 0.4508, 0.6457, 0.3625, 0.4377, 0.3568],
            [0.6488, 0.6525, 0.5140, 0.6725, 0.9261, 0.0609, 0.3910, 0.4608],
            [0.0454, 0.0487, 0.6990, 0.1637, 0.5763, 0.1086, 0.5343, 0.4699],
            [0.9993, 0.0776, 0.3498, 0.0429, 0.1588, 0.3061, 0.1166, 0.3716],
            [0.7625, 0.2410, 0.8888, 0.5027, 0.0913, 0.2520, 0.5625, 0.9873],
        ]
    ).astype(np.float32)
    # cal from benchmark
    out_expect = np.array(
        [
            [0.0733, 0.1059, 0.0590, 0.0902, 0.1291, 0.0725, 0.0875, 0.0714],
            [0.1298, 0.1305, 0.1028, 0.1345, 0.1852, 0.0122, 0.0782, 0.0922],
            [0.0091, 0.0097, 0.1398, 0.0327, 0.1153, 0.0217, 0.1069, 0.0940],
            [0.1999, 0.0155, 0.0700, 0.0086, 0.0318, 0.0612, 0.0233, 0.0743],
            [0.1525, 0.0482, 0.1778, 0.1005, 0.0183, 0.0504, 0.1125, 0.1975],
        ]
    )
    vol = audio.Vol(gain=0.2, gain_type=audio.GainType.AMPLITUDE)
    output = vol(data)
    allclose_nparray(output, out_expect, 0.0001, 0.0001)

    data = np.array(
        [
            [
                [-0.5794799327850342, 0.19526369869709015],
                [-0.5935744047164917, 0.2948109209537506],
                [-0.42077431082725525, 0.04923877865076065],
            ],
            [
                [0.5497273802757263, -0.22815021872520447],
                [-0.05891447141766548, -0.16206198930740356],
                [-1.4782767295837402, -1.3815662860870361],
            ],
        ]
    ).astype(np.float32)
    # cal from benchmark
    out_expect = np.array(
        [
            [
                [-0.5761537551879883, 0.1941428929567337],
                [-0.5901673436164856, 0.2931187152862549],
                [-0.41835910081863403, 0.04895615205168724],
            ],
            [
                [0.5465719699859619, -0.22684065997600555],
                [-0.0585763081908226, -0.16113176941871643],
                [-1.0, -1.0],
            ],
        ]
    )
    vol = audio.Vol(gain=-0.05, gain_type=audio.GainType.DB)
    output = vol(data)
    allclose_nparray(output, out_expect, 0.0001, 0.0001)

    data = np.array(
        [
            [
                [
                    0.09491927176713943,
                    0.11639882624149323,
                    -0.1725238710641861,
                    -0.18556903302669525,
                ],
                [
                    -0.7140364646911621,
                    1.6223102807998657,
                    1.6710518598556519,
                    0.6019048094749451,
                ],
            ],
            [
                [
                    -0.8635917901992798,
                    -0.31538113951683044,
                    -0.2209240198135376,
                    1.3067045211791992,
                ],
                [
                    -2.0922982692718506,
                    0.6822009682655334,
                    0.20066820085048676,
                    0.006392406765371561,
                ],
            ],
        ]
    )
    # cal from benchmark
    out_expect = np.array(
        [
            [
                [
                    0.042449187487363815,
                    0.05205513536930084,
                    -0.07715501636266708,
                    -0.08298899233341217,
                ],
                [
                    -0.31932681798934937,
                    0.7255191802978516,
                    0.7473170757293701,
                    0.2691799998283386,
                ],
            ],
            [
                [
                    -0.38620999455451965,
                    -0.14104272425174713,
                    -0.09880022704601288,
                    0.5843760371208191,
                ],
                [
                    -0.935704231262207,
                    0.30508953332901,
                    0.0897415429353714,
                    0.0028587712440639734,
                ],
            ],
        ]
    )
    vol = audio.Vol(gain=0.2, gain_type=audio.GainType.POWER)
    output = vol(data)
    allclose_nparray(output, out_expect, 0.0001, 0.0001)


def test_vol_pipeline():
    """
    Feature: Vol
    Description: Test Vol in pipeline mode under normal test case
    Expectation: Output is equal to the expected output
    """

    data = np.array([[[0.7012, 0.2500, 0.0108], [0.3617, 0.6367, 0.6096]]]).astype(
        np.float32
    )
    out_expect = np.array([[1.0000, 0.7906, 0.0342], [1.0000, 1.0000, 1.0000]])
    dataset = ds.NumpySlicesDataset(data, column_names=["multi_dimensional_data"])
    transforms = [audio.Vol(gain=10, gain_type=audio.GainType.POWER)]
    dataset = dataset.map(
        operations=transforms, input_columns=["multi_dimensional_data"]
    )
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        output = item["multi_dimensional_data"]
        allclose_nparray(output, out_expect, 0.0001, 0.0001)

    data = np.array(
        [
            [
                [
                    0.9342139979247938,
                    0.613965955965896,
                    0.5356328030249583,
                    0.589909976354571,
                ],
                [
                    0.7301220295167696,
                    0.31194499547960186,
                    0.3982210622160919,
                    0.20984374897512215,
                ],
                [
                    0.18619300588033616,
                    0.9443723899839336,
                    0.7395507950492876,
                    0.4904588086175671,
                ],
            ]
        ]
    )
    data = data.astype(np.float64)
    out_expect = np.array(
        [
            [
                0.18684279918670654,
                0.12279318571090699,
                0.10712655782699586,
                0.1179819941520691,
            ],
            [
                0.1460244059562683,
                0.062388998270034794,
                0.07964421510696412,
                0.04196875095367432,
            ],
            [
                0.03723860085010529,
                0.1888744831085205,
                0.14791015386581421,
                0.09809176325798036,
            ],
        ]
    )
    dataset = ds.NumpySlicesDataset(data, column_names=["multi_dimensional_data"])
    transforms = [audio.Vol(gain=0.2, gain_type=audio.GainType.AMPLITUDE)]
    dataset = dataset.map(
        operations=transforms, input_columns=["multi_dimensional_data"]
    )
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        output = item["multi_dimensional_data"]
        allclose_nparray(output, out_expect, 0.0001, 0.0001)

    data = np.array(
        [
            [
                [0.1302, 0.5908, 0.1225, 0.7044],
                [0.6405, 0.6540, 0.9908, 0.8605],
                [0.7023, 0.0115, 0.8790, 0.5806],
            ]
        ]
    ).astype(np.float32)
    out_expect = np.array(
        [
            [0.1096, 0.4971, 0.1031, 0.5927],
            [0.5389, 0.5503, 0.8336, 0.7240],
            [0.5909, 0.0097, 0.7396, 0.4885],
        ]
    )
    dataset = ds.NumpySlicesDataset(data, column_names=["multi_dimensional_data"])
    transforms = [audio.Vol(gain=-1.5, gain_type=audio.GainType.DB)]
    dataset = dataset.map(
        operations=transforms, input_columns=["multi_dimensional_data"]
    )
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        output = item["multi_dimensional_data"]
        allclose_nparray(output, out_expect, 0.0001, 0.0001)


def test_vol_invalid_input():
    """
    Feature: Vol
    Description: Test Vol with invalid input
    Expectation: Error is raised as expected
    """

    def test_invalid_input(gain, gain_type, error, error_msg):
        with pytest.raises(error) as error_info:
            audio.Vol(gain, gain_type)
        assert error_msg in str(error_info.value)

    test_invalid_input(
        -1.5,
        audio.GainType.POWER,
        ValueError,
        "Input gain is not within the required interval of (0, 16777216].",
    )
    test_invalid_input(
        -1.5,
        audio.GainType.AMPLITUDE,
        ValueError,
        "Input gain is not within the required interval of [0, 16777216].",
    )
    test_invalid_input(
        1.5,
        "TEST",
        TypeError,
        "Argument gain_type with value TEST is not of type [<enum 'GainType'>], but got <class 'str'>.",
    )


def test_vol_transform():
    """
    Feature: Vol
    Description: Test Vol with various valid input parameters and data types
    Expectation: The operation completes successfully
    """

    # Test vol
    waveform = np.random.randn(8, 5, 10, 10).astype(np.float16)
    dataset = ds.NumpySlicesDataset(waveform, column_names=["column1"])
    vol = audio.Vol(gain=1.2, gain_type=audio.GainType.POWER)
    dataset = dataset.map(operations=vol)
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (np.abs(data["column1"]) <= 1).all()

    # Test with 1D input (float64) using gain=0
    waveform = np.random.randn(10).astype(np.float64)
    vol = audio.Vol(gain=0)
    output = vol(waveform)
    assert (np.abs(output) <= 1).all()

    # Test with 3D input using gain=10
    waveform = np.random.randn(10, 10, 5)
    vol = audio.Vol(gain=10)
    output = vol(waveform)
    assert (np.abs(output) <= 1).all()

    # Test with 4D+ input using gain=16777216
    waveform = np.random.randn(5, 4, 4, 6)
    vol = audio.Vol(gain=16777216)
    vol(waveform)

    # Test with 1D input (float64) using gain=0.1
    waveform = np.random.randn(10).astype(np.float64)
    vol = audio.Vol(gain=0.1, gain_type=audio.GainType.POWER)
    output = vol(waveform)
    assert (np.abs(output) <= 1).all()

    # Test with custom array input using gain=0.6128
    waveform = np.array(
        [
            0.344000905,
            0.182902410626,
            -0.136728987017,
            -0.4887817312,
            0,
            2.854126,
            -103.0,
        ]
    )
    vol = audio.Vol(gain=0.6128, gain_type=audio.GainType.POWER)
    output = vol(waveform)
    allclose_nparray(
        output,
        [0.26928922, 0.14317883, -0.10703356, -0.38262588, 0, 1, -1],
        0.0001,
        0.0001,
    )

    # Test with 3D input (float16) using gain=20
    waveform = np.random.randn(3, 5, 5).astype(np.float16)
    vol = audio.Vol(gain=20, gain_type=audio.GainType.POWER)
    output = vol(waveform)
    assert (np.abs(output) <= 1).all()

    # Test with 2D input (float32) using gain=1616
    waveform = np.random.randn(5, 4).astype(np.float32)
    vol = audio.Vol(gain=1616, gain_type=audio.GainType.POWER)
    vol(waveform)

    # Test with 1D input (float64) using gain=0
    waveform = np.random.randn(10).astype(np.float64)
    vol = audio.Vol(gain=0, gain_type=audio.GainType.DB)
    output = vol(waveform)
    assert (np.abs(output) <= 1).all()

    # Test with custom array input using gain=1.3128
    waveform = np.array(
        [-1.2402375, 1.2109375, -1.34863125, -0.012646875, 0, 0.2536, -0.956, 0.84236]
    )
    vol = audio.Vol(gain=1.3128, gain_type=audio.GainType.DB)
    output = vol(waveform)
    allclose_nparray(
        output, [-1, 1, -1, -0.01471036, 0, 0.29497775, -1, 0.97980068], 0.0001, 0.0001
    )

    # Test with 3D input using gain=-20
    waveform = np.random.randint(-100, 100, (5, 5, 5))
    vol = audio.Vol(gain=-20, gain_type=audio.GainType.DB)
    output = vol(waveform)
    assert (np.abs(output) <= 1).all()


def test_vol_param_check():
    """
    Feature: Vol
    Description: Test Vol with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """

    # Test vol
    waveform = np.array(1.0)
    vol = audio.Vol(gain=20, gain_type=audio.GainType.DB)
    with pytest.raises(
        RuntimeError,
        match="the shape of input tensor does not match the requirement of operator. Exp"
        "ecting tensor in shape of <..., time>. But got tensor with dimension 0.",
    ):
        vol(waveform)

    # Test with invalid input format
    waveform = 0.85
    vol = audio.Vol(gain=20, gain_type=audio.GainType.DB)
    with pytest.raises(
        TypeError, match="Input should be NumPy audio, got <class 'float'>."
    ):
        vol(waveform)

    # Test with string input (invalid type)
    waveform = np.array(["1", "2"])
    vol = audio.Vol(gain=20, gain_type=audio.GainType.DB)
    with pytest.raises(
        RuntimeError,
        match="the data type of input tensor does not match the requirement of operator. "
        "Expecting tensor in type of \\[int, float, double\\]. But got type string",
    ):
        vol(waveform)

    # Test with invalid input format
    waveform = [1.0, 1.0, 1.0]
    vol = audio.Vol(gain=20, gain_type=audio.GainType.DB)
    with pytest.raises(
        TypeError, match="Input should be NumPy audio, got <class 'list'>."
    ):
        vol(waveform)

    # Test with exception using gain=20
    vol = audio.Vol(gain=20, gain_type=audio.GainType.DB)
    with pytest.raises(RuntimeError, match="Input Tensor is not valid."):
        vol()

    # Test with exception with custom array input using gain=1.1
    waveform = np.array([1.0, 1.0, 1.0])
    vol = audio.Vol(gain=1.1)
    with pytest.raises(
        RuntimeError, match="The op is OneToOne, can only accept one tensor as input"
    ):
        vol(waveform, waveform)

    # Test with invalid value using gain=-0.1
    with pytest.raises(
        ValueError,
        match="Input gain is not within the required interval of \\[0, 16777216\\].",
    ):
        audio.Vol(gain=-0.1)

    # Test with invalid value using gain=16777216.1
    with pytest.raises(
        ValueError,
        match="Input gain is not within the required interval of \\[0, 16777216\\].",
    ):
        audio.Vol(gain=16777216.1)

    # Test with invalid value using gain=16777216.1
    with pytest.raises(
        ValueError,
        match="Input gain is not within the required interval of \\(0, 16777216\\].",
    ):
        audio.Vol(gain=16777216.1, gain_type=audio.GainType.POWER)

    # Test with invalid value using gain=0
    with pytest.raises(
        ValueError,
        match="Input gain is not within the required interval of \\(0, 16777216\\].",
    ):
        audio.Vol(gain=0, gain_type=audio.GainType.POWER)

    # Test with invalid value using gain=16777216.1
    with pytest.raises(
        ValueError,
        match="Input gain is not within the required interval of \\[-16777216, 16777216\\].",
    ):
        audio.Vol(gain=16777216.1, gain_type=audio.GainType.DB)

    # Test with invalid value using gain=-16777216.1
    with pytest.raises(
        ValueError,
        match="Input gain is not within the required interval of \\[-16777216, 16777216\\].",
    ):
        audio.Vol(gain=-16777216.1, gain_type=audio.GainType.DB)

    # Test with string input (invalid type)
    with pytest.raises(
        TypeError,
        match="Argument gain with value 1.0 is not of type \\[<class"
        " 'int'>, <class 'float'>\\], but got <class 'str'>.",
    ):
        audio.Vol(gain="1.0", gain_type=audio.GainType.DB)

    # Test with invalid type using gain=True
    with pytest.raises(
        TypeError,
        match="Argument gain with value True is not of type \\(<class"
        " 'int'>, <class 'float'>\\), but got <class 'bool'>.",
    ):
        audio.Vol(gain=True, gain_type=audio.GainType.POWER)

    # Test with invalid type using gain=None
    with pytest.raises(
        TypeError,
        match="Argument gain with value None is not of type \\[<class"
        " 'int'>, <class 'float'>\\], but got <class 'NoneType'>.",
    ):
        audio.Vol(gain=None)

    # Test with invalid type
    with pytest.raises(
        TypeError,
        match="Argument gain with value \\[1.0\\] is not of type \\[<class"
        " 'int'>, <class 'float'>\\], but got <class 'list'>.",
    ):
        audio.Vol([1.0])

    # Test with invalid type
    with pytest.raises(
        TypeError,
        match="Argument gain_type with value <enum 'GainType'> is not"
        " of type \\[<enum 'GainType'>\\]",
    ):
        audio.Vol(1.0, audio.GainType)

    # Test with invalid type
    with pytest.raises(
        TypeError,
        match="Argument gain_type with value 1 is not of "
        "type \\[<enum 'GainType'>\\], but got <class 'int'>.",
    ):
        audio.Vol(1.0, 1)

    # Test with invalid type
    with pytest.raises(
        TypeError,
        match="Argument gain_type with value db is not of type"
        " \\[<enum 'GainType'>\\], but got <class 'str'>.",
    ):
        audio.Vol(1.0, "db")

    # Test with invalid type
    with pytest.raises(
        TypeError,
        match="Argument gain_type with value None is not of type"
        " \\[<enum 'GainType'>\\], but got <class 'NoneType'>.",
    ):
        audio.Vol(1.0, None)

    # Test with invalid type
    with pytest.raises(
        TypeError,
        match="Argument gain_type with value \\[<GainType.POWER: 'power'>\\] "
        "is not of type \\[<enum 'GainType'>\\], but got <class 'list'>.",
    ):
        audio.Vol(1.0, [audio.GainType.POWER])


if __name__ == "__main__":
    test_vol_eager()
    test_vol_pipeline()
    test_vol_invalid_input()
    test_vol_transform()
    test_vol_param_check()
