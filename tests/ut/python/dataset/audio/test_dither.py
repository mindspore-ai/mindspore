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
"""Test Dither."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from mindspore.dataset.audio import DensityFunction
from . import count_unequal_element


def test_dither_eager_noise_shaping_false():
    """
    Feature: Dither
    Description: Test Dither in eager mode
    Expectation: The result is as expected
    """
    # Original waveform
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[0.99993896, 1.99990845, 2.99984741],
                                [3.99975586, 4.99972534, 5.99966431]], dtype=np.float64)
    dither = audio.Dither(DensityFunction.TPDF, False)
    # Filtered waveform by Dither
    output = dither(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_dither_eager_noise_shaping_true():
    """
    Feature: Dither
    Description: Test Dither in eager mode
    Expectation: The result is as expected
    """
    # Original waveform
    waveform = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[0.9999, 1.9998, 2.9998],
                                [3.9998, 4.9995, 5.9994],
                                [6.9996, 7.9991, 8.9990]], dtype=np.float64)
    dither = audio.Dither(DensityFunction.TPDF, True)
    # Filtered waveform by Dither
    output = dither(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_dither_pipeline():
    """
    Feature: Dither
    Description: Test Dither in pipeline mode
    Expectation: The result is as expected
    """
    # Original waveform
    waveform_tpdf = np.array([[0.4941969, 0.53911686, 0.4846254], [0.10841596, 0.029320478, 0.52353495],
                              [0.23657, 0.087965, 0.43579]], dtype=np.float64)
    waveform_rpdf = np.array([[0.4941969, 0.53911686, 0.4846254], [0.10841596, 0.029320478, 0.52353495],
                              [0.23657, 0.087965, 0.43579]], dtype=np.float64)
    waveform_gpdf = np.array([[0.4941969, 0.53911686, 0.4846254], [0.10841596, 0.029320478, 0.52353495],
                              [0.23657, 0.087965, 0.43579]], dtype=np.float64)
    # Expect waveform
    expect_tpdf = np.array([[0.49417114, 0.53909302, 0.48461914],
                            [0.10839844, 0.02932739, 0.52352905],
                            [0.23654175, 0.08798218, 0.43579102]], dtype=np.float64)
    expect_rpdf = np.array([[0.4941, 0.5391, 0.4846],
                            [0.1084, 0.0293, 0.5235],
                            [0.2365, 0.0880, 0.4358]], dtype=np.float64)
    expect_gpdf = np.array([[0.4944, 0.5393, 0.4848],
                            [0.1086, 0.0295, 0.5237],
                            [0.2368, 0.0882, 0.4360]], dtype=np.float64)
    dataset_tpdf = ds.NumpySlicesDataset(waveform_tpdf, ["audio"], shuffle=False)
    dataset_rpdf = ds.NumpySlicesDataset(waveform_rpdf, ["audio"], shuffle=False)
    dataset_gpdf = ds.NumpySlicesDataset(waveform_gpdf, ["audio"], shuffle=False)

    # Filtered waveform by Dither of TPDF
    dither_tpdf = audio.Dither()
    dataset_tpdf = dataset_tpdf.map(input_columns=["audio"], operations=dither_tpdf, num_parallel_workers=2)

    # Filtered waveform by Dither of RPDF
    dither_rpdf = audio.Dither(DensityFunction.RPDF, False)
    dataset_rpdf = dataset_rpdf.map(input_columns=["audio"], operations=dither_rpdf, num_parallel_workers=2)

    # Filtered waveform by Dither of GPDF
    dither_gpdf = audio.Dither(DensityFunction.GPDF, False)
    dataset_gpdf = dataset_gpdf.map(input_columns=["audio"], operations=dither_gpdf, num_parallel_workers=2)

    i = 0
    for data1, data2, data3 in zip(dataset_tpdf.create_dict_iterator(output_numpy=True),
                                   dataset_rpdf.create_dict_iterator(output_numpy=True),
                                   dataset_gpdf.create_dict_iterator(output_numpy=True)):
        count_unequal_element(expect_tpdf[i, :], data1['audio'], 0.0001, 0.0001)
        dither_rpdf = data2['audio']
        dither_gpdf = data3['audio']
        count_unequal_element(dither_rpdf, expect_rpdf[i, :], 0.001, 0.001)
        count_unequal_element(dither_gpdf, expect_gpdf[i, :], 0.001, 0.001)
        i += 1


def test_invalid_dither_input():
    """
    Feature: Dither
    Description: Test param check of Dither
    Expectation: Throw correct error and message
    """
    def test_invalid_input(density_function, noise_shaping, error, error_msg):
        with pytest.raises(error) as error_info:
            audio.Dither(density_function, noise_shaping)
        assert error_msg in str(error_info.value)

    test_invalid_input("TPDF", False, TypeError,
                       "Argument density_function with value TPDF is not of type "
                       + "[<enum 'DensityFunction'>], but got <class 'str'>.")
    test_invalid_input(6, False, TypeError,
                       "Argument density_function with value 6 is not of type "
                       + "[<enum 'DensityFunction'>], but got <class 'int'>.")
    test_invalid_input(DensityFunction.GPDF, 1, TypeError,
                       "Argument noise_shaping with value 1 is not of type [<class 'bool'>], but got <class 'int'>.")
    test_invalid_input(DensityFunction.RPDF, "true", TypeError,
                       "Argument noise_shaping with value true is not of type [<class 'bool'>], but got <class 'str'>")


def test_dither_transform():
    """
    Feature: Dither
    Description: Test Dither with various valid input parameters and data types
    Expectation: The operation completes successfully
    """
    # test dither normal
    waveform = np.array([[0.8236, 0.2049, 0.3335], [0.5933, 0.9911, 0.2482],
                         [0.3007, 0.9054, 0.7598], [0.5394, 0.2842, 0.5634], [0.6363, 0.2226, 0.2288]])
    dither = audio.Dither(DensityFunction.TPDF, True)
    dataset1 = ds.NumpySlicesDataset(waveform, ["audio"], shuffle=False)
    dataset2 = ds.NumpySlicesDataset(waveform, ["audio"], shuffle=False)
    dataset2 = dataset2.map(input_columns=["audio"], operations=dither)
    for _, _ in zip(dataset1.create_dict_iterator(output_numpy=True),
                    dataset2.create_dict_iterator(output_numpy=True)):
        pass

    # test dither normal
    waveform = np.random.randn(10, 20, 6).astype(np.float32)
    dither = audio.Dither(DensityFunction.RPDF, True)
    dataset = ds.NumpySlicesDataset(waveform, ["audio"], shuffle=False)
    dataset = dataset.map(input_columns=["audio"], operations=dither)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test eager
    # Original waveform
    waveform = np.random.randn(50, ).astype(np.float64)
    dither = audio.Dither(DensityFunction.TPDF, False)
    dither(waveform)

    # Test eager
    waveform = np.random.randn(5, 4, 4, 8).astype(np.float16)
    dither = audio.Dither(DensityFunction.RPDF, False)
    dither(waveform)

    # Test eager
    waveform = np.random.randint(-10, 10, (10, 10, 5))
    dither = audio.Dither(DensityFunction.GPDF, False)
    dither(waveform)


def test_dither_param_check():
    """
    Feature: Dither
    Description: Test Dither with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """
    # Test eager
    dither = audio.Dither()
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'bool'>."):
        dither(True)

    # Test eager
    waveform = np.array(10)
    dither = audio.Dither()
    with pytest.raises(RuntimeError, match="Dither: the shape of input tensor does not match the requirement"
                                           " of operator. Expecting tensor in shape of <..., time>. "
                                           "But got tensor with dimension 0."):
        dither(waveform)

    # Test eager
    waveform = [1, 2, 3, 4]
    dither = audio.Dither()
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'list'>."):
        dither(waveform)

    # Test eager
    waveform = np.array(["1", "2", "3"])
    dither = audio.Dither()
    with pytest.raises(RuntimeError, match="Dither: the data type of input tensor does not match the"
                                           " requirement of operator. Expecting tensor in type of \\["
                                           "int, float, double\\]. But got type string."):
        dither(waveform)

    # Test eager
    dither = audio.Dither()
    with pytest.raises(RuntimeError, match="Input Tensor is not valid."):
        dither()

    # Test eager
    waveform = np.random.randn(5, 5, 3)
    dither = audio.Dither()
    with pytest.raises(RuntimeError, match="The op is OneToOne, can only accept one tensor as input."):
        dither(waveform, waveform)

    # Test eager
    with pytest.raises(TypeError, match=r"Argument density_function with value None is not of type "
                                        r"\[<enum 'DensityFunction'>\], but got <class 'NoneType'>."):
        audio.Dither(None)

    # Test with invalid type parameter (TypeError expected)
    with pytest.raises(TypeError, match=r"Argument density_function with value TPDF is not of type "
                                        r"\[<enum 'DensityFunction'>\], but got <class 'str'>."):
        audio.Dither("TPDF", False)

    # Test with invalid type parameter (TypeError expected)
    with pytest.raises(TypeError, match=r"Argument density_function with value 6 is not of type "
                                        r"\[<enum 'DensityFunction'>\], but got <class 'int'>."):
        audio.Dither(6, False)

    # Test with invalid type parameter (TypeError expected)
    with pytest.raises(TypeError, match="Argument noise_shaping with value 1 is not of "
                                        "type \\[<class 'bool'>\\], but got <class 'int'>."):
        audio.Dither(DensityFunction.GPDF, 1)

    # Test with invalid type parameter (TypeError expected)
    with pytest.raises(TypeError, match="Argument noise_shaping with value true is not of"
                                        " type \\[<class 'bool'>\\], but got <class 'str'>"):
        audio.Dither(DensityFunction.RPDF, "true")


if __name__ == '__main__':
    test_dither_eager_noise_shaping_false()
    test_dither_eager_noise_shaping_true()
    test_dither_pipeline()
    test_invalid_dither_input()
    test_dither_transform()
    test_dither_param_check()
