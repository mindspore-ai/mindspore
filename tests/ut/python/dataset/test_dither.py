# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.audio as audio
from mindspore import log as logger
from mindspore.dataset.audio.utils import DensityFunction
from util import visualize_audio, diff_mse


def count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_expected) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
            format(data_expected[greater], data_me[greater], error[greater])


def test_dither_eager_noise_shaping_false():
    """
    Feature: Dither
    Description: Test Dither in eager mode
    Expectation: The result is as expected
    """
    logger.info("test Dither in eager mode")

    # Original waveform
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[0.99993896, 1.99990845, 2.99984741],
                                [3.99975586, 4.99972534, 5.99966431]], dtype=np.float64)
    dither_op = audio.Dither(DensityFunction.TPDF, False)
    # Filtered waveform by Dither
    output = dither_op(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_dither_eager_noise_shaping_true():
    """
    Feature: Dither
    Description: Test Dither in eager mode
    Expectation: The result is as expected
    """
    logger.info("test Dither in eager mode")

    # Original waveform
    waveform = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[0.9999, 1.9998, 2.9998],
                                [3.9998, 4.9995, 5.9994],
                                [6.9996, 7.9991, 8.9990]], dtype=np.float64)
    dither_op = audio.Dither(DensityFunction.TPDF, True)
    # Filtered waveform by Dither
    output = dither_op(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_dither_pipeline(plot=False):
    """
    Feature: Dither
    Description: Test Dither in pipeline mode
    Expectation: The result is as expected
    """
    logger.info("test Dither in pipeline mode")

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
        mse_rpdf = diff_mse(dither_rpdf, expect_rpdf[i, :])
        logger.info("dither_rpdf_{}, mse: {}".format(i + 1, mse_rpdf))
        mse_gpdf = diff_mse(dither_gpdf, expect_gpdf[i, :])
        logger.info("dither_gpdf_{}, mse: {}".format(i + 1, mse_gpdf))
        i += 1
        if plot:
            visualize_audio(dither_rpdf, expect_rpdf[i, :])
            visualize_audio(dither_gpdf, expect_gpdf[i, :])


def test_invalid_dither_input():
    """
    Feature: Dither
    Description: Test param check of Dither
    Expectation: Throw correct error and message
    """
    logger.info("test param check of Dither")

    def test_invalid_input(test_name, density_function, noise_shaping, error, error_msg):
        logger.info("Test Dither with bad input: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            audio.Dither(density_function, noise_shaping)
        assert error_msg in str(error_info.value)

    test_invalid_input("invalid density function parameter value", "TPDF", False, TypeError,
                       "Argument density_function with value TPDF is not of type "
                       + "[<enum 'DensityFunction'>], but got <class 'str'>.")

    test_invalid_input("invalid density_function parameter value", 6, False, TypeError,
                       "Argument density_function with value 6 is not of type "
                       + "[<enum 'DensityFunction'>], but got <class 'int'>.")

    test_invalid_input("invalid noise_shaping parameter value", DensityFunction.GPDF, 1, TypeError,
                       "Argument noise_shaping with value 1 is not of type [<class 'bool'>], but got <class 'int'>.")

    test_invalid_input("invalid noise_shaping parameter value", DensityFunction.RPDF, "true", TypeError,
                       "Argument noise_shaping with value true is not of type [<class 'bool'>], but got <class 'str'>")


if __name__ == '__main__':
    test_dither_eager_noise_shaping_false()
    test_dither_eager_noise_shaping_true()
    test_dither_pipeline(plot=False)
    test_invalid_dither_input()
