# Copyright 2021 Huawei Technologies Co., Ltd
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
"""
Testing FrequencyMasking op in DE.
"""

import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.audio.transforms as audio
from mindspore import log as logger

CHANNEL = 2
FREQ = 30
TIME = 30


def gen(shape):
    np.random.seed(0)
    data = np.random.random(shape)
    yield (np.array(data, dtype=np.float32),)


def count_unequal_element(data_expected, data_me, rtol, atol):
    """ Precision calculation func """
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_expected) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}".format(
        data_expected[greater], data_me[greater], error[greater])


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    """ Precision calculation formula  """
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan):
        count_unequal_element(data_expected, data_me, rtol, atol)


def test_func_frequency_masking_eager_random_input():
    """ mindspore eager mode normal testcase:frequency_masking op"""
    logger.info("test frequency_masking op")
    spectrogram = next(gen((CHANNEL, FREQ, TIME)))[0]
    out_put = audio.FrequencyMasking(False, 3, 1, 10)(spectrogram)
    assert out_put.shape == (CHANNEL, FREQ, TIME)


def test_func_frequency_masking_eager_precision():
    """ mindspore eager mode normal testcase:frequency_masking op"""
    logger.info("test frequency_masking op")
    spectrogram = np.array([[[0.17274511, 0.85174704, 0.07162686, -0.45436913],
                             [-1.045921, -1.8204843, 0.62333095, -0.09532598],
                             [1.8175547, -0.25779432, -0.58152324, -0.00221091]],
                            [[-1.205032, 0.18922766, -0.5277673, -1.3090396],
                             [1.8914849, -0.97001046, -0.23726775, 0.00525892],
                             [-1.0271876, 0.33526883, 1.7413973, 0.12313101]]]).astype(np.float32)
    out_ms = audio.FrequencyMasking(False, 2, 0, 0)(spectrogram)
    out_benchmark = np.array([[[0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0],
                               [1.8175547, -0.25779432, -0.58152324, -0.00221091]],
                              [[0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0],
                               [-1.0271876, 0.33526883, 1.7413973, 0.12313101]]]).astype(np.float32)
    allclose_nparray(out_ms, out_benchmark, 0.0001, 0.0001)


def test_func_frequency_masking_pipeline():
    """ mindspore pipeline mode normal testcase:frequency_masking op"""
    logger.info("test frequency_masking op, pipeline")

    generator = gen([CHANNEL, FREQ, TIME])
    data1 = ds.GeneratorDataset(source=generator, column_names=["multi_dimensional_data"])

    transforms = [audio.FrequencyMasking(True, 8)]
    data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])

    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        out_put = item["multi_dimensional_data"]
    assert out_put.shape == (CHANNEL, FREQ, TIME)


def test_frequency_masking_invalid_input():
    def test_invalid_param(test_name, iid_masks, frequency_mask_param, mask_start, error, error_msg):
        logger.info("Test FrequencyMasking with wrong params: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            audio.FrequencyMasking(iid_masks, frequency_mask_param, mask_start)
        assert error_msg in str(error_info.value)

    def test_invalid_input(test_name, iid_masks, frequency_mask_param, mask_start, error, error_msg):
        logger.info("Test FrequencyMasking with wrong params: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            spectrogram = next(gen((CHANNEL, FREQ, TIME)))[0]
            _ = audio.FrequencyMasking(iid_masks, frequency_mask_param, mask_start)(spectrogram)
        assert error_msg in str(error_info.value)

    test_invalid_param("invalid mask_start", True, 2, -10, ValueError,
                       "Input mask_start is not within the required interval of [0, 16777216].")
    test_invalid_param("invalid mask_param", True, -2, 10, ValueError,
                       "Input mask_param is not within the required interval of [0, 16777216].")
    test_invalid_param("invalid iid_masks", "True", 2, 10, TypeError,
                       "Argument iid_masks with value True is not of type [<class 'bool'>], but got <class 'str'>.")

    test_invalid_input("invalid mask_start", False, 2, 100, RuntimeError,
                       "MaskAlongAxis: mask_start should be less than the length of chosen dimension.")
    test_invalid_input("invalid mask_width", False, 200, 2, RuntimeError,
                       "FrequencyMasking: frequency_mask_param should be less than or equal to the length of " +
                       "frequency dimension.")


if __name__ == "__main__":
    test_func_frequency_masking_eager_random_input()
    test_func_frequency_masking_eager_precision()
    test_func_frequency_masking_pipeline()
    test_frequency_masking_invalid_input()
