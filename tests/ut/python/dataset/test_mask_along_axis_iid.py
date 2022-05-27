# Copyright 2022 Huawei Technologies Co., Ltd
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
import copy
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.audio as audio
from mindspore import log as logger

BATCH = 2
CHANNEL = 2
FREQ = 10
TIME = 10


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    """
    Precision calculation formula
    """
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan):
        count_unequal_element(data_expected, data_me, rtol, atol)


def count_unequal_element(data_expected, data_me, rtol, atol):
    """
    Precision calculation func
    """
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_expected) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}".format(
        data_expected[greater], data_me[greater], error[greater])


def gen(shape):
    np.random.seed(0)
    data = np.random.random(shape)
    yield (np.array(data, dtype=np.float32),)


def test_mask_along_axis_iid_eager():
    """
    Feature: MaskAlongAxisIID
    Description: Mindspore eager mode with normal testcase
    Expectation: The returned result is as expected
    """
    logger.info("test MaskAlongAxisIID op, eager")
    spectrogram_01 = next(gen((BATCH, CHANNEL, FREQ, TIME)))[0]
    output_01 = audio.MaskAlongAxisIID(mask_param=8, mask_value=5.0, axis=1)(spectrogram_01)
    assert output_01.shape == (BATCH, CHANNEL, FREQ, TIME)

    spectrogram_02 = next(gen((BATCH, CHANNEL, FREQ, TIME)))[0]
    expect_output = copy.deepcopy(spectrogram_02)
    output_02 = audio.MaskAlongAxisIID(mask_param=0, mask_value=5.0, axis=1)(spectrogram_02)
    allclose_nparray(output_02, expect_output, 0.0001, 0.0001)


def test_mask_along_axis_iid_pipeline():
    """
    Feature: MaskAlongAxisIID
    Description: Mindspore pipeline mode with normal testcase
    Expectation: The returned result is as expected
    """
    logger.info("test MaskAlongAxisIID op, pipeline")

    generator = gen([BATCH, CHANNEL, FREQ, TIME])
    data1 = ds.GeneratorDataset(source=generator, column_names=["multi_dimensional_data"])

    transforms = [audio.MaskAlongAxisIID(mask_param=8, mask_value=5.0, axis=2)]
    data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])

    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        out_put = item["multi_dimensional_data"]
    assert out_put.shape == (BATCH, CHANNEL, FREQ, TIME)


def test_mask_along_axis_iid_invalid_input():
    """
    Feature: MaskAlongAxisIID
    Description: Mindspore eager mode with invalid input
    Expectation: The returned result is as expected
    """
    def test_invalid_param(test_name, mask_param, mask_value, axis, error, error_msg):
        """
        a function used for checking correct error and message
        """
        logger.info("Test MaskAlongAxisIID with wrong params: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            audio.MaskAlongAxisIID(mask_param, mask_value, axis)
        assert error_msg in str(error_info.value)

    test_invalid_param("invalid mask_param", 1.0, 1.0, 1, TypeError,
                       "Argument mask_param with value 1.0 is not of type [<class 'int'>], but got <class 'float'>.")
    test_invalid_param("invalid mask_param", -1, 1.0, 1, ValueError,
                       "Input mask_param is not within the required interval of [0, 2147483647].")
    test_invalid_param("invalid axis", 5, 1.0, 5.0, TypeError,
                       "Argument axis with value 5.0 is not of type [<class 'int'>], but got <class 'float'>.")
    test_invalid_param("invalid axis", 5, 1.0, 0, ValueError,
                       "Input axis is not within the required interval of [1, 2].")
    test_invalid_param("invalid axis", 5, 1.0, 3, ValueError,
                       "Input axis is not within the required interval of [1, 2].")


if __name__ == "__main__":
    test_mask_along_axis_iid_eager()
    test_mask_along_axis_iid_invalid_input()
    test_mask_along_axis_iid_pipeline()
