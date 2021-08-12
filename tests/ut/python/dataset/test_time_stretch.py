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
Testing TimeStretch op in DE
"""
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.audio.transforms as c_audio
from mindspore import log as logger

CHANNEL_NUM = 2
FREQ = 1025
FRAME_NUM = 300
COMPLEX = 2


def gen(shape):
    np.random.seed(0)
    data = np.random.random(shape)
    yield (np.array(data, dtype=np.float32),)


def count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_expected) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}".format(
        data_expected[greater], data_me[greater], error[greater])


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan):
        count_unequal_element(data_expected, data_me, rtol, atol)


def test_time_stretch_pipeline():
    """
    Test TimeStretch op. Pipeline.
    """
    logger.info("test TimeStretch op")
    generator = gen([CHANNEL_NUM, FREQ, FRAME_NUM, COMPLEX])
    data1 = ds.GeneratorDataset(source=generator, column_names=["multi_dimensional_data"])

    transforms = [c_audio.TimeStretch(512, FREQ, 1.3)]
    data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])

    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        out_put = item["multi_dimensional_data"]
    assert out_put.shape == (CHANNEL_NUM, FREQ, np.ceil(FRAME_NUM / 1.3), COMPLEX)


def test_time_stretch_pipeline_invalid_param():
    """
    Test TimeStretch op. Set invalid param. Pipeline.
    """
    logger.info("test TimeStretch op with invalid values")
    generator = gen([CHANNEL_NUM, FREQ, FRAME_NUM, COMPLEX])
    data1 = ds.GeneratorDataset(source=generator, column_names=["multi_dimensional_data"])

    with pytest.raises(ValueError, match=r"Input fixed_rate is not within the required interval of \(0, 16777216\]."):
        transforms = [c_audio.TimeStretch(512, FREQ, -1.3)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])

        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            out_put = item["multi_dimensional_data"]
        assert out_put.shape == (CHANNEL_NUM, FREQ, np.ceil(FRAME_NUM / 1.3), COMPLEX)


def test_time_stretch_eager():
    """
    Test TimeStretch op. Set param. Eager.
    """
    logger.info("test TimeStretch op with customized parameter values")
    spectrogram = next(gen([CHANNEL_NUM, FREQ, FRAME_NUM, COMPLEX]))[0]
    out_put = c_audio.TimeStretch(512, FREQ, 1.3)(spectrogram)
    assert out_put.shape == (CHANNEL_NUM, FREQ, np.ceil(FRAME_NUM / 1.3), COMPLEX)


def test_percision_time_stretch_eager():
    """
    Test TimeStretch op. Compare precision. Eager.
    """
    logger.info("test TimeStretch op with default values")
    spectrogram = np.array([[[[1.0402449369430542, 0.3807601034641266],
                              [-1.120057225227356, -0.12819576263427734],
                              [1.4303032159805298, -0.08839055150747299]],
                             [[1.4198592901229858, 0.6900091767311096],
                              [-1.8593409061431885, 0.16363371908664703],
                              [-2.3349387645721436, -1.4366451501846313]]],
                            [[[-0.7083967328071594, 0.9325454831123352],
                              [-1.9133838415145874, 0.011225821450352669],
                              [1.477278232574463, -1.0551637411117554]],
                             [[-0.6668586134910583, -0.23143270611763],
                              [-2.4390718936920166, 0.17638640105724335],
                              [-0.4795735776424408, 0.1345423310995102]]]]).astype(np.float64)
    out_expect = np.array([[[[1.0402449369430542, 0.3807601034641266],
                             [-1.302264928817749, -0.1490504890680313]],
                            [[1.4198592901229858, 0.6900091767311096],
                             [-2.382312774658203, 0.2096325159072876]]],
                           [[[-0.7083966732025146, 0.9325454831123352],
                             [-1.8545820713043213, 0.010880803689360619]],
                            [[-0.6668586134910583, -0.23143276572227478],
                             [-1.2737033367156982, 0.09211209416389465]]]]).astype(np.float64)
    out_ms = c_audio.TimeStretch(64, 2, 1.6)(spectrogram)

    allclose_nparray(out_ms, out_expect, 0.001, 0.001)


if __name__ == '__main__':
    test_time_stretch_pipeline()
    test_time_stretch_pipeline_invalid_param()
    test_time_stretch_eager()
    test_percision_time_stretch_eager()
