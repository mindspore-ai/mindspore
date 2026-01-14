# Copyright 2025 Huawei Technologies Co., Ltd
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
# ============================================================================

import os

import pytest

import mindspore.dataset as ds
from mindspore.dataset import vision
from tests.mark_utils import arg_mark

PWD = os.path.dirname(__file__)


@pytest.fixture(scope="function", autouse=True)
def set_backend():
    ds.config.set_video_backend("Ascend")
    yield
    ds.config.set_video_backend("CPU")


def check_mindspore_data(mindspore_data, expected_data, error_rate_limit=0.05):
    """
    Check mindspore_data with expected_data.

    Args:
        mindspore_data (tuple(numpy.ndarray, numpy.ndarray, Dict)): the data returned by read_video
            numpy.ndarray, four dimensions uint8 data for video. The format is [T, H, W, C].
            `T` is the number of frames, `H` is the height, `W` is the width, `C` is the channel for RGB.
            numpy.ndarray, two dimensions float for audio. The format is [K, L]. `K` is the number of channels.
            `L` is the length of the points.
            Dict, metadata for the video and audio. It contains video_fps(float), audio_fps(int).

        expected_data (tuple(numpy.ndarray, float, numpy.ndarray, float, float, int)): the generated data.
            numpy.ndarray, four dimensions uint8 data for video. The format is [T, H, W, C].
            `T` is the number of frames, `H` is the height, `W` is the width, `C` is the channel for RGB.
            float, the sum of the four dimensions uint8 data for video.
            numpy.ndarray, two dimensions float for audio. The format is [K, L]. `K` is the number of channels.
            `L` is the length of the points.
            float, the sum of the two dimensions float for audio.
            float, the video_fps.
            int, the audio_fps.

        error_rate_limit (float, optional): the maximum error rate. Default: 0.05.

    Expectation: Pass all the assets.
    """

    # Check the video data
    assert mindspore_data[0].shape == expected_data[0]
    if expected_data[1] > 0:
        assert abs(1.0 - mindspore_data[0].sum() / expected_data[1]) < error_rate_limit
    else:
        assert mindspore_data[0].sum() == 0

    # Check the audio data
    assert mindspore_data[1].shape == expected_data[2]
    if abs(expected_data[3]) > 1.0e-5:
        assert abs(1.0 - mindspore_data[1].sum() / expected_data[3]) < error_rate_limit
    else:
        assert abs(mindspore_data[1].sum()) <= 1.0e-5

    # Check the metadata: video_fps
    if expected_data[4] > 0:
        assert abs(1.0 - mindspore_data[2]["video_fps"] / expected_data[4]) < error_rate_limit
    else:
        assert mindspore_data[2]["video_fps"] == 0

    # Check the metadata: audio_fps
    assert int(mindspore_data[2]["audio_fps"]) == int(expected_data[5])


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=8.0)
def test_read_video_with_h264_pts():
    """
    Feature: read_video
    Description: Read a H264 file by "pts" as the pts_unit
    Expectation: The output is as expected
    """
    filename = PWD + "/data/campus.h264"
    mindspore_output = vision.read_video(filename, pts_unit="pts")
    expected_output = ((19, 480, 270, 3), 701517643, (2, 15360), 1.7901399, 30.0, 44100)
    check_mindspore_data(mindspore_output, expected_output)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=8.0)
def test_read_video_with_h264_sec():
    """
    Feature: read_video
    Description: Read a H264 file by "sec" as the pts_unit
    Expectation: The output is as expected
    """
    filename = PWD + "/data/campus.h264"
    mindspore_output = vision.read_video(filename, pts_unit="sec")
    expected_output = ((19, 480, 270, 3), 701517643, (2, 15360), 1.7901399, 30.0, 44100)
    check_mindspore_data(mindspore_output, expected_output)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=8.0)
def test_read_video_with_h264_start_pts_end_pts():
    """
    Feature: read_video
    Description: Read an H264 file by start_pts, end_pts, and "pts" as the pts_unit
    Expectation: The output is as expected
    """
    filename = PWD + "/data/campus.h264"
    # The start_pts is 0, end_pts is 0.7.
    mindspore_output = vision.read_video(filename, start_pts=0, end_pts=0.7, pts_unit="sec")
    expected_output = ((19, 480, 270, 3), 701517643, (2, 15360), 1.7901399, 30.0, 44100)
    check_mindspore_data(mindspore_output, expected_output)

    # The start_pts is 0, end_pts is 0.034.
    mindspore_output = vision.read_video(filename, start_pts=0, end_pts=0.034, pts_unit="sec")
    expected_output = ((2, 480, 270, 3), 73719664, (2, 2048), 0.0, 30.0, 44100)
    check_mindspore_data(mindspore_output, expected_output)

    # The start_pts is 0.1, end_pts is 0.2.
    mindspore_output = vision.read_video(filename, start_pts=0.1, end_pts=0.2, pts_unit="sec")
    expected_output = ((4, 480, 270, 3), 147439328, (2, 4728), 0.0, 30.0, 44100)
    check_mindspore_data(mindspore_output, expected_output)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_read_video_with_h265_pts():
    """
    Feature: read_video
    Description: Read a H265 file by "pts" as the pts_unit
    Expectation: The output is as expected
    """
    filename = PWD + "/data/campus.h265"
    mindspore_output = vision.read_video(filename, pts_unit="pts")
    expected_output = ((1, 576, 720, 3), 48184768, (2, 4608), 0.0, 25.0, 44100)
    check_mindspore_data(mindspore_output, expected_output)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_read_video_with_h265_sec():
    """
    Feature: read_video
    Description: Read a H265 file by "sec" as the pts_unit
    Expectation: The output is as expected
    """
    filename = PWD + "/data/campus.h265"
    mindspore_output = vision.read_video(filename, pts_unit="sec")
    expected_output = ((1, 576, 720, 3), 48184768, (2, 4608), 0.0, 25.0, 44100)
    check_mindspore_data(mindspore_output, expected_output)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_read_video_with_h265_start_pts_end_pts():
    """
    Feature: read_video
    Description: Read an H265 file by start_pts, end_pts, and "pts" as the pts_unit
    Expectation: The output is as expected
    """
    filename = PWD + "/data/campus.h265"
    # The start_pts is 0, end_pts is 8709.
    mindspore_output = vision.read_video(filename, start_pts=0, end_pts=8709, pts_unit="pts")
    expected_output = ((1, 576, 720, 3), 48184768, (2, 4608), 0.0, 25.0, 44100)
    check_mindspore_data(mindspore_output, expected_output)


def read_video_pipeline(independent_process, python_multiprocessing, start_method):
    os.environ['MS_INDEPENDENT_DATASET'] = independent_process
    if python_multiprocessing:
        ds.config.set_multiprocessing_start_method(start_method)

    class VideoDataset:
        def __init__(self):
            self.filename = PWD + "/data/campus.h265"

        def __getitem__(self, index):
            return vision.read_video(self.filename, pts_unit="sec")

        def __len__(self):
            return 10

    dataset = ds.GeneratorDataset(VideoDataset(), column_names=["video", "audio", "metadata"],
                                  num_parallel_workers=2, python_multiprocessing=python_multiprocessing)
    num_epochs = 3
    iterator = dataset.create_tuple_iterator(num_epochs=num_epochs)
    expected_output = ((1, 576, 720, 3), 48184768, (2, 4608), 0.0, 25.0, 44100)
    for _ in range(num_epochs):
        for data in iterator:
            check_mindspore_data(data, expected_output)

    if python_multiprocessing:
        ds.config.set_multiprocessing_start_method("fork")

    del os.environ['MS_INDEPENDENT_DATASET']


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=16.0)
def test_read_video_pipeline_multithreading():
    """
    Feature: read_video
    Description: Read video in GeneratorDataset with multithreading
    Expectation: The output is as expected
    """

    read_video_pipeline(independent_process="False", python_multiprocessing=False, start_method=None)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=12.0)
def test_read_video_pipeline_multiprocessing_fork():
    """
    Feature: read_video
    Description: Read video in GeneratorDataset with multiprocess in fork mode
    Expectation: The output is as expected
    """
    read_video_pipeline(independent_process="False", python_multiprocessing=True, start_method="fork")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=12.0)
def test_read_video_pipeline_multiprocessing_spawn():
    """
    Feature: read_video
    Description: Read video in GeneratorDataset with multiprocess in spawn mode
    Expectation: The output is as expected
    """
    read_video_pipeline(independent_process="False", python_multiprocessing=True, start_method="spawn")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=12.0)
def test_read_video_pipeline_multithreading_independent():
    """
    Feature: read_video
    Description: Read video in GeneratorDataset with multithreading in independent mode
    Expectation: The output is as expected
    """
    read_video_pipeline(independent_process="True", python_multiprocessing=False, start_method=None)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=12.0)
def test_read_video_pipeline_multiprocessing_fork_independent():
    """
    Feature: read_video
    Description: Read video in GeneratorDataset with multiprocess in fork and independent mode
    Expectation: The output is as expected
    """
    read_video_pipeline(independent_process="True", python_multiprocessing=True, start_method="fork")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=12.0)
def test_read_video_pipeline_multiprocessing_spawn_independent():
    """
    Feature: read_video
    Description: Read video in GeneratorDataset with multiprocess in fork and independent mode
    Expectation: The output is as expected
    """
    read_video_pipeline(independent_process="True", python_multiprocessing=True, start_method="spawn")
