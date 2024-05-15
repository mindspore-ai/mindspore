# Copyright 2024 Huawei Technologies Co., Ltd
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
Testing decode_video
"""
import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import vision


def check_mindspore_data(mindspore_data, expected_data, error_rate_limit=0.05):
    """
    Check if result is as expected.

    Args:
        mindspore_data (tuple(numpy.ndarray, numpy.ndarray)): The result of DecodeVideo.
        expected_data (tuple(numpy.ndarray, float, numpy.ndarray, float)): The expected result.
        error_rate_limit (float, optional): The maximum error rate. Default: 0.05.
    """
    # Check the video dta
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


def test_decode_video_with_avi():
    """
    Feature: decode_video
    Description: Decode an AVI numpy.ndarray
    Expectation: The Output is equal to the expected output
    """
    filename = "../data/dataset/video/campus.avi"
    raw_ndarray = np.fromfile(filename, np.uint8)
    mindspore_output = vision.DecodeVideo()(raw_ndarray)
    expected_output = ((5, 160, 240, 3), 22301347, (2, 9216), -5.220537e-07)
    check_mindspore_data(mindspore_output, expected_output)


def test_decode_video_with_h264():
    """
    Feature: decode_video
    Description: Decode a H264 numpy.ndarray
    Expectation: The Output is equal to the expected output
    """
    filename = "../data/dataset/video/campus.h264"
    raw_ndarray = vision.read_file(filename)
    mindspore_output = vision.DecodeVideo()(raw_ndarray)
    expected_output = ((19, 480, 270, 3), 701517643, (2, 15360), 1.7901399)
    check_mindspore_data(mindspore_output, expected_output)


def test_decode_video_with_h265():
    """
    Feature: decode_video
    Description: Decode a H265 numpy.ndarrays
    Expectation: The Output is equal to the expected output
    """
    filename = "../data/dataset/video/campus.h265"
    raw_ndarray = np.fromfile(filename, np.uint8)
    mindspore_output = vision.DecodeVideo()(raw_ndarray)
    expected_output = ((1, 576, 720, 3), 48184768, (2, 4608), 0.0)
    check_mindspore_data(mindspore_output, expected_output)


def test_decode_video_with_mov():
    """
    Feature: decode_video
    Description: Decode a MOV numpy.ndarray
    Expectation: The Output is equal to the expected output
    """
    filename = "../data/dataset/video/campus.mov"
    raw_ndarray = np.fromfile(filename, np.uint8)
    mindspore_output = vision.DecodeVideo()(raw_ndarray)
    expected_output = ((5, 160, 240, 3), 22306864, (1, 9216), 0.0)
    check_mindspore_data(mindspore_output, expected_output)


def test_decode_video_with_mp4():
    """
    Feature: decode_video
    Description: Decode a MP4 numpy.ndarray
    Expectation: The Output is equal to the expected output
    """
    filename = "../data/dataset/video/campus.mp4"
    raw_ndarray = np.fromfile(filename, np.uint8)
    mindspore_output = vision.DecodeVideo()(raw_ndarray)
    expected_output = ((5, 160, 240, 3), 22296128, (1, 9216), 0.0)
    check_mindspore_data(mindspore_output, expected_output)


def test_decode_video_with_wmv():
    """
    Feature: decode_video
    Description: Decode a WMV numpy.ndarray
    Expectation: The Output is equal to the expected output
    """
    filename = "../data/dataset/video/campus.wmv"
    raw_ndarray = np.fromfile(filename, np.uint8)
    mindspore_output = vision.DecodeVideo()(raw_ndarray)
    expected_output = ((4, 576, 720, 3), 195420025, (2, 10240), 0.0)
    check_mindspore_data(mindspore_output, expected_output)


class VideoDataset:
    """Custom class to generate and read video dataset"""
    def __init__(self, file_list):
        self.file_list = file_list

    def __getitem__(self, index):
        filename = self.file_list[index]
        return np.fromfile(filename, np.uint8)

    def __len__(self):
        return len(self.file_list)


def test_decode_video_pipeline_custom_dataset():
    """
    Feature: decode_video
    Description: Decode a custom dataset
    Expectation: The Output is equal to the expected output
    """
    file_list = ["../data/dataset/video/campus.avi"]
    dataset = ds.GeneratorDataset(VideoDataset(file_list), ["data"])
    decode_video = vision.DecodeVideo()
    dataset = dataset.map(operations=[decode_video], input_columns=["data"], output_columns=["visual", "audio"])
    for visual, audio in dataset:
        assert visual.shape == (5, 160, 240, 3)
        assert audio.shape == (2, 9216)


def invalid_video(video, error, error_msg):
    """
    a function to check error and message with invalid parameter
    """
    with pytest.raises(error) as error_info:
        vision.DecodeVideo()(video)
    assert error_msg in str(error_info.value)


def test_decode_video_with_invalid_type_of_video():
    """
    Feature: decode_video
    Description: Decode a string.
    Expectation: Error is caught when the type of video is not one of [bytes, np.ndarray].
    """
    error_message = "Input should be ndarray, got"
    invalid_video("invalid_type_of_video", TypeError, error_message)


def test_decode_video_with_invalid_dimension_of_video():
    """
    Feature: decode_video
    Description: Decode a 2D.
    Expectation: Error is caught when the dimensions of video is 2.
    """
    video = np.ndarray(shape=(10, 10), dtype=np.uint8)
    error_message = "invalid input shape, only support 1D input, got rank: 2"
    invalid_video(video, RuntimeError, error_message)


def test_decode_video_one_param():
    """
    Feature: decode_video
    Description: Test decode_video with one parameter
    Expectation: Error is caught when there is one parameter
    """
    # Test with more than five parameter
    filename = "../data/dataset/video/campus.avi"
    raw_video = np.fromfile(filename, np.uint8)
    error_message = "__init__() takes 1 positional argument but 2 were given"
    with pytest.raises(TypeError) as error_info:
        vision.DecodeVideo("1")(raw_video)
    assert error_message in str(error_info.value)


def test_decode_video_with_invalid_data_elements():
    """
    Feature: decode_video
    Description: Decode a video contains float32 elements.
    Expectation: Error is caught when the video contains float32 elements.
    """
    video = np.ndarray(shape=(10), dtype=np.float32)
    error_message = "The type of the elements of input data should be UINT8, but got float32."
    invalid_video(video, RuntimeError, error_message)


def test_decode_video_with_empty():
    """
    Feature: decode_video
    Description: Decode a ndarray has no data.
    Expectation: Error is caught when there is no data.
    """
    empty_numpy = np.empty(0, dtype=np.uint8)
    error_message = "Input Tensor has no data."
    invalid_video(empty_numpy, RuntimeError, error_message)


if __name__ == "__main__":
    test_decode_video_with_avi()
    test_decode_video_with_h264()
    test_decode_video_with_h265()
    test_decode_video_with_mov()
    test_decode_video_with_mp4()
    test_decode_video_with_wmv()
    test_decode_video_pipeline_custom_dataset()
    test_decode_video_with_invalid_type_of_video()
    test_decode_video_with_invalid_dimension_of_video()
    test_decode_video_one_param()
    test_decode_video_with_invalid_data_elements()
    test_decode_video_with_empty()
