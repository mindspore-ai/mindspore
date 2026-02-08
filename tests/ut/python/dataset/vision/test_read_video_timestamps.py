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
Testing read_video_timestamps
"""
import pytest

from mindspore.dataset import vision


def check_mindspore_data(mindspore_data, expected_data, error_rate_limit=0.0005):
    """
    Check the error rate between the mindspore_data and expected_data.

    Args:
        mindspore_data (tuple(list, float)): timestamps.
        expected_data (tuple(list, float)): expected timestamps.
        error_rate_limit (float, optional): the maximum error rate. Default: 0.0005.
    """
    assert len(mindspore_data[0]) == len(expected_data[0])

    pts_length = len(mindspore_data[0])

    mindspore_timestamp_sum = 0.0
    expected_timestamp_sum = 0

    for index in range(pts_length):
        mindspore_timestamp = (float)(mindspore_data[0][index])
        expected_timestamp = (float)(expected_data[0][index])
        mindspore_timestamp_sum += mindspore_timestamp
        expected_timestamp_sum += expected_timestamp

    difference = abs(mindspore_timestamp_sum - expected_timestamp_sum)
    if expected_timestamp_sum > 0:
        pts_error_rate = difference / expected_timestamp_sum
    else:
        pts_error_rate = difference

    difference = abs(mindspore_data[1] - expected_data[1])
    if expected_data[1] > 1.0e-5:
        fps_error_rate = difference / expected_data[1]
    else:
        fps_error_rate = difference

    assert pts_error_rate <= error_rate_limit
    assert fps_error_rate <= error_rate_limit


def test_read_video_timestamps_with_avi_pts():
    """
    Feature: read_video_timestamps
    Description: Read the timestamps of a AVI Video file by "pts" as the pts_unit
    Expectation: The output list contains 5 integer elements
    """
    filename = "../data/dataset/video/campus.avi"
    mindspore_output = vision.read_video_timestamps(filename)
    expected_fps = 29.97003
    expected_output = (list(range(5)), expected_fps)
    check_mindspore_data(mindspore_output, expected_output)


def test_read_video_timestamps_with_avi_sec():
    """
    Feature: read_video_timestamps
    Description: Read the timestamps of a AVI Video file by "sec" as the pts_unit
    Expectation: The output list contains 5 float elements
    """
    filename = "../data/dataset/video/campus.avi"
    mindspore_output = vision.read_video_timestamps(filename, "sec")
    expected_fps = 29.97003
    expected_output = (list(x / expected_fps for x in range(5)), expected_fps)
    check_mindspore_data(mindspore_output, expected_output)


def test_read_video_timestamps_with_h264_pts():
    """
    Feature: read_video_timestamps
    Description: Read the timestamps of a H264 Video file by "pts" as the pts_unit
    Expectation: The output list contains 19 integer elements
    """
    filename = "../data/dataset/video/campus.h264"
    mindspore_output = vision.read_video_timestamps(filename, "pts")
    expected_fps = 30.0
    expected_output = (list(x * 512 for x in range(19)), expected_fps)
    check_mindspore_data(mindspore_output, expected_output)


def test_read_video_timestamps_with_h264_sec():
    """
    Feature: read_video_timestamps
    Description: Read the timestamps of a H264 Video file by "sec" as the pts_unit
    Expectation: The output list contains 19 float elements
    """
    filename = "../data/dataset/video/campus.h264"
    mindspore_output = vision.read_video_timestamps(filename, "sec")
    expected_fps = 30.0
    expected_output = (list(x / expected_fps for x in range(19)), expected_fps)
    check_mindspore_data(mindspore_output, expected_output)


def test_read_video_timestamps_with_h265_pts():
    """
    Feature: read_video_timestamps
    Description: Read the timestamps of a H265 Video file by "pts" as the pts_unit
    Expectation: The output list contains 1 integer element
    """
    filename = "../data/dataset/video/campus.h265"
    mindspore_output = vision.read_video_timestamps(filename, "pts")
    expected_fps = 25.0
    expected_output = (list((x+1) * 5110 for x in range(1)), expected_fps)
    check_mindspore_data(mindspore_output, expected_output)


def test_read_video_timestamps_with_h265_sec():
    """
    Feature: read_video_timestamps
    Description: Read the timestamps of a H265 Video file by "sec" as the pts_unit
    Expectation: The output list contains 1 float element
    """
    filename = "../data/dataset/video/campus.h265"
    mindspore_output = vision.read_video_timestamps(filename, "sec")
    expected_fps = 25.0
    expected_output = (list((x+1) * 5110 / (expected_fps * 3600.0) for x in range(1)), expected_fps)
    check_mindspore_data(mindspore_output, expected_output)


def test_read_video_timestamps_with_mov_pts():
    """
    Feature: read_video_timestamps
    Description: Read the timestamps of a MOV Video file by "pts" as the pts_unit
    Expectation: The output list contains 5 integer elements
    """
    filename = "../data/dataset/video/campus.mov"
    mindspore_output = vision.read_video_timestamps(filename, "pts")
    expected_fps = 25.0
    expected_output = (list(x * 512 for x in range(5)), expected_fps)
    check_mindspore_data(mindspore_output, expected_output)


def test_read_video_timestamps_with_mov_sec():
    """
    Feature: read_video_timestamps
    Description: Read the timestamps of a MOV Video file by "sec" as the pts_unit
    Expectation: The output list contains 5 float elements
    """
    filename = "../data/dataset/video/campus.mov"
    mindspore_output = vision.read_video_timestamps(filename, "sec")
    expected_fps = 25.0
    expected_output = (list(x / expected_fps for x in range(5)), expected_fps)
    check_mindspore_data(mindspore_output, expected_output)


def test_read_video_timestamps_with_mp4_pts():
    """
    Feature: read_video_timestamps
    Description: Read the timestamps of a MP4 Video file by "pts" as the pts_unit
    Expectation: The output list contains 5 integer elements
    """
    filename = "../data/dataset/video/campus.mp4"
    mindspore_output = vision.read_video_timestamps(filename, "pts")
    expected_fps = 25.0
    expected_output = (list(x * 512 for x in range(5)), expected_fps)
    check_mindspore_data(mindspore_output, expected_output)


def test_read_video_timestamps_with_mp4_sec():
    """
    Feature: read_video_timestamps
    Description: Read the timestamps of a MP4 Video file by "sec" as the pts_unit
    Expectation: The output list contains 5 float elements
    """
    filename = "../data/dataset/video/campus.mp4"
    mindspore_output = vision.read_video_timestamps(filename, "sec")
    expected_fps = 25.0
    expected_output = (list(x / expected_fps for x in range(5)), expected_fps)
    check_mindspore_data(mindspore_output, expected_output)


def test_read_video_timestamps_with_wmv_pts():
    """
    Feature: read_video_timestamps
    Description: Read the timestamps of a MMV Video file by "pts" as the pts_unit
    Expectation: The output list contains 4 integer elements
    """
    filename = "../data/dataset/video/campus.wmv"
    mindspore_output = vision.read_video_timestamps(filename, "pts")
    expected_fps = 25.0
    expected_output = (list(x * 40.0 for x in range(4)), expected_fps)
    check_mindspore_data(mindspore_output, expected_output)


def test_read_video_timestamps_with_wmv_sec():
    """
    Feature: read_video_timestamps
    Description: Read the timestamps of a MP4 Video file by "sec" as the pts_unit
    Expectation: The output list contains 4 float elements
    """
    filename = "../data/dataset/video/campus.wmv"
    mindspore_output = vision.read_video_timestamps(filename, "sec")
    expected_fps = 25.0
    expected_output = (list(x / expected_fps for x in range(4)), expected_fps)
    check_mindspore_data(mindspore_output, expected_output)


def invalid_param(filename_param, pts_unit_param, error, error_msg):
    """
    a function used for checking correct error and message with invalid parameter
    """
    with pytest.raises(error) as error_info:
        vision.read_video_timestamps(filename_param, pts_unit_param)
    assert error_msg in str(error_info.value)


def test_read_video_timestamps_param_filename_invalid_type():
    """
    Feature: read_video_timestamps
    Description: Test read_video_timestamps with invalid parameter
    Expectation: Exception is raised as expected
    """
    # Test with an invalid type for the filename
    error_message = "Input filename is not of type"
    invalid_param(0, "pts", TypeError, error_message)


def test_read_video_timestamps_param_filename_is_directory():
    """
    Feature: read_video_timestamps
    Description: Test read_video_timestamps with invalid parameter
    Expectation: Exception is raised as expected
    """
    # Test with a directory name
    wrong_filename = "../data/dataset/"
    error_message = "Invalid file path, " + wrong_filename + " is not a regular file."
    invalid_param(wrong_filename, "pts", RuntimeError, error_message)


def test_read_video_timestamps_param_filename_not_exist():
    """
    Feature: read_video_timestamps
    Description: Test read_video_timestamps with invalid parameter
    Expectation: Exception is raised as expected
    """
    # Test with a not exist filename
    wrong_filename = "this_file_is_not_exist"
    error_message = "Invalid file path, " + wrong_filename + " does not exist."
    invalid_param(wrong_filename, "pts", RuntimeError, error_message)


def test_read_video_timestamps_param_filename_not_supported():
    """
    Feature: read_video_timestamps
    Description: Test read_video_timestamps with invalid parameter
    Expectation: Exception is raised as expected
    """
    # Test with a not supported gif file
    wrong_filename = "../data/dataset/declient.cfg"
    error_message = "Failed to open the file " + wrong_filename
    invalid_param(wrong_filename, "pts", RuntimeError, error_message)


def test_read_video_timestamps_param_pts_unit_invalid_type():
    """
    Feature: read_video_timestamps
    Description: Test read_video_timestamps with invalid parameter
    Expectation: Exception is raised as expected
    """
    # Test with an invalid type for the pts_unit
    filename = "../data/dataset/video/campus.avi"
    error_message = "Input pts_unit is not of type"
    invalid_param(filename, 0, TypeError, error_message)


def test_read_video_timestamps_param_pts_unit_not_supported():
    """
    Feature: read_video_timestamps
    Description: Test read_video_timestamps with invalid parameter
    Expectation: Exception is raised as expected
    """
    # Test with an invalid type for the pts_unit
    filename = "../data/dataset/video/campus.avi"
    error_message = "Not supported pts_unit"
    invalid_param(filename, "min", RuntimeError, error_message)


def test_read_video_timestamps_no_param():
    """
    Feature: read_video_timestamps
    Description: Test read_video_timestamps without parameter
    Expectation: Exception is raised as expected
    """
    error_message = "read_video_timestamps() missing 1 required positional argument: 'filename'"
    with pytest.raises(TypeError) as error_info:
        vision.read_video_timestamps()
    assert error_message in str(error_info.value)


def test_read_video_timestamps_three_params():
    """
    Feature: read_video_timestamps
    Description: Test read_video_timestamps more than two parameters
    Expectation: Exception is raised as expected
    """
    error_message = "read_video_timestamps() takes from 1 to 2 positional arguments but 3 were given"
    with pytest.raises(TypeError) as error_info:
        vision.read_video_timestamps("1", "2", 3)
    assert error_message in str(error_info.value)


if __name__ == "__main__":
    test_read_video_timestamps_with_avi_pts()
    test_read_video_timestamps_with_avi_sec()
    test_read_video_timestamps_with_h264_pts()
    test_read_video_timestamps_with_h264_sec()
    test_read_video_timestamps_with_h265_pts()
    test_read_video_timestamps_with_h265_sec()
    test_read_video_timestamps_with_mov_pts()
    test_read_video_timestamps_with_mov_sec()
    test_read_video_timestamps_with_mp4_pts()
    test_read_video_timestamps_with_mp4_sec()
    test_read_video_timestamps_with_wmv_pts()
    test_read_video_timestamps_with_wmv_sec()
    test_read_video_timestamps_param_filename_invalid_type()
    test_read_video_timestamps_param_filename_is_directory()
    test_read_video_timestamps_param_filename_not_exist()
    test_read_video_timestamps_param_filename_not_supported()
    test_read_video_timestamps_param_pts_unit_invalid_type()
    test_read_video_timestamps_param_pts_unit_not_supported()
    test_read_video_timestamps_no_param()
    test_read_video_timestamps_three_params()
