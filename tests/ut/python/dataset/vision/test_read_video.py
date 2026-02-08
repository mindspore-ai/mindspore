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
Testing read_video
"""
import pytest

from mindspore.dataset import vision


def check_mindspore_data(mindspore_data, expected_data, error_rate_limit=0.05):
    """
    Description: Check mindspore_data with expected_data.
    Args:
        mindspore_data (tuple(numpy.ndarray, numpy.ndarray, Dict)): the data returned by read_video
        numpy.ndarray, four dimensions uint8 data for video. The format is [T, H, W, C]. `T` is the number of frames,
            `H` is the height, `W` is the width, `C` is the channel for RGB.
        numpy.ndarray, two dimensions float for audio. The format is [K, L]. `K` is the number of channels.
            `L` is the length of the points.
        Dict, metadata for the video and audio. It contains video_fps(float), audio_fps(int).

        expected_data (tuple(numpy.ndarray, float, numpy.ndarray, float, float, int)): the generated data.
        numpy.ndarray, four dimensions uint8 data for video. The format is [T, H, W, C]. `T` is the number of frames,
            `H` is the height, `W` is the width, `C` is the channel for RGB.
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


def test_read_video_with_avi_pts():
    """
    Feature: read_video
    Description: Read an AVI file by "pts" as the pts_unit
    Expectation: The Output is equal to the expected output
    """
    filename = "../data/dataset/video/campus.avi"

    mindspore_output = vision.read_video(filename, pts_unit="pts")
    expected_output = ((5, 160, 240, 3), 22301347, (2, 9216), -5.220537e-07, 29.97003, 48000)
    check_mindspore_data(mindspore_output, expected_output)


def test_read_video_with_avi_sec():
    """
    Feature: read_video
    Description: Read an AVI file by "sec" as the pts_unit
    Expectation: The Output is equal to the expected output
    """
    filename = "../data/dataset/video/campus.avi"
    mindspore_output = vision.read_video(filename, pts_unit="sec")
    expected_output = ((5, 160, 240, 3), 22301347, (2, 9216), -5.220537e-07, 29.97003, 48000)
    check_mindspore_data(mindspore_output, expected_output)


def test_read_video_with_avi_start_pts_end_pts():
    """
    Feature: read_video
    Description: Read an AVI file by start_pts, end_pts, and "pts" as the pts_unit
    Expectation: The Output is equal to the expected output
    """
    filename = "../data/dataset/video/campus.avi"
    # The start_pts is 0, end_pts is 4.
    mindspore_output = vision.read_video(filename, start_pts=0, end_pts=4, pts_unit="pts")
    expected_output = ((5, 160, 240, 3), 22301347, (2, 9216), -5.220537e-07, 29.97003, 48000)
    check_mindspore_data(mindspore_output, expected_output)

    # The start_pts is 0, end_pts is 0.
    mindspore_output = vision.read_video(filename, start_pts=0, end_pts=0, pts_unit="pts")
    expected_output = ((1, 160, 240, 3), 4453162, (2, 1843), 1.814212e-06, 29.97003, 48000)
    check_mindspore_data(mindspore_output, expected_output)

    # The start_pts is 1, end_pts is 2.
    mindspore_output = vision.read_video(filename, start_pts=1, end_pts=2, pts_unit="pts")
    expected_output = ((2, 160, 240, 3), 8916279, (2, 3686), -1.2559944e-06, 29.97003, 48000)
    check_mindspore_data(mindspore_output, expected_output)


def test_read_video_with_h264_pts():
    """
    Feature: read_video
    Description: Read a H264 file by "pts" as the pts_unit
    Expectation: The Output is equal to the expected output
    """
    filename = "../data/dataset/video/campus.h264"
    mindspore_output = vision.read_video(filename, pts_unit="pts")
    expected_output = ((19, 480, 270, 3), 701517643, (2, 15360), 1.7901399, 30.0, 44100)
    check_mindspore_data(mindspore_output, expected_output)


def test_read_video_with_h264_sec():
    """
    Feature: read_video
    Description: Read a H264 file by "sec" as the pts_unit
    Expectation: The Output is equal to the expected output
    """
    filename = "../data/dataset/video/campus.h264"
    mindspore_output = vision.read_video(filename, pts_unit="sec")
    expected_output = ((19, 480, 270, 3), 701517643, (2, 15360), 1.7901399, 30.0, 44100)
    check_mindspore_data(mindspore_output, expected_output)


def test_read_video_with_h264_start_pts_end_pts():
    """
    Feature: read_video
    Description: Read an H264 file by start_pts, end_pts, and "pts" as the pts_unit
    Expectation: The Output is equal to the expected output
    """
    filename = "../data/dataset/video/campus.h264"
    # The start_pts is 0, end_pts is 0.7.
    mindspore_output = vision.read_video(filename, start_pts=0, end_pts=0.7, pts_unit="sec")
    expected_output = ((19, 480, 270, 3), 701517643, (2, 15360), 1.7901399, 30.0, 44100)
    check_mindspore_data(mindspore_output, expected_output)

    # The start_pts is 0, end_pts is 0.034.
    mindspore_output = vision.read_video(filename, start_pts=0, end_pts=0.034, pts_unit="sec")
    expected_output = ((2, 480, 270, 3), 73719664, (2, 1617), 0.0, 30.0, 44100)
    check_mindspore_data(mindspore_output, expected_output)

    # The start_pts is 0.1, end_pts is 0.2.
    mindspore_output = vision.read_video(filename, start_pts=0.1, end_pts=0.2, pts_unit="sec")
    expected_output = ((4, 480, 270, 3), 147439328, (2, 3234), 0.0, 30.0, 44100)
    check_mindspore_data(mindspore_output, expected_output)


def test_read_video_with_h265_pts():
    """
    Feature: read_video
    Description: Read a H265 file by "pts" as the pts_unit
    Expectation: The Output is equal to the expected output
    """
    filename = "../data/dataset/video/campus.h265"
    mindspore_output = vision.read_video(filename, pts_unit="pts")
    expected_output = ((1, 576, 720, 3), 48184768, (2, 4608), 0.0, 25.0, 44100)
    check_mindspore_data(mindspore_output, expected_output)


def test_read_video_with_h265_sec():
    """
    Feature: read_video
    Description: Read a H265 file by "sec" as the pts_unit
    Expectation: The Output is equal to the expected output
    """
    filename = "../data/dataset/video/campus.h265"
    mindspore_output = vision.read_video(filename, pts_unit="sec")
    expected_output = ((1, 576, 720, 3), 48184768, (2, 4608), 0.0, 25.0, 44100)
    check_mindspore_data(mindspore_output, expected_output)


def test_read_video_with_h265_start_pts_end_pts():
    """
    Feature: read_video
    Description: Read an H265 file by start_pts, end_pts, and "pts" as the pts_unit
    Expectation: The Output is equal to the expected output
    """
    filename = "../data/dataset/video/campus.h265"
    # The start_pts is 0, end_pts is 8709.
    mindspore_output = vision.read_video(filename, start_pts=0, end_pts=8709, pts_unit="pts")
    expected_output = ((1, 576, 720, 3), 48184768, (2, 4608), 0.0, 25.0, 44100)
    check_mindspore_data(mindspore_output, expected_output)


def test_read_video_with_mov_pts():
    """
    Feature: read_video
    Description: Read a MOV file by "pts" as the pts_unit
    Expectation: The Output is equal to the expected output
    """
    filename = "../data/dataset/video/campus.mov"
    mindspore_output = vision.read_video(filename, pts_unit="pts")
    expected_output = ((5, 160, 240, 3), 22306864, (1, 9216), 0.0, 25.0, 44100)
    check_mindspore_data(mindspore_output, expected_output)


def test_read_video_with_mov_sec():
    """
    Feature: read_video
    Description: Read a MOV file by "sec" as the pts_unit
    Expectation: The Output is equal to the expected output
    """
    filename = "../data/dataset/video/campus.mov"
    mindspore_output = vision.read_video(filename, pts_unit="sec")
    expected_output = ((5, 160, 240, 3), 22306864, (1, 9216), 0.0, 25.0, 44100)
    check_mindspore_data(mindspore_output, expected_output)


def test_read_video_with_mov_start_pts_end_pts():
    """
    Feature: read_video
    Description: Read a MOV file by start_pts, end_pts, and "pts" as the pts_unit
    Expectation: The Output is equal to the expected output
    """
    filename = "../data/dataset/video/campus.mov"
    # The start_pts is 0, end_pts is 4.
    mindspore_output = vision.read_video(filename, start_pts=0, end_pts=4, pts_unit="pts")
    expected_output = ((1, 160, 240, 3), 4453814, (1, 1843), 0.0, 25.0, 44100)
    check_mindspore_data(mindspore_output, expected_output)


def test_read_video_with_mp4_pts():
    """
    Feature: read_video
    Description: Read a MP4 file by "pts" as the pts_unit
    Expectation: The Output is equal to the expected output
    """
    filename = "../data/dataset/video/campus.mp4"
    mindspore_output = vision.read_video(filename, pts_unit="pts")
    expected_output = ((5, 160, 240, 3), 22296128, (1, 9216), 0.0, 25.0, 44100)
    check_mindspore_data(mindspore_output, expected_output)


def test_read_video_with_mp4_sec():
    """
    Feature: read_video
    Description: Read a MP4 file by "sec" as the pts_unit
    Expectation: The Output is equal to the expected output
    """
    filename = "../data/dataset/video/campus.mp4"
    mindspore_output = vision.read_video(filename, pts_unit="sec")
    expected_output = ((5, 160, 240, 3), 22296128, (1, 9216), 0.0, 25.0, 44100)
    check_mindspore_data(mindspore_output, expected_output)


def test_read_video_with_mp4_start_pts_end_pts():
    """
    Feature: read_video
    Description: Read a MP4 file by start_pts, end_pts, and "sec" as the pts_unit
    Expectation: The Output is equal to the expected output
    """
    filename = "../data/dataset/video/campus.mp4"
    # The start_pts is 0, end_pts is 1.
    mindspore_output = vision.read_video(filename, start_pts=0, end_pts=1, pts_unit="sec")
    expected_output = ((5, 160, 240, 3), 22296128, (1, 9216), 0.0, 25.0, 44100)
    check_mindspore_data(mindspore_output, expected_output)


def test_read_video_with_wmv_pts():
    """
    Feature: read_video
    Description: Read a WMV file by "pts" as the pts_unit
    Expectation: The Output is equal to the expected output
    """
    filename = "../data/dataset/video/campus.wmv"
    mindspore_output = vision.read_video(filename, pts_unit="pts")
    expected_output = ((4, 576, 720, 3), 195420025, (2, 10240), 0.0, 25.0, 48000)
    check_mindspore_data(mindspore_output, expected_output)


def test_read_video_with_wmv_sec():
    """
    Feature: read_video
    Description: Read a WMV file by "sec" as the pts_unit
    Expectation: The Output is equal to the expected output
    """
    filename = "../data/dataset/video/campus.wmv"
    mindspore_output = vision.read_video(filename, pts_unit="sec")
    expected_output = ((4, 576, 720, 3), 195420025, (2, 10240), 0.0, 25.0, 48000)
    check_mindspore_data(mindspore_output, expected_output)


def test_read_video_with_wmv_start_pts_end_pts():
    """
    Feature: read_video
    Description: Read a WMV file by start_pts, end_pts, and "sec" as the pts_unit
    Expectation: The Output is equal to the expected output
    """
    filename = "../data/dataset/video/campus.wmv"
    # The start_pts is 0, end_pts is 10.
    mindspore_output = vision.read_video(filename, start_pts=0, end_pts=None, pts_unit="sec")
    expected_output = ((4, 576, 720, 3), 195420025, (2, 10240), 0.0, 25.0, 48000)
    check_mindspore_data(mindspore_output, expected_output)


def invalid_param(filename_param, start_pts_param, end_pts_param, pts_unit_param, error, error_msg):
    """
    a function to check error and message with invalid parameter
    """
    with pytest.raises(error) as error_info:
        vision.read_video(filename_param, start_pts=start_pts_param, end_pts=end_pts_param, pts_unit=pts_unit_param)
    assert error_msg in str(error_info.value)


def test_read_video_param_filename_invalid_type():
    """
    Feature: read_video
    Description: Test read_video with invalid parameter
    Expectation: Error is caught when the parameter is invalid
    """
    # Test with an invalid type for the filename
    error_message = "Input filename is not of type"
    invalid_param(0, 0, None, "pts", TypeError, error_message)


def test_read_video_param_filename_is_directory():
    """
    Feature: read_video
    Description: Test read_video with invalid parameter
    Expectation: Error is caught when the parameter is invalid
    """
    # Test with a directory name
    wrong_filename = "../data/dataset/"
    error_message = "Invalid file path, " + wrong_filename + " is not a regular file."
    invalid_param(wrong_filename, 0, None, "pts", ValueError, error_message)


def test_read_video_param_filename_not_exist():
    """
    Feature: read_video
    Description: Test read_video with invalid parameter
    Expectation: Error is caught when the parameter is invalid
    """
    # Test with a not exist filename
    wrong_filename = "this_file_is_not_exist"
    error_message = "Invalid file path, " + wrong_filename + " does not exist."
    invalid_param(wrong_filename, 0, None, "pts", ValueError, error_message)


def test_read_video_param_filename_not_supported():
    """
    Feature: read_video
    Description: Test read_video with invalid parameter
    Expectation: Error is caught when the parameter is invalid
    """
    # Test with a not supported gif file
    wrong_filename = "../data/dataset/declient.cfg"
    error_message = "Failed to open the file " + wrong_filename
    invalid_param(wrong_filename, 0, None, "pts", RuntimeError, error_message)


def test_read_video_param_start_pts_invalid_type():
    """
    Feature: read_video
    Description: Test read_video with invalid parameter
    Expectation: Error is caught when the parameter is invalid
    """
    # Test with an invalid type for the pts_unit
    filename = "../data/dataset/video/campus.avi"
    error_message = "Input start_pts is not of type"
    invalid_param(filename, "0", None, "pts", TypeError, error_message)


def test_read_video_param_start_pts_invalid_value():
    """
    Feature: read_video
    Description: Test read_video with invalid value -1 for the parameter start_pts
    Expectation: Error is caught when the parameter is invalid
    """
    # Test with an invalid value for the pts_unit
    filename = "../data/dataset/video/campus.avi"
    error_message = "Not supported start_pts for -1"
    invalid_param(filename, -1, None, "pts", ValueError, error_message)


def test_read_video_param_end_pts_invalid_type():
    """
    Feature: read_video
    Description: Test read_video with invalid parameter
    Expectation: Error is caught when the parameter is invalid
    """
    # Test with an invalid type for the pts_unit
    filename = "../data/dataset/video/campus.avi"
    error_message = "Input end_pts is not of type"
    invalid_param(filename, 0, "None", "pts", TypeError, error_message)


def test_read_video_param_end_pts_invalid_value():
    """
    Feature: read_video
    Description: Test read_video with invalid value -1 for the parameter end_pts
    Expectation: Error is caught when the parameter is invalid
    """
    # Test with an invalid value for the end_pts
    filename = "../data/dataset/video/campus.avi"
    error_message = "Not supported end_pts for -1"
    invalid_param(filename, 0, -1, "pts", ValueError, error_message)


def test_read_video_param_pts_unit_invalid_type():
    """
    Feature: read_video
    Description: Test read_video with invalid parameter
    Expectation: Error is caught when the parameter is invalid
    """
    # Test with an invalid type for the pts_unit
    filename = "../data/dataset/video/campus.avi"
    error_message = "Input pts_unit is not of type"
    invalid_param(filename, 0, None, 0, TypeError, error_message)


def test_read_video_param_pts_unit_not_supported():
    """
    Feature: read_video
    Description: Test read_video with invalid parameter
    Expectation: Error is caught when the parameter is invalid
    """
    # Test with an invalid value for the pts_unit
    filename = "../data/dataset/video/campus.avi"
    error_message = "Not supported pts_unit"
    invalid_param(filename, 0, None, "min", ValueError, error_message)


def test_read_video_no_param():
    """
    Feature: read_video
    Description: Test read_video without a parameter
    Expectation: Error is caught when there is no parameter
    """
    # Test without a parameter
    error_message = "read_video() missing 1 required positional argument: 'filename'"
    with pytest.raises(TypeError) as error_info:
        vision.read_video()
    assert error_message in str(error_info.value)


def test_read_video_five_params():
    """
    Feature: read_video
    Description: Test read_video with five parameters
    Expectation: Error is caught when there are five parameters
    """
    # Test with more than five parameter
    error_message = "read_video() takes from 1 to 4 positional arguments but 5 were given"
    with pytest.raises(TypeError) as error_info:
        vision.read_video("1", 2, 3, "4", 5)
    assert error_message in str(error_info.value)


if __name__ == "__main__":
    test_read_video_with_avi_pts()
    test_read_video_with_avi_sec()
    test_read_video_with_avi_start_pts_end_pts()
    test_read_video_with_h264_pts()
    test_read_video_with_h264_sec()
    test_read_video_with_h264_start_pts_end_pts()
    test_read_video_with_h265_pts()
    test_read_video_with_h265_sec()
    test_read_video_with_h265_start_pts_end_pts()
    test_read_video_with_mov_pts()
    test_read_video_with_mov_sec()
    test_read_video_with_mov_start_pts_end_pts()
    test_read_video_with_mp4_pts()
    test_read_video_with_mp4_sec()
    test_read_video_with_mp4_start_pts_end_pts()
    test_read_video_with_wmv_pts()
    test_read_video_with_wmv_sec()
    test_read_video_with_wmv_start_pts_end_pts()
    test_read_video_param_filename_invalid_type()
    test_read_video_param_filename_is_directory()
    test_read_video_param_filename_not_exist()
    test_read_video_param_filename_not_supported()
    test_read_video_param_start_pts_invalid_type()
    test_read_video_param_start_pts_invalid_value()
    test_read_video_param_end_pts_invalid_type()
    test_read_video_param_end_pts_invalid_value()
    test_read_video_param_pts_unit_invalid_type()
    test_read_video_param_pts_unit_not_supported()
    test_read_video_no_param()
    test_read_video_five_params()
