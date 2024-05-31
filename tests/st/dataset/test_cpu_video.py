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
# ============================================================================
import av
import cv2
import numpy as np
import pytest

from mindspore import log as logger
from mindspore.dataset import vision


def check_mindspore_opencv_data(mindspore_data, opencv_data, error_rate_limit=0.0002):
    """check_mindspore_opencv_data"""
    assert mindspore_data[0].shape == tuple(opencv_data[0].shape)
    assert mindspore_data[1].shape == tuple(opencv_data[1].shape)
    assert np.allclose(mindspore_data[0], opencv_data[0], error_rate_limit, error_rate_limit)
    assert np.allclose(mindspore_data[1], opencv_data[1], error_rate_limit, error_rate_limit)


def read_audio_pyav(filename: str):
    """read_audio_pyav"""
    av_container = av.open(filename, metadata_errors="ignore")
    total_audio_array = None
    audio_stream = av_container.streams.audio[0]
    if audio_stream is not None:
        for packet in av_container.demux(audio_stream):
            for audio_frame in packet.decode():
                audio_array = audio_frame.to_ndarray()
                if total_audio_array is None:
                    total_audio_array = audio_array
                else:
                    total_audio_array = np.concatenate((total_audio_array, audio_array), axis=1)
    av_container.close()
    return total_audio_array


def read_visual_opencv(filename: str):
    """read_visual_opencv"""
    video_capture = cv2.VideoCapture(filename, cv2.CAP_FFMPEG)
    if not video_capture.isOpened():
        logger.info("Can not open this file")
    else:
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_list = []
        status, frame = video_capture.read()
        i = 0
        while status:
            rgb_frame = np.empty(shape=(frame_height, frame_width, 3), dtype=np.uint8)
            rgb_frame[::, ::, 0] = frame[::, ::, 2]
            rgb_frame[::, ::, 1] = frame[::, ::, 1]
            rgb_frame[::, ::, 2] = frame[::, ::, 0]
            frame_list.append(rgb_frame)
            i += 1
            status, frame = video_capture.read()
    video_capture.release()
    return np.asarray(frame_list)


def read_visual_pyav(filename: str):
    """read_visual_pyav"""
    av_container = av.open(filename, metadata_errors="ignore")
    frame_list = []
    video_stream = av_container.streams.audio[0]
    if video_stream is not None:
        for packet in av_container.demux(video_stream):
            for video_frame in packet.decode():
                video_array = video_frame.to_ndarray(format="rgb24")
                frame_list.append(video_array)
    av_container.close()
    total_video_array = np.asarray(frame_list)
    av_container.close()
    return total_video_array


@pytest.mark.level0
@pytest.mark.platform_arm_cpu
@pytest.mark.env_onecard
def test_acc_read_video():
    """
    Feature: read_video
    Description: Read an AVI file by "pts" as the pts_unit
    Expectation: The Output is equal to the expected output
    """
    filename = "../../ut/python/data/dataset/video/campus.avi"

    mindspore_output = vision.read_video(filename, pts_unit="pts")
    cv_video_output_, cv_audio_output = read_visual_opencv(filename), read_audio_pyav(filename)
    check_mindspore_opencv_data(mindspore_output, [cv_video_output_, cv_audio_output])


@pytest.mark.level0
@pytest.mark.platform_arm_cpu
@pytest.mark.env_onecard
def test_acc_decode_video():
    """
    Feature: DecodeVideo op
    Description: Decode an AVI numpy.ndarray
    Expectation: The Output is equal to the expected output
    """
    filename = "../../ut/python/data/dataset/video/campus.avi"
    raw_ndarray = np.fromfile(filename, np.uint8)
    mindspore_output = vision.DecodeVideo()(raw_ndarray)
    cv_video_output_, cv_audio_output = read_visual_pyav(filename), read_audio_pyav(filename)
    check_mindspore_opencv_data(mindspore_output, [cv_video_output_, cv_audio_output])
