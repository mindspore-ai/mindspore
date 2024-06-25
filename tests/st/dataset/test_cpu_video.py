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
import numpy as np

from mindspore.dataset import vision
from tests.mark_utils import arg_mark


filename = "/home/workspace/mindspore_dataset/video_file/final_output.avi"
read_video_video_path = "/home/workspace/mindspore_dataset/video_file/read_video_video_output.npy"
read_video_audio_path = "/home/workspace/mindspore_dataset/video_file/read_video_audio_output.npy"
decode_video_video_path = "/home/workspace/mindspore_dataset/video_file/decode_video_video_output.npy"
decode_video_audio_path = "/home/workspace/mindspore_dataset/video_file/decode_video_audio_output.npy"


def check_mindspore_opencv_data(mindspore_data, video_path, audio_path, error_rate_limit=0.0002):
    """check_mindspore_opencv_data"""
    cv_video_output = np.load(video_path)
    cv_audio_output = np.load(audio_path)
    assert np.allclose(mindspore_data[0], cv_video_output, error_rate_limit, error_rate_limit)
    assert np.allclose(mindspore_data[1], cv_audio_output, error_rate_limit, error_rate_limit)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_acc_read_video():
    """
    Feature: read_video
    Description: Read an AVI file by "pts" as the pts_unit
    Expectation: The Output is equal to the expected output
    """
    mindspore_output = vision.read_video(filename, pts_unit="pts")
    check_mindspore_opencv_data(mindspore_output, read_video_video_path, read_video_audio_path)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_acc_decode_video():
    """
    Feature: DecodeVideo op
    Description: Decode an AVI numpy.ndarray
    Expectation: The Output is equal to the expected output
    """
    raw_ndarray = np.fromfile(filename, np.uint8)
    mindspore_output = vision.DecodeVideo()(raw_ndarray)
    check_mindspore_opencv_data(mindspore_output, decode_video_video_path, decode_video_audio_path)
