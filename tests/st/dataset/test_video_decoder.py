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
# ==============================================================================
"""
Test VideoDecode
"""
import os
import random

import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from tests.mark_utils import arg_mark

PWD = os.path.dirname(__file__)
filename_h264 = PWD + "/data/campus.h264"
filename_h265 = PWD + "/data/campus_h265.mp4"


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=8.0)
@pytest.mark.parametrize("filename", [filename_h264, filename_h265])
def test_video_decoder(filename):
    """
    Feature: VideoDecoder
    Description: Extracting frames from H264/H265-encoded content
    Expectation: The Output is equal to the expected output
    """
    original_video_backend = ds.config.get_video_backend()
    ds.config.set_video_backend("Ascend")

    output_frames = vision.read_video(filename, pts_unit="pts")[0]

    reader = vision.VideoDecoder(source=filename)
    metadata = reader.metadata
    assert metadata["width"] == 270
    assert metadata["height"] == 480
    assert metadata["duration_seconds"] == 0.633333
    assert metadata["num_frames"] == 19
    assert metadata["average_fps"] == 30.0

    random_list = random.sample(range(19), k=10)
    # Random frame extraction
    results = reader.get_frames_at(random_list)
    for i in range(len(results)):
        assert np.array_equal(results[i], output_frames[random_list[i]])

    random_list.sort()
    # Sequential frame extraction
    results2 = reader.get_frames_at(random_list)
    for i in range(len(results2)):
        assert np.array_equal(results2[i], output_frames[random_list[i]])

    random_list = random.choices(range(15), k=20)
    # Repeat frame extraction
    results3 = reader.get_frames_at(random_list)
    for i in range(len(results3)):
        assert np.array_equal(results3[i], output_frames[random_list[i]])

    # Empty list
    results4 = reader.get_frames_at([])
    assert np.array_equal(results4, np.empty(0, dtype=np.uint8))

    ds.config.set_video_backend(original_video_backend)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_video_decoder_exception_case():
    """
    Feature: VideoDecoder
    Description: Exception case: extracting frames from H.264-encoded content
    Expectation: Correct error is raised as expected
    """
    original_video_backend = ds.config.get_video_backend()
    ds.config.set_video_backend("Ascend")
    with pytest.raises(ValueError) as err:
        _ = vision.VideoDecoder(source="filename")
    assert "does not exist or permission denied!" in str(err.value)

    with pytest.raises(TypeError) as err:
        _ = vision.VideoDecoder(source=True)
    assert "path: True is not string" in str(err.value)

    with pytest.raises(TypeError) as err:
        reader = vision.VideoDecoder(source=filename_h264)
        _ = reader.get_frames_at((1, 2))
    assert "Argument indices with value (1, 2) is not of type [<class 'list'>], but got <class 'tuple'>." \
           in str(err.value)

    with pytest.raises(TypeError) as err:
        reader = vision.VideoDecoder(source=filename_h264)
        _ = reader.get_frames_at([1, 2.0])
    assert "Argument indices[1] with value 2.0 is not of type [<class 'int'>], but got <class 'float'>." \
           in str(err.value)

    with pytest.raises(TypeError) as err:
        reader = vision.VideoDecoder(source=filename_h264)
        _ = reader.get_frames_at([1, True])
    assert "Argument indices[1] with value True is not of type (<class 'int'>,), but got <class 'bool'>." \
           in str(err.value)

    with pytest.raises(ValueError) as err:
        reader = vision.VideoDecoder(source=filename_h264)
        _ = reader.get_frames_at([0, 19])
    assert "Input Invalid frame index[1]=19 is not within the required interval of [0, 19)." in str(err.value)

    with pytest.raises(ValueError) as err:
        reader = vision.VideoDecoder(source=filename_h264)
        _ = reader.get_frames_at([0, -1])
    assert "Input Invalid frame index[1]=-1 is not within the required interval of [0, 19)." in str(err.value)

    filename2 = PWD + "/data/campus.avi"
    with pytest.raises(RuntimeError) as err:
        reader = vision.VideoDecoder(source=filename2)
        _ = reader.get_frames_at([0])
    assert "not supported on DVPP backend" in str(err.value)

    ds.config.set_video_backend("CPU")
    with pytest.raises(RuntimeError) as err:
        reader = vision.VideoDecoder(source=filename_h264)
        _ = reader.get_frames_at([0, 5])
    assert "only supported on Ascend platform" in str(err.value)

    ds.config.set_video_backend(original_video_backend)


if __name__ == '__main__':
    test_video_decoder(filename)
    test_video_decoder_exception_case()
