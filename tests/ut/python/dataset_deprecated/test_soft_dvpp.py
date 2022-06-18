# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
Test SoftDvppDecodeResizeJpeg and SoftDvppDecodeRandomCropResizeJpeg
"""
import numpy as np
import pytest

# SoftDvpp will be deprecated in future so will not be included in transforms.py
import mindspore.dataset.vision.c_transforms as vision


def test_soft_dvpp_deprecated_pipeline():
    """
    Feature: SoftDvppDecodeResizeJpeg and SoftDvppDecodeRandomCropResizeJpeg
    Description: Test deprecation message and error in pipeline mode
    Expectation: Raise NotImplementedError
    """
    with pytest.raises(NotImplementedError) as e:
        vision.SoftDvppDecodeResizeJpeg((256, 512))
    assert "SoftDvppDecodeResizeJpeg is not supported as of 1.8 version" in str(e.value)

    with pytest.raises(NotImplementedError) as e:
        vision.SoftDvppDecodeRandomCropResizeJpeg((256, 512), (1, 1), (0.5, 0.5))
    assert "SoftDvppDecodeRandomCropResizeJpeg is not supported as of 1.8 version" in str(e.value)


def test_soft_dvpp_deprecated_eager():
    """
    Feature: SoftDvppDecodeResizeJpeg and SoftDvppDecodeRandomCropResizeJpeg
    Description: Test deprecation message and error in eager mode
    Expectation: Raise NotImplementedError
    """
    img = np.fromfile("../data/dataset/apple.jpg", dtype=np.uint8)

    with pytest.raises(NotImplementedError) as e:
        vision.SoftDvppDecodeResizeJpeg((256, 512))(img)
    assert "SoftDvppDecodeResizeJpeg is not supported as of 1.8 version" in str(e.value)

    with pytest.raises(NotImplementedError) as e:
        vision.SoftDvppDecodeRandomCropResizeJpeg((256, 512), (1, 1), (0.5, 0.5))
    assert "SoftDvppDecodeRandomCropResizeJpeg is not supported as of 1.8 version" in str(e.value)


if __name__ == "__main__":
    test_soft_dvpp_deprecated_pipeline()
    test_soft_dvpp_deprecated_eager()
