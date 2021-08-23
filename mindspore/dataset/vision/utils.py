# Copyright 2019 Huawei Technologies Co., Ltd
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
"""
Interpolation Mode, Resampling Filters
"""
from enum import Enum, IntEnum


class Inter(IntEnum):
    """
    Interpolation Modes.

    Possible enumeration values are: Inter.NEAREST, Inter.ANTIALIAS, Inter.LINEAR, Inter.BILINEAR, Inter.CUBIC,
    Inter.BICUBIC, Inter.AREA, Inter.PILCUBIC.

    - Inter.NEAREST: means interpolation method is nearest-neighbor interpolation.
    - Inter.ANTIALIAS: means the interpolation method is antialias interpolation.
    - Inter.LINEAR: means interpolation method is bilinear interpolation, here is the same as Inter.BILINEAR.
    - Inter.BILINEAR: means interpolation method is bilinear interpolation.
    - Inter.CUBIC: means the interpolation method is bicubic interpolation, here is the same as Inter.BICUBIC.
    - Inter.BICUBIC: means the interpolation method is bicubic interpolation.
    - Inter.AREA: means interpolation method is pixel area interpolation.
    - Inter.PILCUBIC: means interpolation method is bicubic interpolation like implemented in pillow, input
      should be in 3 channels format.
    """
    NEAREST = 0
    ANTIALIAS = 1
    BILINEAR = LINEAR = 2
    BICUBIC = CUBIC = 3
    AREA = 4
    PILCUBIC = 5


class Border(str, Enum):
    """
    Padding Mode, Border Type.

    Possible enumeration values are: Border.CONSTANT, Border.EDGE, Border.REFLECT, Border.SYMMETRIC.

    - Border.CONSTANT: means it fills the border with constant values.
    - Border.EDGE: means it pads with the last value on the edge.
    - Border.REFLECT: means it reflects the values on the edge omitting the last value of edge.
    - Border.SYMMETRIC: means it reflects the values on the edge repeating the last value of edge.

    Note: This class derived from class str to support json serializable.
    """
    CONSTANT: str = "constant"
    EDGE: str = "edge"
    REFLECT: str = "reflect"
    SYMMETRIC: str = "symmetric"


class ImageBatchFormat(IntEnum):
    """
    Data Format of images after batch operation.

    Possible enumeration values are: ImageBatchFormat.NHWC, ImageBatchFormat.NCHW.

    - ImageBatchFormat.NHWC: in orders like, batch N, height H, width W, channels C to store the data.
    - ImageBatchFormat.NCHW: in orders like, batch N, channels C, height H, width W to store the data.
    """
    NHWC = 0
    NCHW = 1


class ConvertMode(IntEnum):
    """The color conversion code"""
    COLOR_BGR2BGRA = 0
    COLOR_RGB2RGBA = COLOR_BGR2BGRA
    COLOR_BGRA2BGR = 1
    COLOR_RGBA2RGB = COLOR_BGRA2BGR
    COLOR_BGR2RGBA = 2
    COLOR_RGB2BGRA = COLOR_BGR2RGBA
    COLOR_RGBA2BGR = 3
    COLOR_BGRA2RGB = COLOR_RGBA2BGR
    COLOR_BGR2RGB = 4
    COLOR_RGB2BGR = COLOR_BGR2RGB
    COLOR_BGRA2RGBA = 5
    COLOR_RGBA2BGRA = COLOR_BGRA2RGBA
    COLOR_BGR2GRAY = 6
    COLOR_RGB2GRAY = 7
    COLOR_GRAY2BGR = 8
    COLOR_GRAY2RGB = COLOR_GRAY2BGR
    COLOR_GRAY2BGRA = 9
    COLOR_GRAY2RGBA = COLOR_GRAY2BGRA
    COLOR_BGRA2GRAY = 10
    COLOR_RGBA2GRAY = 11


class SliceMode(IntEnum):
    """
    Mode to Slice Tensor into multiple parts.

    Possible enumeration values are: SliceMode.PAD, SliceMode.DROP.

    - SliceMode.PAD: pad some pixels before slice the Tensor if needed.
    - SliceMode.DROP: drop remainder pixels before slice the Tensor if needed.
    """
    PAD = 0
    DROP = 1
