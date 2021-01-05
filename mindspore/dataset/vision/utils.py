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
    NEAREST = 0
    ANTIALIAS = 1
    BILINEAR = LINEAR = 2
    BICUBIC = CUBIC = 3
    AREA = 4


# Padding Mode, Border Type
# Note: This class derived from class str to support json serializable.
class Border(str, Enum):
    CONSTANT: str = "constant"
    EDGE: str = "edge"
    REFLECT: str = "reflect"
    SYMMETRIC: str = "symmetric"


# Image Batch Format
class ImageBatchFormat(IntEnum):
    NHWC = 0
    NCHW = 1
