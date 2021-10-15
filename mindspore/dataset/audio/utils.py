# Copyright 2021 Huawei Technologies Co., Ltd
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
Enum for audio ops.
"""

from enum import Enum


class FadeShape(str, Enum):
    """
    Fade Shapes.

    Possible enumeration values are: FadeShape.EXPONENTIAL, FadeShape.HALFSINE, FadeShape.LINEAR,
    FadeShape.LOGARITHMIC, FadeShape.QUARTERSINE.

    - FadeShape.EXPONENTIAL: means the fade shape is exponential mode.
    - FadeShape.HALFSINE: means the fade shape is half_sine mode.
    - FadeShape.LINEAR: means the fade shape is linear mode.
    - FadeShape.LOGARITHMIC: means the fade shape is logarithmic mode.
    - FadeShape.QUARTERSINE: means the fade shape is quarter_sine mode.
    """
    LINEAR: str = "linear"
    EXPONENTIAL: str = "exponential"
    LOGARITHMIC: str = "logarithmic"
    QUARTERSINE: str = "quarter_sine"
    HALFSINE: str = "half_sine"


class GainType(str, Enum):
    """"
    Gain Types.

    Possible enumeration values are: GainType.AMPLITUDE, GainType.DB, GainType.POWER.

    - GainType.AMPLITUDE: means input gain type is amplitude.
    - GainType.DB: means input gain type is decibel.
    - GainType.POWER: means input gain type is power.
    """
    POWER: str = "power"
    AMPLITUDE: str = "amplitude"
    DB: str = "db"


class ScaleType(str, Enum):
    """
    Scale Types.

    Possible enumeration values are: ScaleType.MAGNITUDE, ScaleType.POWER.

    - ScaleType.MAGNITUDE: means the scale of input audio is magnitude.
    - ScaleType.POWER: means the scale of input audio is power.
    """
    POWER: str = "power"
    MAGNITUDE: str = "magnitude"
