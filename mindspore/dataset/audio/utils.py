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
enum for audio ops
"""

from enum import Enum


class FadeShape(str, Enum):
    """Fade Shape"""
    LINEAR: str = "linear"
    EXPONENTIAL: str = "exponential"
    LOGARITHMIC: str = "logarithmic"
    QUARTERSINE: str = "quarter_sine"
    HALFSINE: str = "half_sine"


class GainType(str, Enum):
    """Gain Type"""
    POWER: str = "power"
    AMPLITUDE: str = "amplitude"
    DB: str = "db"


class ScaleType(str, Enum):
    """Scale Type"""
    POWER: str = "power"
    MAGNITUDE: str = "magnitude"
