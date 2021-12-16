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
# ==============================================================================
"""
enums for exceptions
"""


class BaseEnum:
    """
    Enum base class.
    """


class LogRuntime(BaseEnum):
    """Log runtime enum."""
    RT_HOST = 0b01
    RT_DEVICE = 0b10


class ErrorCodeType(BaseEnum):
    """Error code type enum."""
    ERROR_CODE = 0b01
    EXCEPTION_CODE = 0b10


class ErrorLevel(BaseEnum):
    """Error level."""
    COMMON_LEVEL = 0b000
    SUGGESTION_LEVEL = 0b001
    MINOR_LEVEL = 0b010
    MAJOR_LEVEL = 0b011
    CRITICAL_LEVEL = 0b100
