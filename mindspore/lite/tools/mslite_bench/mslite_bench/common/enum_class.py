# Copyright 2023 Huawei Technologies Co., Ltd
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
"""
data type classes
"""
from enum import Enum
import numpy as np


class NumpyDtype(Enum):
    """numpy data type class"""
    INT32 = np.dtype('int32')
    INT64 = np.dtype('int64')
    FLOAT32 = np.dtype('float32')
    FLOAT64 = np.dtype('float64')
    FLOAT16 = np.dtype('float16')
    UINT8 = np.dtype('uint8')
    INT8 = np.dtype('int8')
