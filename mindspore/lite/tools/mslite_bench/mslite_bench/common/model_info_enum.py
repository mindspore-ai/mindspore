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
model related enum infos
"""

from enum import Enum


class TaskType(Enum):
    """task type enum"""
    MODEL_INFER = "infer"
    FRAMEWORK_CMP = "framework_cmp"
    CONVERTER = "convert"
    AUTO_CMP = "auto_cmp"
    NPU_DYNAMIC_INFER = "npu_dynamic_infer"


class DeviceType(Enum):
    """device type enum"""
    CPU = 'cpu'
    ASCEND = 'ascend'
    GPU = 'gpu'


class FrameworkType(Enum):
    """framework type enum"""
    TF = 'TF'
    ONNX = 'ONNX'
    MSLITE = 'MSLITE'
    PADDLE = 'PADDLE'


class SaveFileType(Enum):
    """save file type enum"""
    DONT_SAVE = 'dont_save'
    NPY = 'npy'
    BIN = 'bin'


class ErrorAlgType(Enum):
    """
    Algorithm types to calculate error between features
    - MEAN_RELATIVE_ERROR: sum(abs(A-B) / A) / A.size
    - COSINE_SIMILARITY: sum(A * B) / (sqrt(sum(A * A)) * sqrt(sum(B * B)))
    """
    MEAN_RELATIVE_ERROR = "mean_relative_error"
    COSINE_SIMILARITY = "cosine_similarity"
