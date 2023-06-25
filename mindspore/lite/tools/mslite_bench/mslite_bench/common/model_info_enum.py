"""
model related enum infos
"""

from enum import Enum


class TaskType(Enum):
    """task type enum"""
    MODEL_INFER = 0
    FRAMEWORK_CMP = 1
    NPU_DYNAMIC_INFER = 2


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
    NPY = 0
    BIN = 1
