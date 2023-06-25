"""
data type classes
"""
from enum import Enum
import numpy as np


class NumpyDtype(Enum):
    """numpy data type class"""
    INT32 = np.dtype('int32')
    INT64 = np.dtype('int64')
    FLOAT = np.dtype('float32')
    FLOAT64 = np.dtype('float64')
    FLOAT16 = np.dtype('float16')
    UINT8 = np.dtype('uint8')
    INT8 = np.dtype('int8')
