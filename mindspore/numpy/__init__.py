# Copyright 2020 Huawei Technologies Co., Ltd
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
Numpy-like interfaces in mindspore.

Examples:
    >>> import mindspore.numpy as np

Note:
    - array_ops.py define all the array generation and operation interfaces.
    - math_ops.py define all the math operations on tensors.
    - dtypes.py define all the mindspore.numpy dtypes (mainly redirected from mindspore)
    - random/ defines all the random operations.
"""

from .array_ops import (array, asarray, asfarray, ones, zeros, full, arange,
                        linspace, logspace, eye, identity, transpose, expand_dims,
                        squeeze, rollaxis, swapaxes, reshape, ravel, concatenate)
from .array_ops import copy_ as copy
from .dtypes import (int_, int8, int16, int32, int64, uint, uint8, uint16,
                     uint32, uint64, float_, float16, float32, float64, bool_, inf,
                     numeric_types)
from .math_ops import mean, inner


array_ops_module = ['array', 'asarray', 'asfarray', 'copy', 'ones', 'zeros', 'arange',
                    'linspace', 'logspace', 'eye', 'identity', 'transpose', 'expand_dims',
                    'squeeze', 'rollaxis', 'swapaxes', 'reshape', 'ravel', 'concatenate']

math_module = ['mean', 'inner']

__all__ = array_ops_module + math_module + numeric_types

__all__.sort()
