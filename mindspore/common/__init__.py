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
"""Top-level reference to dtype of common module."""
from . import dtype
from . import monad
from .api import ms_function
from .dtype import *
from .parameter import Parameter, ParameterTuple
from .tensor import Tensor, RowTensor, SparseTensor
from .seed import set_seed, get_seed


__all__ = dtype.__all__
__all__.extend([
    "Tensor", "RowTensor", "SparseTensor",  # tensor
    'ms_function',  # api
    'Parameter', 'ParameterTuple',  # parameter
    "dtype",
    'monad',
    "set_seed", "get_seed"  # random seed
    ])
