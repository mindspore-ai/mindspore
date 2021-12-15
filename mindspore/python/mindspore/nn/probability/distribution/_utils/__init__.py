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
Distribution operation utility functions.
"""
from .custom_ops import exp_generic, log_generic, broadcast_to
from .utils import cast_to_tensor, check_greater_equal_zero, check_greater_zero
from .utils import check_greater, check_prob, CheckTensor, CheckTuple, set_param_type

__all__ = [
    'cast_to_tensor',
    'check_greater',
    'check_greater_equal_zero',
    'check_greater_zero',
    'check_prob',
    'exp_generic',
    'log_generic',
    'broadcast_to',
    'set_param_type',
    'CheckTensor',
    'CheckTuple',
]
