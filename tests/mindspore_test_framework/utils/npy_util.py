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

"""Utils for npy file operations."""

import numpy as np

from mindspore.common.tensor import Tensor
from .config_util import get_expect_config
from .other_util import shape2tensor


def load_npy(p):
    s, dtype, scale, max_error, check_tolerance, relative_tolerance, absolute_tolerance = get_expect_config(p)
    if isinstance(s, str):
        try:
            ret = Tensor(np.array((np.load(s, allow_pickle=True) * scale).astype(dtype)))
        except ValueError:
            ret = Tensor(np.array((np.load(s, allow_pickle=False) * scale).astype(dtype)))
    else:
        ret = shape2tensor(s, dtype, scale)
    return ret, max_error, check_tolerance, relative_tolerance, absolute_tolerance


def load_data_from_npy_or_shape(dpaths, skip_expect_config=True):
    ret = []
    for p in dpaths:
        d, max_error, check_tolerance, relative_tolerance, absolute_tolerance = load_npy(p)
        if skip_expect_config:
            ret.append(d)
        else:
            ret.append((d, max_error, check_tolerance, relative_tolerance, absolute_tolerance))
    return ret
