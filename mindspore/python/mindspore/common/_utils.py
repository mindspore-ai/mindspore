# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
# Copyright 2022 Huawei Technologies Co., Ltd
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
"""common utils."""

import os
import math
import functools

from mindspore.common import dtype as mstype


def is_shape_unknown(shape):
    """Check whether the shape is unknown."""
    flag = False
    for i in shape:
        if i < -2:
            raise ValueError(f"'shape' should not have values less than -2 but got ({shape}).")
        if i == -1:
            flag = True
    return is_dim_unknown(shape) or flag


def is_dim_unknown(shape):
    """Check whether the dim is unknown."""
    if len(shape) == 1 and shape[0] == -2:
        return True
    if -2 in shape:
        raise ValueError(f"'shape' should have only one -2 or no -2 at all but got ({shape}).")
    return False


def split_to_slice_if_need(dtype, shape):
    # check if size of data is too huge, and cut it to a smaller one.
    num_element = functools.reduce(lambda x, y: x * y, shape, 1)
    data_size = num_element * mstype.type_size_in_bytes(dtype)
    emb_cache_size = int(os.getenv("MS_EMBEDDING_REMOTE_CACHE_MEMORY_SIZE", "100")) << 30
    slice_num = 1
    if data_size <= emb_cache_size:
        return slice_num
    slice_num = math.ceil(data_size / emb_cache_size)
    return slice_num


def update_and_return_dict(dic, key, val):
    dic.__setitem__(key, val)
    return dic
