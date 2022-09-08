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
