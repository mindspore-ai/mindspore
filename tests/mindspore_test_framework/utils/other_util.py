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

"""Other utils."""
import numpy as np
import mindspore._c_expression as _c_expression

from mindspore.common.tensor import Tensor


def wrap(x):
    if isinstance(x, (tuple, list)):
        return x
    return (x,)


def to_numpy_list(tl):
    tl = wrap(tl)
    ret = []
    for x in tl:
        if isinstance(x, (Tensor, _c_expression.Tensor)):
            ret.append(x.asnumpy())
        else:
            ret.append(x)
    return ret


def to_numpy(x):
    if isinstance(x, (Tensor, _c_expression.Tensor)):
        return x.asnumpy()
    return x


def shape2tensor(shp, dtype=np.float32, scale=6):
    if isinstance(shp, list):
        if not shp:
            return Tensor((np.random.rand() * scale).astype(dtype))
        return Tensor((np.random.rand(*shp) * scale).astype(dtype))
    return shp


def select_from_config_tuple(t, index, default):
    if not isinstance(t, tuple):
        return default
    if not isinstance(t[-1], dict):
        return default
    if index > len(t) - 1:
        return default
    return t[index]
