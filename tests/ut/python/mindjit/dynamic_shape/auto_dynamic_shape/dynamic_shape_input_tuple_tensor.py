# Copyright 2025 Huawei Technologies Co., Ltd
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

import numpy as np
from mindspore import Tensor, jit, mutable
from mindspore import dtype as mstype


@jit(dynamic=1)
def func(x):
    out = x[0] + x[1]
    return out

x1 = Tensor(np.random.rand(2, 3, 4), mstype.float32)
y1 = Tensor(np.random.rand(2, 3, 4), mstype.float32)
x2 = Tensor(np.random.rand(3, 3, 4), mstype.float32)
y2 = Tensor(np.random.rand(3, 3, 4), mstype.float32)
x3 = Tensor(np.random.rand(4, 3, 4), mstype.float32)
y3 = Tensor(np.random.rand(4, 3, 4), mstype.float32)
x4 = Tensor(np.random.rand(5, 3, 4), mstype.float32)
y4 = Tensor(np.random.rand(5, 3, 4), mstype.float32)

func(mutable((x1, y1)))
func(mutable((x2, y2)))
func(mutable((x3, y3)))
func(mutable((x4, y4)))
