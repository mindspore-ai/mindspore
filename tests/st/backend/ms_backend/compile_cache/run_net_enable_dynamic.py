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
"""
test compile cache with enable_dynamic in JIT mode
"""
import numpy as np
import mindspore as ms
from mindspore import Tensor


d1 = Tensor(shape=[None, 4], dtype=ms.float32)
d2 = Tensor(shape=[3, None], dtype=ms.float32)
ds = [d1, (d1, d2)]
@ms.jit(backend="ms_backend")
@ms.enable_dynamic(x=ds)
def my_mul(x):
    return x[0] * 2, x[1][1] * 3

def my_mul_nojit(y):
    return y[0] * 2, y[1][1] * 3


if __name__ == "__main__":
    x_shapes = [(1, 4), (2, 4), (3, 4)]
    y_shapes = [(3, 4), (3, 2), (3, 3)]
    lists = []
    for i in range(3):
        x_input = ms.Tensor(np.ones(x_shapes[i]), ms.float32)
        y_input = ms.Tensor(np.ones(y_shapes[i]), ms.float32)
        lists.append([x_input, (x_input, y_input)])

    for list_ in lists:
        out_0, out_1 = my_mul(ms.mutable(list_))
        nojit_out_0, nojit_out_1 = my_mul_nojit(ms.mutable(list_))
        assert (out_0 == nojit_out_0).all()
        assert (out_1 == nojit_out_1).all()

    print("RUNTIME_COMPILE", out_0[0], "RUNTIME_CACHE")
    print("RUNTIME_COMPILE", out_0[0].asnumpy().shape, "RUNTIME_CACHE")
