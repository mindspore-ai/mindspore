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
test compile cache with mindspore jit func
"""
import numpy as np
from mindspore import context
from mindspore import Tensor
import mindspore as ms

@ms.jit
def func(x, y):
    output = y - y * x
    return output


if __name__ == "__main__":
    context.set_context(mode=context.PYNATIVE_MODE)
    input_x = Tensor(np.array([1]).astype(np.float32))
    input_y = Tensor(np.array([2]).astype(np.float32))
    res = func(input_x, input_y)
    print("RUNTIME_COMPILE", res, "RUNTIME_CACHE")
    print("RUNTIME_COMPILE", res.asnumpy().shape, "RUNTIME_CACHE")
