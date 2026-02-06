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

from mindspore.common import Tensor
from mindspore import context, jit
from mindspore.ops.composite import GradOperation


@jit(backend="ms_backend")
def func(x, y):
    x = x * 3
    return 2 * x[0] + y


context.set_context(mode=context.PYNATIVE_MODE)
a = Tensor([1, 2, 3])
b = Tensor([1, 1, 1])
res = GradOperation()(func)(a, b)
print("AAA", res, "BBB")
print("AAA", res.asnumpy().shape, "BBB")
