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
Test for tracker.
"""
import mindspore
from mindspore import Tensor

a = Tensor(1.0)
b = Tensor([1.0, 2.0, 3.0])

s1 = mindspore.hal.Stream()
e1 = mindspore.hal.Event()

a.storage().resize_(12)
with mindspore.hal.StreamCtx(s1):
    a.copy_(b)
    e1.record()

e1.wait()
c = a + 1
print(c, flush=True)
