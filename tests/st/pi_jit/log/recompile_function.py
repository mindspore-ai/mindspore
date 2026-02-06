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
"""pijit process"""
from mindspore import jit, ops


@jit(capture_mode="bytecode")
def fn(x):
    y = ops.add(x, 3)
    print("Hi", flush=True)
    z = ops.add(x, y)
    return z


if __name__ == "__main__":
    fn(ops.randn(4))
    fn(ops.randn(4,4))
