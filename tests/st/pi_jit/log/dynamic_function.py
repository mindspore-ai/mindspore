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
from mindspore import jit, Tensor
from mindspore.nn import Cell


class Net(Cell):
    def __init__(self):
        super().__init__()
        self.x = 1

    @jit(capture_mode="bytecode", backend="ms_backend")
    def construct(self, x):
        if self.x > 3:
            self.x = x
        else:
            self.x = self.x + 1
        return self.x + x


if __name__ == "__main__":
    net = Net()
    for i in range(10):
        x = Tensor([i])
        y = net(x)
