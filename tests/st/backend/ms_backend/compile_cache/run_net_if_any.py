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
test compile cache with control flow.
"""
from mindspore import context, Tensor, nn, jit


@jit
class Net(nn.Cell):
    def __init__(self, input1, input2):
        super().__init__()
        self.input1 = input1
        self.input2 = input2

    def construct(self):
        if self.input1.all() == self.input2:
            return self.input1.any()
        return self.input2


if __name__ == "__main__":
    context.set_context(jit_config={"jit_level": "O1"})
    x = Tensor([True, True, False])
    y = Tensor([False])
    net = Net(x, y)
    output = net()
    assert output
    print("RUNTIME_COMPILE", output, "RUNTIME_CACHE")
    print("RUNTIME_COMPILE", output.asnumpy().shape, "RUNTIME_CACHE")
