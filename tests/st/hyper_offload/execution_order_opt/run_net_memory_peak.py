# Copyright 2026 Huawei Technologies Co., Ltd
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
"Net cases for memory peak based hyper offload strategy test"
import numpy as np
from mindspore import nn
from mindspore import dtype as mstype
from mindspore import Tensor, jit
from mindspore.graph._parse import compile_config


class MemoryPeakTestNet(nn.Cell):
    """Test network for memory peak based hyper offload strategy."""

    @jit(auto_offload="activaction")
    def construct(self, x):
        """Forward pass with activation offload."""
        step1 = x * 2
        step2 = step1 + x
        step3 = step2 * 3
        step4 = step3 - x
        step5 = step4 / 2
        step6 = step5 + step4
        step7 = step6 + step3
        step8 = step7 + step2
        return step8 + step1


if __name__ == "__main__":
    origin_max_device_memory = compile_config.HYPER_OFFLOAD_MAX_DEVICE_MEMORY
    compile_config.HYPER_OFFLOAD_MAX_DEVICE_MEMORY = str(100 * 1024 * 1024)
    input_data = Tensor(np.ones([1000, 100, 100]), dtype=mstype.float32)
    net = MemoryPeakTestNet()
    net(input_data)
    compile_config.HYPER_OFFLOAD_MAX_DEVICE_MEMORY = origin_max_device_memory
