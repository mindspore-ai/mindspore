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
"Net cases for test"
import numpy as np
from mindspore import nn
from mindspore import dtype as mstype
from mindspore import Tensor, Parameter, jit
from mindspore._extends.parse import compile_config


class TestNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.p = Parameter(Tensor(np.ones([1000, 100, 100]), dtype=mstype.float32), device="Remote", name="p")
        self.local_p = Parameter(Tensor(np.ones([100, 100]), dtype=mstype.float32), name="local_p")

    @jit(auto_offload="weight")
    def construct(self, x):
        m1 = x / 2
        m2 = m1 * self.local_p
        m3 = m2 * 2
        remote_compute1 = self.p * 3.14
        remote_compute2 = remote_compute1 + self.p * 0.86
        remote_compute3 = remote_compute2 / self.p
        remote_compute4 = remote_compute3 * self.p
        m4 = m3 + remote_compute4[:, :100, :100]
        m5 = m4 * 2
        final_remote = self.p[:, :100, :100] * m5
        result = final_remote - m1[:, :100, :100]
        return result

if __name__ == "__main__":
    origin_select_distance = compile_config.HYPER_OFFLOAD_SELECT_DISTANCE
    origin_prefetch_distance = compile_config.HYPER_OFFLOAD_PREFETCH_DISTANCE
    compile_config.HYPER_OFFLOAD_SELECT_DISTANCE = 1
    compile_config.HYPER_OFFLOAD_PREFETCH_DISTANCE = 0
    input_data = Tensor(np.ones([1000, 100, 100]), dtype=mstype.float32)
    net = TestNet()
    net(input_data)
    compile_config.HYPER_OFFLOAD_SELECT_DISTANCE = origin_select_distance
    compile_config.HYPER_OFFLOAD_PREFETCH_DISTANCE = origin_prefetch_distance
