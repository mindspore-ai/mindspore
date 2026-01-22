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
from mindspore import dtype as mstype
from mindspore import Tensor, jit
from mindspore.graph._parse import compile_config


@jit(auto_offload="activaction")
def foo(x):
    m1 = x / 2
    m3 = m1 * 2
    m4 = m3 * 2
    m5 = m3 + m4
    return (m5 * 2) - m1


if __name__ == "__main__":
    origin_select_distance = compile_config.HYPER_OFFLOAD_SELECT_DISTANCE
    origin_prefetch_distance = compile_config.HYPER_OFFLOAD_PREFETCH_DISTANCE
    compile_config.HYPER_OFFLOAD_SELECT_DISTANCE = 1
    compile_config.HYPER_OFFLOAD_PREFETCH_DISTANCE = 0
    input_data = Tensor(np.ones([1000, 100, 100]), dtype=mstype.float32)
    foo(input_data)
    compile_config.HYPER_OFFLOAD_SELECT_DISTANCE = origin_select_distance
    compile_config.HYPER_OFFLOAD_PREFETCH_DISTANCE = origin_prefetch_distance
