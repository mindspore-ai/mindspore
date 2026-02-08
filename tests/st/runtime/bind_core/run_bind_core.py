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
script of binding core: msrun --bind_core and mindspore.runtime.set_cpu_affinity
"""
import os
import json
import numpy as np
import mindspore as ms
import mindspore.ops as P
from mindspore import nn
from mindspore import Tensor, jit
from mindspore import dtype as mstype
from mindspore.common import Parameter

steps = 10

ms.set_context(mode=ms.PYNATIVE_MODE)

def _get_env_with_json(env_name, default):
    value = os.environ.get(env_name, default)
    if value:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return default


#  Env 'DISTRIBUTED' is set to 1 while launching with msrun.
if os.getenv("DISTRIBUTED") == "1":
    ms.communication.init()

if os.getenv("RANK_ID") == "0" or os.getenv("RANK_ID") is None:
    affinity_cpu_list = _get_env_with_json('AFFINITY_CPU_LIST', None)
else:
    affinity_cpu_list = _get_env_with_json('AFFINITY_CPU_LIST_2', None)
module_to_cpu_dict = _get_env_with_json('MODULE_TO_CPU_DICT', None)
#  Env 'THREAD_BIND' is set to 1 while enabling 'mindspore.runtime.set_cpu_affinity'.
if os.getenv("THREAD_BIND") == "1":
    ms.runtime.set_cpu_affinity(True, affinity_cpu_list, module_to_cpu_dict)

class Net(nn.Cell):
    """Network with jit and dynamic shape."""
    def __init__(self):
        super().__init__()
        self.param = Parameter(Tensor(2, ms.float32))
        self.add = P.Add()
        self.mul = P.Mul()

    @jit(backend="ms_backend")
    def construct(self, x):
        x = self.add(x, self.param)
        for _ in range(5):
            x = self.add(x, 0.1)
            x = self.add(x, 0.2)
        x = self.mul(x, 2)
        x = self.add(x, 0.5)
        return x

base_shape = (2, 3)
net = Net()
dyn_input_data = Tensor(shape=[2, None], dtype=mstype.float32)
net.set_inputs(dyn_input_data)

for i in range(steps):
    input_data = Tensor(np.full(base_shape, i).astype(np.float32))
    output = net(input_data)
