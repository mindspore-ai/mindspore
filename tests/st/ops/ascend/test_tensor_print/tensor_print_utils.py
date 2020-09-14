# Copyright 2020 Huawei Technologies Co., Ltd
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
import numpy as np

import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

class TensorPrint(nn.Cell):
    def __init__(self):
        super().__init__()
        self.print = P.Print()

    def construct(self, *inputs):
        self.print(*inputs)
        return inputs[0]

def get_tensor(is_scalar, input_type):
    if is_scalar == 'scalar':
        if input_type == mindspore.bool_:
            return Tensor(True, dtype=input_type)
        if input_type in [mindspore.uint8, mindspore.uint16, mindspore.uint32, mindspore.uint64]:
            return Tensor(1, dtype=input_type)
        if input_type in [mindspore.int8, mindspore.int16, mindspore.int32, mindspore.int64]:
            return Tensor(-1, dtype=input_type)
        if input_type in [mindspore.float16, mindspore.float32, mindspore.float64]:
            return Tensor(0.01, dtype=input_type)
    else:
        if input_type == mindspore.bool_:
            return Tensor(np.array([[True, False], [False, True]]), dtype=input_type)
        if input_type in [mindspore.uint8, mindspore.uint16, mindspore.uint32, mindspore.uint64]:
            return Tensor(np.array([[1, 2, 3], [4, 5, 6]]), dtype=input_type)
        if input_type in [mindspore.int8, mindspore.int16, mindspore.int32, mindspore.int64]:
            return Tensor(np.array([[-1, 2, -3], [-4, 5, -6]]), dtype=input_type)
        if input_type in [mindspore.float16, mindspore.float32, mindspore.float64]:
            return Tensor(np.array([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]]), dtype=input_type)
    return Tensor(False, np.bool)

if __name__ == "__main__":
    net = TensorPrint()
    net(get_tensor('scalar', mindspore.bool_), get_tensor('scalar', mindspore.uint8),
        get_tensor('scalar', mindspore.int8), get_tensor('scalar', mindspore.uint16),
        get_tensor('scalar', mindspore.int16), get_tensor('scalar', mindspore.uint32),
        get_tensor('scalar', mindspore.int32), get_tensor('scalar', mindspore.uint64),
        get_tensor('scalar', mindspore.int64), get_tensor('scalar', mindspore.float16),
        get_tensor('scalar', mindspore.float32), get_tensor('scalar', mindspore.float64),
        get_tensor('array', mindspore.bool_), get_tensor('array', mindspore.uint8),
        get_tensor('array', mindspore.int8), get_tensor('array', mindspore.uint16),
        get_tensor('array', mindspore.int16), get_tensor('array', mindspore.uint32),
        get_tensor('array', mindspore.int32), get_tensor('array', mindspore.uint64),
        get_tensor('array', mindspore.int64), get_tensor('array', mindspore.float16),
        get_tensor('array', mindspore.float32), get_tensor('array', mindspore.float64))
