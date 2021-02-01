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
""" test Add """
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.Add()

    def construct(self, input1, input2):
        return self.add(input1, input2)


def test_tensor_add():
    """test_tensor_add"""
    add = P.Add()
    input1 = Tensor(np.random.rand(1, 3, 4, 4).astype(np.float32))
    input2 = Tensor(np.random.rand(1, 3, 4, 4).astype(np.float32))
    output = add(input1, input2)
    output_np = output.asnumpy()
    input1_np = input1.asnumpy()
    input2_np = input2.asnumpy()
    print(input1_np[0][0][0][0])
    print(input2_np[0][0][0][0])
    print(output_np[0][0][0][0])
    assert isinstance(output_np[0][0][0][0], np.float32)
