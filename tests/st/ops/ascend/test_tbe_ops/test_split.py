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

import mindspore.context as context
import mindspore.nn as nn
from mindspore.ops import operations as P

context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", device_id=6)

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.split = P.Split(0, 2)

    def construct(self, x):
        return self.split(x)


arr_x = np.random.randn(2, 4).astype(np.float32)

def test_f_tensor_split_int(mode):
    """
    Feature: tensor_split
    Description: Verify the result of tensor_split when the type of `indices_or_sections` is int.
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = TensorSplitNet()
    a = np.array(np.arange(20).reshape((10, 2)), dtype=np.float32)
    x = ms.Tensor(a, dtype=ms.float32)
    indices_or_sections = 3
    out = net(x, indices_or_sections)
    expect = np.array_split(a, indices_or_sections)
    for res, exp in zip(out, expect):
        assert np.allclose(res.asnumpy(), exp)
