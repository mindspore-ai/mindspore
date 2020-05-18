# Copyright 2019 Huawei Technologies Co., Ltd
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

import pytest
import numpy as np
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
import mindspore.nn as nn
import mindspore.context as context


class NetArgmax(nn.Cell):
    def __init__(self):
        super(NetArgmax, self).__init__()
        axis1 = 0
        axis2 = -1
        self.argmax1 = P.Argmax(axis1, output_type=mstype.int32)
        self.argmax2 = P.Argmax(axis2, output_type=mstype.int32)
        self.argmax3 = P.Argmax(output_type=mstype.int32)

    def construct(self, x):
        return (self.argmax1(x), self.argmax2(x), self.argmax3(x))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_argmax():
    x = Tensor(np.array([[1., 20., 5.],
                         [67., 8., 9.],
                         [130., 24., 15.],
                         [0.3, -0.4, -15.]]).astype(np.float32))
    expect1 = np.array([2, 2, 2]).astype(np.int32)
    expect2 = np.array([1, 0, 0, 0]).astype(np.int32)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    Argmax = NetArgmax()
    output = Argmax(x)
    assert (output[0].asnumpy() == expect1).all()
    assert (output[1].asnumpy() == expect2).all()
    assert (output[2].asnumpy() == expect2).all()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    Argmax1 = NetArgmax()
    output1 = Argmax(x)
    assert (output1[0].asnumpy() == expect1).all()
    assert (output1[1].asnumpy() == expect2).all()
    assert (output1[2].asnumpy() == expect2).all()
