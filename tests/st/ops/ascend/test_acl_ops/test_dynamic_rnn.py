# Copyright 2022 Huawei Technologies Co., Ltd
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
import pytest
import mindspore
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P

context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")


class Net(Cell):
    "DynamicRNN network."

    def __init__(self):
        super(Net, self).__init__()
        self.op = P.DynamicRNN()

    def construct(self, x, w, b, init_h, init_c):
        x = self.op(x, w, b, None, init_h, init_c)
        return x


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_batchmatmul_acl_dynamic_shape():
    """
    Feature: Test acl call with pynative mode and dynamic shape.
    Description: The first input is dynamic.
    Expectation: print output x.
    """
    np.random.seed(1024)
    x = Tensor(np.random.rand(2, 16, 64).astype(np.float16))
    w = Tensor(np.random.rand(96, 128).astype(np.float16))
    b = Tensor(np.random.rand(128).astype(np.float16))
    init_h = Tensor(np.random.rand(1, 16, 32).astype(np.float16))
    init_c = Tensor(np.random.rand(1, 16, 32).astype(np.float16))
    dynamic_rnn = Net()
    dynamic_rnn.set_inputs(Tensor(shape=[None, 16, 64], dtype=mindspore.float16), w, b, init_h, init_c)
    output = dynamic_rnn(x, w, b, init_h, init_c)
    print(output)
