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
from mindspore import Parameter, Tensor
from mindspore.ops import operations as P
import mindspore.context as context
import mindspore.nn as nn

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

x_np = np.random.rand(3, 3).astype(np.complex128)
y_np = np.random.rand(3, 3).astype(np.complex128)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.xdivy = P.Xdivy()
        self.xlogy = P.Xlogy()
        self.x = Parameter(Tensor(x_np), name="x")
        self.y = Parameter(Tensor(y_np), name="y")

    def construct(self):
        z1 = self.xdivy(self.x, self.y)
        z2 = self.xlogy(self.x, self.y)
        return z1, z2


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_xdivy():
    """
    Feature: xdivy
    Description: Test complex128 of input
    Expectation: The results are as expected
    """
    expect_output_np1 = x_np / y_np
    expect_output_np2 = x_np * np.log(y_np)
    net = Net()
    out1 = net()[0]
    out2 = net()[1]
    out_mindspore1 = out1.asnumpy()
    print("loss:", expect_output_np1-out_mindspore1)
    out_mindspore2 = out2.asnumpy()
    print("expect_output_np1:", expect_output_np1)
    print("out_mindspore:", out_mindspore1)
    print(expect_output_np2)
    print(out_mindspore2)
    eps = np.array([1e-6 for i in range(9)]).reshape(3, 3)
    assert np.all(expect_output_np1 - out_mindspore1 < eps) and np.all(expect_output_np2 - out_mindspore2 < eps)
