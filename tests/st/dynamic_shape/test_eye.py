# Copyright 2023 Huawei Technologies Co., Ltd
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
import mindspore as ms
from mindspore import Tensor, nn, ops

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")


class EyeNet(nn.Cell):
    def __init__(self, dtype=ms.int32):
        super(EyeNet, self).__init__()
        self.dtype = dtype

    def construct(self, x):
        return ops.eye(x.shape[1], 3, dtype=self.dtype)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_eye_dynamic():
    """
    Feature: Test operator Eye with dynamic input in PyNative mode
    Description:  Test operator Eye with dynamic input in ACL mode
    Expectation: the result of Eye is correct.
    """
    net = EyeNet()
    input_dyn = Tensor(shape=[3, None], dtype=ms.float32)
    net.set_inputs(input_dyn)
    x = Tensor(np.ones([3, 3]), dtype=ms.float32)
    out = net(x).asnumpy()
    error = 1.0e-6
    assert np.allclose(out, np.eye(3), error, error)
