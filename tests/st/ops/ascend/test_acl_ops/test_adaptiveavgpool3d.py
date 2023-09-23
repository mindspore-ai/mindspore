# Copyright 2021 Huawei Technologies Co., Ltd
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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.ops.operations.nn_ops as P
from mindspore.common.api import jit

context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend')


class Net(nn.Cell):
    def __init__(self, output_size):
        super(Net, self).__init__()
        self.adaptive_avg_pool3d = P.AdaptiveAvgPool3D(output_size)

    @jit
    def construct(self, x):
        return self.adaptive_avg_pool3d(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_adaptiveavgpool3d_acl():
    '''
    Feature: Test adaptive_avg_pool3d on ACL
    Description: A randomly generated 5-dimensional matrix, Expected pooled output size
    Expectation: Successfully get output with expected output size
    '''
    output_size = (3, 4, 5)
    shape = (1, 32, 9, 9, 9)
    net = Net(output_size)
    x = Tensor(np.random.randn(*shape).astype(np.float32))
    output = net(x)
    expect_shape = shape[:-3] + output_size
    assert output.asnumpy().shape == expect_shape
