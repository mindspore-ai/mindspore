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
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common.api import _bind_device_context


class Add(nn.Cell):
    def __init__(self):
        super(Add, self).__init__()
        self.add = P.Add()

    def construct(self, x1, x2):
        return self.add(x1, x2)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bind_cuda_ctx_api():
    """
    Feature: _bind_device_ctx api
    Description: Test _bind_device_ctx api.
    Expectation: The results are as expected
    """
    x1 = Tensor(np.array([1]))
    x2 = Tensor(np.array([2]))

    _bind_device_context()
    net = Add()
    output = net(x1, x2)
    assert output.asnumpy() == np.array([3])

    _bind_device_context()
    output = net(x1, output)
    assert output.asnumpy() == np.array([4])
