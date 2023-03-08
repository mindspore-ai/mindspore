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
import pytest
import numpy as np

import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self, eod_token_id):
        super(Net, self).__init__()
        self.mask = P.GenerateEodMask(eod_token_id=eod_token_id)

    def construct(self, tensor):
        return self.mask(tensor)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_generate_eod_mask_wrong_type():
    """
    Feature: Test GenerateEodMask.
    Description: Test float_token_id
    Expectation: raise TypeError.
    """
    x = np.array([[1, 0, 3, 4, 0, 6, 7, 8], [1, 0, 3, 0, 0, 6, 7, 0]])
    net = Net(0)
    with pytest.raises(TypeError):
        net(Tensor(x, dtype=mindspore.float32))



@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_generate_eod_mask_float_token_id():
    """
    Feature: Test GenerateEodMask.
    Description: Test float_token_id
    Expectation: raise TypeError.
    """
    x = np.array([[1, 0, 3, 4, 0, 6, 7, 8], [1, 0, 3, 0, 0, 6, 7, 0]])
    with pytest.raises(TypeError):
        net = Net(-1.0)
        net(Tensor(x, dtype=mindspore.float32))


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [mindspore.int32, mindspore.int64, mindspore.uint16,
                                   mindspore.uint32, mindspore.uint64])
def test_generate_eod_mask_support_dtype(dtype):
    """
    Feature: Test GenerateEodMask.
    Description: Test multi dtype inputs
    Expectation: Successful graph compilation.
    """
    x = np.array([[1, 0, 3, 4, 0, 6, 7, 8], [1, 0, 3, 0, 0, 6, 7, 0]])
    net = Net(0)
    net(Tensor(x, dtype=dtype))
