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


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_generate_eod_mask():
    """
    Feature: Test GenerateEodMask.
    Description: Test eod_token_id 0
    Expectation: raise TypeError.
    """
    x = np.array([[1, 0, 3, 4, 0, 6, 7, 8], [1, 0, 3, 0, 0, 6, 7, 0]])
    net = Net(0)
    position, mask = net(Tensor(x, dtype=mindspore.int32))
    assert position.shape == (2, 8)
    assert mask.shape == (2, 8, 8)
    assert np.all(mask.asnumpy() == np.array([[[1, 0, 0, 0, 0, 0, 0, 0],
                                               [1, 1, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 1, 0, 0, 0, 0, 0],
                                               [0, 0, 1, 1, 0, 0, 0, 0],
                                               [0, 0, 1, 1, 1, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 1, 0, 0],
                                               [0, 0, 0, 0, 0, 1, 1, 0],
                                               [0, 0, 0, 0, 0, 1, 1, 1]],
                                              [[1, 0, 0, 0, 0, 0, 0, 0],
                                               [1, 1, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 1, 0, 0, 0, 0, 0],
                                               [0, 0, 1, 1, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 1, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 1, 0, 0],
                                               [0, 0, 0, 0, 0, 1, 1, 0],
                                               [0, 0, 0, 0, 0, 1, 1, 1]]]))

    assert np.all(position.asnumpy() == np.array([[0, 1, 0, 1, 2, 0, 1, 2],
                                                  [0, 1, 0, 1, 0, 0, 1, 2]]))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_generate_eod_mask_negative_value():
    """
    Feature: Test GenerateEodMask.
    Description: Test EodTokenId Negative
    Expectation: no errors
    """
    x = np.array([[1, -1, 3, 4, -1, 6, 7, 8], [1, -1, 3, -1, -1, 6, 7, -1]])
    net = Net(eod_token_id=-1)
    position, mask = net(Tensor(x, dtype=mindspore.int32))
    assert position.shape == (2, 8)
    assert mask.shape == (2, 8, 8)
    assert np.all(mask.asnumpy() == np.array([[[1, 0, 0, 0, 0, 0, 0, 0],
                                               [1, 1, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 1, 0, 0, 0, 0, 0],
                                               [0, 0, 1, 1, 0, 0, 0, 0],
                                               [0, 0, 1, 1, 1, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 1, 0, 0],
                                               [0, 0, 0, 0, 0, 1, 1, 0],
                                               [0, 0, 0, 0, 0, 1, 1, 1]],
                                              [[1, 0, 0, 0, 0, 0, 0, 0],
                                               [1, 1, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 1, 0, 0, 0, 0, 0],
                                               [0, 0, 1, 1, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 1, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 1, 0, 0],
                                               [0, 0, 0, 0, 0, 1, 1, 0],
                                               [0, 0, 0, 0, 0, 1, 1, 1]]]))

    assert np.all(position.asnumpy() == np.array([[0, 1, 0, 1, 2, 0, 1, 2],
                                                  [0, 1, 0, 1, 0, 0, 1, 2]]))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_generate_eod_mask_dynamic_inputs():
    """
    Feature: Test GenerateEodMask.
    Description: Test dynamic inputs
    Expectation: No error.
    """
    x = np.array([[1, -1, 3, 4, -1, 6, 7, 8], [1, -1, 3, -1, -1, 6, 7, -1]])
    net = Net(eod_token_id=-1)
    dyn_x = Tensor(shape=(None, None), dtype=mindspore.int32)
    net.set_inputs(dyn_x)
    position, mask = net(Tensor(x, dtype=mindspore.int32))
    assert position.shape == (2, 8)
    assert mask.shape == (2, 8, 8)
    assert np.all(mask.asnumpy() == np.array([[[1, 0, 0, 0, 0, 0, 0, 0],
                                               [1, 1, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 1, 0, 0, 0, 0, 0],
                                               [0, 0, 1, 1, 0, 0, 0, 0],
                                               [0, 0, 1, 1, 1, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 1, 0, 0],
                                               [0, 0, 0, 0, 0, 1, 1, 0],
                                               [0, 0, 0, 0, 0, 1, 1, 1]],
                                              [[1, 0, 0, 0, 0, 0, 0, 0],
                                               [1, 1, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 1, 0, 0, 0, 0, 0],
                                               [0, 0, 1, 1, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 1, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 1, 0, 0],
                                               [0, 0, 0, 0, 0, 1, 1, 0],
                                               [0, 0, 0, 0, 0, 1, 1, 1]]]))

    assert np.all(position.asnumpy() == np.array([[0, 1, 0, 1, 2, 0, 1, 2],
                                                  [0, 1, 0, 1, 0, 0, 1, 2]]))
