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
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter


class Net(nn.Cell):
    def __init__(self, param):
        super(Net, self).__init__()
        self.lamb_apply_weight_assign = P.LambApplyWeightAssign()
        self.param = Parameter(param, name='param')

    def construct(self, w_norm, g_norm, lr, update):
        return self.lamb_apply_weight_assign(w_norm, g_norm, lr, update, self.param)


def get_output(w_norm, g_norm, lr, update, param, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    opt = Net(Tensor(param))
    _ = opt(Tensor(w_norm), Tensor(g_norm), Tensor(lr), Tensor(update))
    return opt.param.data.asnumpy()


def lamb_apply_weight_assign():

    w_norm = np.array([0.11]).astype(np.float32)
    g_norm = np.array([1.2]).astype(np.float32)
    lr = np.array([0.012]).astype(np.float32)
    update = np.array([0.01, 0.03, 0.05]).astype(np.float32)
    param = np.array([1, 3, 5]).astype(np.float32)

    expect = get_output(w_norm, g_norm, lr, update, param, False)
    output = get_output(w_norm, g_norm, lr, update, param, True)

    assert np.allclose(output, expect)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_lamb_apply_weight_assign_ascend():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    lamb_apply_weight_assign()
