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
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P


class AssignAdd(nn.Cell):
    def __init__(self, value):
        super(AssignAdd, self).__init__()
        self.var = Parameter(value, name="var")
        self.add = P.AssignAdd()

    def construct(self, y):
        self.add(self.var, y)
        return self.var


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_assign_add():
    x2 = Tensor(np.arange(1 * 3 * 3 * 3).reshape(1, 3, 3, 3).astype(np.float32))
    y2 = Tensor(np.arange(1 * 3 * 3 * 3).reshape(1, 3, 3, 3).astype(np.float32))

    context.set_context(mode=context.GRAPH_MODE,
                        enable_graph_kernel=True, device_target="GPU")
    add = AssignAdd(x2)
    result_gk_on_1 = add(y2)
    add_2 = AssignAdd(result_gk_on_1)
    result_gk_on_2 = add_2(y2)

    context.set_context(mode=context.GRAPH_MODE,
                        enable_graph_kernel=False, device_target="GPU")
    add_beta = AssignAdd(x2)
    result_gk_off_1 = add_beta(y2)
    add_beta_2 = AssignAdd(result_gk_off_1)
    result_gk_off_2 = add_beta_2(y2)
    assert (result_gk_on_1.asnumpy() == result_gk_off_1.asnumpy()).all()
    assert (result_gk_on_2.asnumpy() == result_gk_off_2.asnumpy()).all()
