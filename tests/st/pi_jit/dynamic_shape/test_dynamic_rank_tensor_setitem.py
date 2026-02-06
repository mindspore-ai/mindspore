# Copyright 2025 Huawei Technologies Co., Ltd
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
# ==============================================================================
from mindspore.nn import Cell
from mindspore.common import dtype as mstype
from mindspore.common import Tensor
import numpy as np
from .test_dynamic_rank_tensor_getitem import IndexFactory
from tests.mark_utils import arg_mark


class Net1(Cell):
    def __init__(self):
        super().__init__()
        self.n = 2

    def construct(self, x):
        x = x * 1
        x[...] = 1
        out = x
        return out * self.n


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_rank_setitem_ellipsis():
    '''
    Description:
        1. dynamic rank setitem ellipsis
    Expectation:
        1. the net run ok
        2. the result is the same as psjit
    '''
    ps_net = Net1()
    pi_net = Net1()
    x = Tensor(np.random.rand(2, 3, 4), dtype=mstype.float32)
    d = Tensor(None, dtype=mstype.float32)
    ps_net.set_inputs(d)
    pi_net.set_inputs(d)
    fact = IndexFactory(ps_net, pi_net)
    fact.compare_forward_grad(x)
