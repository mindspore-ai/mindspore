# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
""" test_celllist """
import numpy as np

from mindspore import context, nn, Tensor, Model, ParameterTuple
from mindspore import dtype as mstype
from ...ut_filter import non_graph_engine


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.tuple = (nn.ReLU(), nn.ReLU())

    def construct(self, x):
        for op in self.tuple:
            x = op(x)
        return x


@non_graph_engine
def test_cell_list():
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    net = Net()
    context.set_context(mode=context.GRAPH_MODE)
    model = Model(net)
    model.predict(input_me)


class CellListNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.all = nn.CellList([nn.Conv2d(120, 240, 4, has_bias=False,
                                          weight_init=Tensor(np.ones([240, 120, 4, 4]), mstype.float32)),
                                nn.Conv2d(240, 480, 4, has_bias=False,
                                          weight_init=Tensor(np.ones([480, 240, 4, 4]), mstype.float32))])
        self.params = ParameterTuple(self.get_parameters())
        self.weight_list = [(240, 120, 4, 4), (480, 240, 4, 4)]
        self.info = [self.all, self.params, self.weight_list]

    def construct(self, x):
        func = None
        conv, params, weight_list = self.info
        for _, (_conv, _, _weight_list) in enumerate(zip(conv, params, weight_list)):
            if _weight_list[0] == 240:
                func = _conv
        out = func(x)
        return out


def test_cell_list_zip():
    """
    Feature: nn.CellList
    Description: Fix the problem of no manager for this func graph when using nn.CellList.
    Expectation: No exception.
    """
    x = Tensor(np.ones([1, 120, 1024, 640]), mstype.float32)
    CellListNet()(x)


def test_cell_list_attr():
    """
    Feature: nn.CellList
    Description: Fix the problem of func_graph cloner.
    Expectation: No exception.
    """
    class AttrNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 3)
            self.conv.from_idx = 1
            self.model = nn.CellList([self.conv])

        def construct(self, *inputs):
            for m in self.model:
                return m.from_idx
            return 0

    x = Tensor(np.random.randn(2, 3, 4, 4), mstype.float32)
    out = AttrNet()(x)
    assert out == 1
