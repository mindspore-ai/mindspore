# Copyright 2020 Huawei Technologies Co., Ltd
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

from mindspore import Tensor, Model
from mindspore import context
from mindspore.nn import AvgPool2d
from mindspore.nn import Cell
from mindspore.nn import Flatten
from mindspore.nn import ReLU
from mindspore.nn import SequentialCell
from ...ut_filter import non_graph_engine


# pylint: disable=W0212


class Net3(Cell):
    def __init__(self):
        super().__init__()
        self.tuple = (ReLU(), ReLU())

    def construct(self, x):
        for op in self.tuple:
            x = op(x)
        return x


@non_graph_engine
def test_cell_list():
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    net = Net3()
    context.set_context(mode=context.GRAPH_MODE)
    model = Model(net)
    model.predict(input_me)


class SequenceNet(Cell):
    def __init__(self):
        super().__init__()
        self.seq = SequentialCell([AvgPool2d(3, 1), ReLU(), Flatten()])
        self.values = list(self.seq._cells.values())

    def construct(self, x):
        x = self.seq(x)
        return x
