# Copyright 2022 Huawei Technologies Co., Ltd
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

import copy
from mindspore.nn import Cell
from mindspore.ops.operations import array_ops as P


class Net(Cell):
    def __init__(self):
        super().__init__()
        self.scatter_add = P.TensorScatterAdd()

    def construct(self, x):
        x = self.scatter_add(x, x, x)
        return x


def test_deepcopy_success():
    """
    Feature: deepcopy for TensorScatterXXX ops
    Description: deepcopy a net
    Expectation: copy success
    """
    a = Net()
    b = copy.deepcopy(a)
    assert b is not None
