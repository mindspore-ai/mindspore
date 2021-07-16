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
# ===========================================================================
"""utils."""
import mindspore.ops.operations as P
from mindspore.nn import Cell
from mindspore.ops import functional as F


class ComputeBase(Cell):
    """define some basic OPs"""
    def __init__(self):
        super(ComputeBase, self).__init__()
        self.mul = F.tensor_mul
        self.sum = F.reduce_sum
        self.cos = F.cos
        self.sin = F.sin
        self.sqrt = F.sqrt
        self.div = P.Div()
        self.reshape = F.reshape
        self.concat = P.Concat(axis=1)
