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
"""generate json desc for Sigmoid"""
from ._utils import Expander


class Sigmoid(Expander):
    """Sigmoid expander"""

    def _expand(self, graph_builder):
        input_x = self.inputs[0]
        # Calculate sigmoid(x)
        # sigmoid of x is 1 / (1 + exp(-x))
        const_one = graph_builder.value(input_x.dtype, 1.0)
        neg_x = graph_builder.emit('Neg', [input_x])
        exp_neg_x = graph_builder.emit('Exp', [neg_x])
        add_exp = graph_builder.emit('Add', [const_one, exp_neg_x])
        res = graph_builder.emit('RealDiv', [const_one, add_exp])
        return res
