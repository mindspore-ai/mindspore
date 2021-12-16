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
"""generate json desc for relu"""
from ._utils import Expander


class ReLU(Expander):
    """ReLU expander"""

    def _expand(self, graph_builder):
        input_x = self.inputs[0]

        const_zero = graph_builder.value(input_x.dtype, 0)
        ge_result = graph_builder.emit('Greater', [input_x, const_zero])
        ge_result = graph_builder.emit('Cast', [ge_result], attrs={'dst_type': input_x.dtype})
        result = graph_builder.emit('Mul', [ge_result, input_x])

        return result
