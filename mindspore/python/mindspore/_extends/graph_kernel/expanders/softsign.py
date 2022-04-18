# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""generate json desc for Softsign"""
from ._utils import Expander


class Softsign(Expander):
    """Softsign expander"""

    def _expand(self, graph_builder):
        x = self.inputs[0]
        x_abs = graph_builder.emit('Abs', [x])
        one = graph_builder.value(x.dtype, 1.0)
        sum_x = graph_builder.emit('Add', [x_abs, one])
        result = graph_builder.emit('RealDiv', [x, sum_x])
        return result
