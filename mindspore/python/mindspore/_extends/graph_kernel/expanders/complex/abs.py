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
"""generate json desc for cabs"""
from mindspore._extends.graph_kernel.expanders._utils import Expander


class CAbs(Expander):
    """CAbs expander"""

    def _expand(self, graph_builder):
        input_x = self.inputs[0]
        x_real = graph_builder.emit('CReal', [input_x])
        x_imag = graph_builder.emit('CImag', [input_x])
        squre_x_real = graph_builder.emit('Mul', [x_real, x_real])
        squre_x_imag = graph_builder.emit('Mul', [x_imag, x_imag])
        squre_sum = graph_builder.emit('Add', [squre_x_real, squre_x_imag])
        result = graph_builder.emit('Sqrt', [squre_sum])
        return result
