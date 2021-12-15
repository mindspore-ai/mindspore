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
"""generate json desc for tanh_grad"""
from ._utils import Expander, ExpanderInfoValidator as VLD


@VLD.check_all_formats_same
class TanhGrad(Expander):
    """TanhGrad expander"""

    def _expand(self, graph_builder):
        input_y, input_dy = self.inputs

        const_one = graph_builder.value(input_y.dtype, 1)
        double_y = graph_builder.emit('Mul', [input_y, input_y])
        one_sub_double_y = graph_builder.emit('Sub', [const_one, double_y])
        result = graph_builder.emit('Mul', [input_dy, one_sub_double_y])

        return result
