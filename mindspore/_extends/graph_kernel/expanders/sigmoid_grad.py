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
"""generate json desc for SigmoidGrad"""
from ._utils import Expander, ExpanderInfoValidator as VLD


@VLD.check_all_formats_same
class SigmoidGrad(Expander):
    """SigmoidGrad expander"""

    def _expand(self, graph_builder):
        input_y, dy = self.inputs
        # Calculate sigmoid_grad(y, dy)
        # formula of sigmoid_grad is : (1 - y) * y * dy
        const_one = graph_builder.value(input_y.dtype, 1.0)
        one_mins_y = graph_builder.emit('Sub', [const_one, input_y])
        y_mul_dy = graph_builder.emit('Mul', [input_y, dy])
        res = graph_builder.emit('Mul', [one_mins_y, y_mul_dy])
        return res
