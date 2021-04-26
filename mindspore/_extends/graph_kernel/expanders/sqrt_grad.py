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
"""generate json desc for sqrtgrad"""
from ._utils import Expander, ExpanderInfoValidator as VLD


@VLD.check_all_formats_same
class SqrtGrad(Expander):
    """SqrtGrad expander"""

    def _expand(self, graph_builder):
        # formula of sqrt_grad is dout / (2 * x)
        x, dout = self.inputs
        const_two = graph_builder.value(x.dtype, 2)
        dividend = graph_builder.emit('Mul', [x, const_two])
        result = graph_builder.emit('RealDiv', [dout, dividend])
        return result
