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
"""generate json desc for minimum_grad"""
from mindspore._extends.graph_kernel.model.model import GraphKernelUnsupportedException as GKException
from ._utils import Expander, ExpanderInfoValidator as VLD


@VLD.check_all_formats_same
class MinimumGrad(Expander):
    """MinimumGrad expander"""

    def _check(self):
        if not self.attrs.get('grad_x', True) and not self.attrs.get('grad_y', True):
            raise GKException("both grad_x and grad_y are False.")
        return super()._check()

    def _expand(self, graph_builder):
        input_x, input_y, input_dout = self.inputs

        le_result = graph_builder.emit('LessEqual', [input_x, input_y])
        le_result = graph_builder.emit('Cast', [le_result], attrs={'dst_type': input_x.dtype})
        dx = graph_builder.emit('Mul', [le_result, input_dout])
        dy = graph_builder.emit('Sub', [input_dout, dx])

        # output two results, regardless of grad_x and grad_y
        return dx, dy
