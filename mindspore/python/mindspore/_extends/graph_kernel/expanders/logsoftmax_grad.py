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
"""generate json desc for LogSoftmaxGrad"""
from mindspore._extends.graph_kernel.model.model import DataFormat as DF
from ._utils import Expander, ExpanderInfoValidator as VLD


@VLD.add_format(DF.DEFAULT, DF.DEFAULT)
@VLD.check_attrs('axis')
class LogSoftmaxGrad(Expander):
    """LogSoftmaxGrad expander"""

    def _expand(self, graph_builder):
        input_logits, input_dy = self.inputs
        axis = self.attrs['axis']
        if isinstance(axis, int):
            axis = (axis,)

        softmax = graph_builder.emit('Exp', [input_logits])
        dy_sum = graph_builder.emit('ReduceSum', [input_dy], attrs={'reduce_axis': axis, 'keep_dims': True})
        mul_result = graph_builder.emit('Mul', [softmax, dy_sum])
        result = graph_builder.emit('Sub', [input_dy, mul_result])

        return result
