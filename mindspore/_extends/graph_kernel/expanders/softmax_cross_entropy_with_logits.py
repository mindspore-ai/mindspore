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
"""generate json desc for SoftmaxCrossEntropyWithLogits"""
from mindspore._extends.graph_kernel.model.model import DataFormat as DF
from ._utils import Expander, ExpanderInfoValidator as VLD


@VLD.add_format(DF.DEFAULT, DF.DEFAULT)
class SoftmaxCrossEntropyWithLogits(Expander):
    """SoftmaxCrossEntropyWithLogits expander"""

    def _expand(self, graph_builder):
        logits, label = self.inputs
        # Calculate softmax_cross_entropy_with_logits(logits, label)
        # formula of softmax_cross_entropy_with_logits is : -reduce_sum(label * log(softmax(logits)))
        axis = (-1,)
        max_x = graph_builder.emit('ReduceMax', [logits], attrs={'reduce_axis': axis, 'keep_dims': True})
        data_sub = graph_builder.emit('Sub', [logits, max_x])
        data_exp = graph_builder.emit('Exp', [data_sub])
        data_expsum = graph_builder.emit('ReduceSum', [data_exp], attrs={'reduce_axis': axis, 'keep_dims': True})
        data_softmax = graph_builder.emit('RealDiv', [data_exp, data_expsum])
        softmax_log = graph_builder.emit('Log', [data_softmax])
        label_mul_log = graph_builder.emit('Mul', [label, softmax_log])
        tmp_res = data_expsum = graph_builder.emit('ReduceSum', [label_mul_log], attrs={
            'reduce_axis': axis, 'keep_dims': True})
        loss = graph_builder.emit('Neg', [tmp_res])
        dlogits = graph_builder.emit('Sub', [data_softmax, label])
        return loss, dlogits
