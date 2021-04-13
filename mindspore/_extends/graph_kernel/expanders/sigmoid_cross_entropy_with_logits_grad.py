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
"""generate json desc for SigmoidCrossEntropyWithLogitsGrad"""
from ._utils import Expander, ExpanderInfoValidator as VLD


@VLD.check_all_formats_same
class SigmoidCrossEntropyWithLogitsGrad(Expander):
    """SigmoidCrossEntropyWithLogitsGrad expander"""

    def _expand(self, graph_builder):
        logits, label, dout = self.inputs
        # Calculate sigmoid_cross_entropy_with_logits_grad(logits, label, dout)
        # formula of sigmoid_cross_entropy_with_logits_grad is : (sigmoid(logits) - label) * dout
        const_one = graph_builder.value(logits.dtype, 1.0)
        neg_x = graph_builder.emit('Neg', [logits])
        exp_neg_x = graph_builder.emit('Exp', [neg_x])
        add_exp = graph_builder.emit('Add', [const_one, exp_neg_x])
        sigmoid_res = graph_builder.emit('RealDiv', [const_one, add_exp])
        sigmoid_res_sub_label = graph_builder.emit('Sub', [sigmoid_res, label])
        res = graph_builder.emit('Mul', [sigmoid_res_sub_label, dout])
        return res
