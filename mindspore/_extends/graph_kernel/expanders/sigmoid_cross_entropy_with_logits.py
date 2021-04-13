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
"""generate json desc for SigmoidCrossEntropyWithLogits"""
from ._utils import Expander, ExpanderInfoValidator as VLD


@VLD.check_all_formats_same
class SigmoidCrossEntropyWithLogits(Expander):
    """SigmoidCrossEntropyWithLogits expander"""

    def _expand(self, graph_builder):
        logits, label = self.inputs
        # Calculate sigmoid_cross_entropy_with_logits(logits, label)
        # formula is: -(label * log(sigmoid(logits)) + (1 - label) * log(1 - sigmoid(logits)))
        const_one = graph_builder.value(logits.dtype, 1.0)
        neg_x = graph_builder.emit('Neg', [logits])
        exp_neg_x = graph_builder.emit('Exp', [neg_x])
        add_exp = graph_builder.emit('Add', [const_one, exp_neg_x])
        p = graph_builder.emit('RealDiv', [const_one, add_exp])
        one_sub_p = graph_builder.emit('Sub', [const_one, p])
        one_sub_label = graph_builder.emit('Sub', [const_one, label])
        log_p = graph_builder.emit('Log', [p])
        log_one_sub_p = graph_builder.emit('Log', [one_sub_p])
        res_tmp_1 = graph_builder.emit('Mul', [one_sub_label, log_one_sub_p])
        res_tmp_2 = graph_builder.emit('Mul', [label, log_p])
        res_tmp = graph_builder.emit('Add', [res_tmp_1, res_tmp_2])
        res = graph_builder.emit('Neg', [res_tmp])
        return res
