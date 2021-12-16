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
        logits, labels = self.inputs
        # Calculate sigmoid_cross_entropy_with_logits(logits, labels)
        # formula of sigmoid_cross_entropy_with_logits is:
        #     -(labels * log(sigmoid(logits)) + (1 - labels) * log(1 - sigmoid(logits)))
        # To ensure stability and avoid overflow, the formula equal to :
        #     max(logits, 0) - logits * labels + log(1 + exp(-abs(logits)))
        const_one = graph_builder.value(logits.dtype, 1.0)
        const_zero = graph_builder.value(logits.dtype, 0.0)
        max_logits = graph_builder.emit('Maximum', [logits, const_zero])
        logits_mul_labels = graph_builder.emit('Mul', [logits, labels])
        abs_logits = graph_builder.emit('Abs', [logits])
        neg_abs_logits = graph_builder.emit('Neg', [abs_logits])
        exp_neg_abs_logits = graph_builder.emit('Exp', [neg_abs_logits])
        one_add_exp_neg_abs_logits = graph_builder.emit('Add', [const_one, exp_neg_abs_logits])
        log_one_add_exp_neg_abs_logits = graph_builder.emit('Log', [one_add_exp_neg_abs_logits])
        res_tmp = graph_builder.emit('Sub', [max_logits, logits_mul_labels])
        res = graph_builder.emit('Add', [res_tmp, log_one_add_exp_neg_abs_logits])
        return res
