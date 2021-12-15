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
"""generate json desc for GkDropout"""
from ._utils import Expander, ExpanderInfoValidator as VLD


@VLD.check_all_formats_same
@VLD.check_attrs('keep_prob')
class GkDropout(Expander):
    """GkDropout expander"""

    def _expand(self, graph_builder):
        input_x, input_mask = self.inputs
        keep_prob = self.attrs['keep_prob']

        r_keep_prob = graph_builder.value(input_x.dtype, 1.0 / keep_prob)
        keep_prob = graph_builder.value(input_x.dtype, keep_prob)

        if input_mask.dtype != input_x.dtype:
            input_mask = graph_builder.emit('Cast', [input_mask], attrs={'dst_type': input_x.dtype})
        mask = graph_builder.emit('LessEqual', [input_mask, keep_prob])  # output is bool type
        mask = graph_builder.emit('Cast', [mask], attrs={'dst_type': input_x.dtype})

        # compute result
        result = graph_builder.emit('Mul', [r_keep_prob, input_x])
        result = graph_builder.emit('Mul', [result, mask])

        return result, mask
