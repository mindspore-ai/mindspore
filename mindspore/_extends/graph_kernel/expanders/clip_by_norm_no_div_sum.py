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
"""generate json desc for ClipByNormNoDivSum"""
from ._utils import Expander, ExpanderInfoValidator as VLD


@VLD.check_all_formats_same
class ClipByNormNoDivSum(Expander):
    """ClipByNormNoDivSum expander"""

    def _expand(self, graph_builder):
        input_x0, input_x1, input_x2, input_x3 = self.inputs

        # cal result
        greater_res = graph_builder.emit('Greater', [input_x0, input_x1], attrs={'fusion': 'SelectGT_000'})
        select_res0 = graph_builder.emit('Select', [greater_res, input_x0, input_x2],
                                         attrs={'fusion': 'SelectGT_000_end'})
        sqrt_res = graph_builder.emit('Sqrt', [select_res0])
        select_res1 = graph_builder.emit('Select', [greater_res, sqrt_res, input_x0],
                                         attrs={'fusion': 'SelectGT_000_end'})
        result = graph_builder.emit('Maximum', [select_res1, input_x3])

        return result
