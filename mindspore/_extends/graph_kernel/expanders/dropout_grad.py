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
"""generate json desc for DropoutGrad"""
from ._utils import Expander, ExpanderInfoValidator as VLD


@VLD.check_all_formats_same
@VLD.check_attrs('keep_prob')
class DropoutGrad(Expander):
    """DropoutGrad expander"""

    def _expand(self, graph_builder):
        input_dy, input_mask = self.inputs
        keep_prob = self.attrs['keep_prob']
        r_keep_prob = graph_builder.value(input_dy.dtype, 1.0 / keep_prob)
        result = graph_builder.emit('Mul', [input_dy, r_keep_prob])
        result = graph_builder.emit('Mul', [result, input_mask])
        return result
