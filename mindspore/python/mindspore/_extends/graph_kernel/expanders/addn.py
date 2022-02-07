# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""generate json desc for addn"""
from mindspore._extends.graph_kernel.model.model import GraphKernelUnsupportedException as GKException
from ._utils import Expander, ExpanderInfoValidator as VLD


@VLD.check_all_formats_same
class AddN(Expander):
    """Expand AddN to multiple Adds"""

    def _check(self):
        if len(self.inputs) < 2:
            raise GKException("For 'AddN', the inputs num should be greater than 1, but got {}"
                              .format(len(self.inputs)))

    def _expand(self, graph_builder):
        result = self.inputs[0]
        for inp in self.inputs[1:]:
            result = graph_builder.emit('Add', [result, inp])
        return result
