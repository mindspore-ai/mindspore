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
"""generate json desc for squared_difference"""
from mindspore._extends.graph_kernel.model.model import GraphKernelUnsupportedException as GKException
from ._utils import Expander, ExpanderInfoValidator as VLD


@VLD.check_all_formats_same
class SquaredDifference(Expander):
    """SquaredDifference expander"""

    def __init__(self, expand_info):
        super().__init__(expand_info)
        self.dtype_x = self.inputs[0]['data_type']
        self.dtype_y = self.inputs[1]['data_type']

    def _check(self):
        if self.dtype_x == "float64" or self.dtype_y == "float64":
            raise GKException("For 'SquaredDifference', the inputs data type must not be float64")
        if self.dtype_x != self.dtype_y:
            raise GKException("For 'SquaredDifference', the inputs data type should be same, but got {} and {}"
                              .format(self.dtype_x, self.dtype_y))

    def _expand(self, graph_builder):
        input_x = self.inputs[0]
        input_y = self.inputs[1]

        sub_val = graph_builder.emit('Sub', [input_x, input_y])
        result = graph_builder.emit('Mul', [sub_val, sub_val])

        return result
