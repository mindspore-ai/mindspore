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
"""generate json desc for equal_count"""
from mindspore._extends.graph_kernel.model.model import GraphKernelUnsupportedException as GKException
from ._utils import Expander, ExpanderInfoValidator as VLD


@VLD.check_all_formats_same
class EqualCount(Expander):
    """EqualCount expander"""

    def __init__(self, expand_info):
        super().__init__(expand_info)
        self.shape_x = self.inputs[0]['shape']
        self.shape_y = self.inputs[1]['shape']
        self.dtype_x = self.inputs[0]['data_type']
        self.dtype_y = self.inputs[1]['data_type']

    def _check(self):
        if self.shape_x != self.shape_y:
            raise GKException("For 'EqualCount', the inputs shape should be same, but got {} and {}"
                              .format(self.shape_x, self.shape_y))
        if self.dtype_x != self.dtype_y:
            raise GKException("For 'EqualCount', the inputs data type should be same, but got {} and {}"
                              .format(self.dtype_x, self.dtype_y))

    def _expand(self, graph_builder):
        input_x = self.inputs[0]
        input_y = self.inputs[1]

        eql_val = graph_builder.emit('Equal', [input_x, input_y])
        cast_val = graph_builder.emit('Cast', [eql_val], attrs={'dst_type': 'float32'})
        axis = list(range(len(input_x.shape)))
        result = graph_builder.emit('ReduceSum', [cast_val], attrs={'reduce_axis': axis, 'keep_dims': False})

        if result.dtype != input_x.dtype:
            result = graph_builder.emit('Cast', [result], attrs={'dst_type': input_x.dtype})
        return result
