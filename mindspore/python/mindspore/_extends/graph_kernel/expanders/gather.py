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
"""generate json desc for gather"""
from ._utils import Expander, ExpanderInfoValidator as VLD


@VLD.check_all_formats_same
@VLD.check_attrs('axis')
class Gather(Expander):
    """Expand Gather"""

    def _expand(self, graph_builder):
        inputs, indices = self.inputs
        axis = self.attrs['axis']
        if axis < 0:
            axis += len(inputs.shape)
        if len(indices.shape) == 1:
            result = graph_builder.emit('Gather', [inputs, indices], attrs={'axis': axis})
        else:
            ori_indices_shape = indices.shape
            indices_shape_one_dim = 1
            for dim in ori_indices_shape:
                indices_shape_one_dim *= dim
            new_indices_shape = [indices_shape_one_dim]
            reshape_indices = graph_builder.emit('Reshape', [indices], attrs={'shape': new_indices_shape})
            tmp_result = graph_builder.emit('Gather', [inputs, reshape_indices], attrs={'axis': axis})
            output_shape = inputs.shape.copy()
            output_shape[axis:axis] = ori_indices_shape
            del output_shape[axis + len(ori_indices_shape)]
            result = graph_builder.emit('Reshape', [tmp_result], attrs={'shape': output_shape})
        return result
