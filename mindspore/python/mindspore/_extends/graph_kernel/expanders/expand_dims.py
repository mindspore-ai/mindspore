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
"""generate json desc for expand_dims"""
from ._utils import Expander, ExpanderInfoValidator as VLD


@VLD.check_attrs('axis')
class ExpandDims(Expander):
    """ExpandDims expander"""

    def _expand(self, graph_builder):
        input_x = self.inputs[0]
        shape = self.infer_shape(input_x.shape, self.attrs['axis'])
        result = graph_builder.emit('Reshape', [input_x], attrs={'shape': shape})

        return result

    @staticmethod
    def infer_shape(shape, axis):
        """infer shape for expand_dims"""
        def insert_axis(shape, axis):
            if not isinstance(axis, int) or axis > len(shape) or axis < -len(shape) - 1:
                raise ValueError("invalid dim for ExpandDims")
            if axis >= 0:
                shape.insert(axis, 1)
            else:
                shape.insert(axis + len(shape) + 1, 1)
            return shape
        out_shape = shape[:]
        if isinstance(axis, int):
            return insert_axis(out_shape, axis)
        if isinstance(axis, (list, tuple)):
            for i in axis:
                out_shape = insert_axis(out_shape, i)
            return out_shape
        raise ValueError("invalid dim for ExpandDims")
