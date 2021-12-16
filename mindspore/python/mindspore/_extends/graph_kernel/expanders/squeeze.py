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
"""generate json desc for squeeze"""
from ._utils import Expander, ExpanderInfoValidator as VLD


@VLD.check_attrs('axis')
class Squeeze(Expander):
    """Squeeze expander"""

    def _expand(self, graph_builder):
        input_x = self.inputs[0]
        out_shape = self.infer_shape(input_x.shape, self.attrs['axis'])
        result = graph_builder.emit('Reshape', [input_x], attrs={'shape': out_shape})

        return result

    @staticmethod
    def infer_shape(shape, axis):
        """infer shape for squeeze"""
        def squeeze_axis(shape, axis):
            if not axis:
                out_shape = [d for d in shape if d != 1]
            else:
                ndim = len(shape)
                out_shape = [shape[i] for i in range(ndim) if not (i in axis or (i - ndim) in axis)]
            if not out_shape:
                out_shape = [1]
            return out_shape
        if isinstance(shape, (list, tuple)):
            if isinstance(axis, int):
                axis = [axis]
            if isinstance(axis, (list, tuple)):
                return squeeze_axis(shape, axis)
        raise ValueError("Invalid axis for Squeeze.")
