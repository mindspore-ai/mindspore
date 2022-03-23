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
"""generate json desc for slice"""
from ._utils import Expander, ExpanderInfoValidator as VLD


@VLD.check_attrs('begin', 'size')
class Slice(Expander):
    """Slice expander"""

    def _expand(self, graph_builder):
        input_x = self.inputs[0]
        begin = self.attrs['begin']
        size = self.attrs['size']
        end = []
        strides = []
        for i, begin_idx in enumerate(begin):
            strides.append(1)
            end.append(begin_idx + size[i])
        output = graph_builder.tensor(size, input_x.dtype, input_x.data_format)
        graph_builder.op('StridedSlice', output, [input_x], attrs={'begin': begin, 'end': end, 'strides': strides})

        return output
