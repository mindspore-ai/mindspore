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
"""generate json desc for reduce_mean"""
from mindspore._extends.graph_kernel.model.model import DataFormat as DF
from ._utils import Expander, ExpanderInfoValidator as VLD


@VLD.add_format(DF.DEFAULT)
@VLD.check_attrs('axis', 'keep_dims')
class ReduceMean(Expander):
    """ReduceMean expander"""

    def _expand(self, graph_builder):
        x = self.inputs[0]
        axis = self.attrs['axis']
        keep_dims = self.attrs['keep_dims']

        if not isinstance(axis, (tuple, list)):
            axis = (axis,)
        elif not axis:
            axis = list(range(len(x.shape)))
        reduce_size = 1.0
        for idx in axis:
            reduce_size *= x.shape[idx]

        reduce_size_value = graph_builder.value(x.dtype, reduce_size)

        sum_x = graph_builder.emit('ReduceSum', [x], attrs={'reduce_axis': axis, 'keep_dims': keep_dims})
        result = graph_builder.emit('RealDiv', [sum_x, reduce_size_value])

        return result
