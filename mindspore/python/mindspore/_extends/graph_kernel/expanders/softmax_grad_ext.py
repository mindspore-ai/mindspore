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
"""generate json desc for SoftmaxGradExt"""
from mindspore._extends.graph_kernel.model.model import DataFormat as DF
from ._utils import Expander, ExpanderInfoValidator as VLD
from ._utils import get_reduce_axis_shape


@VLD.add_format(DF.FRAC_NZ, DF.FRAC_NZ, DF.DEFAULT)
@VLD.add_format(DF.DEFAULT, DF.DEFAULT, DF.DEFAULT)
@VLD.check_attrs('axis')
class SoftmaxGradExt(Expander):
    """SoftmaxGradExt expander"""

    def _expand(self, graph_builder):
        x, y, z = self.inputs
        axis = self.attrs['axis']

        reduce_axis, ori_reduced_shape = get_reduce_axis_shape(x.shape, x.data_format, axis)

        data_mul = graph_builder.emit('Mul', [x, y])
        data_sum = graph_builder.emit('ReduceSum', [data_mul],
                                      attrs={'reduce_axis': reduce_axis, 'keep_dims': True, 'reduce_output_fuse': True})
        if x.data_format == DF.FRAC_NZ:
            data_sum = graph_builder.emit('Reshape', [data_sum], attrs={'shape': ori_reduced_shape})
        data_sub = graph_builder.emit('Sub', [x, data_sum])
        data_mul2 = graph_builder.emit('Mul', [data_sub, y])
        result = graph_builder.emit('Mul', [data_mul2, z])
        return result
