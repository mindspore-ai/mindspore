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
"""generate json desc for SquareSumV1"""
from mindspore._extends.graph_kernel.model.model import DataFormat as DF
from ._utils import Expander, ExpanderInfoValidator as VLD
from ._utils import infer_shape_from_fractalnz, get_reduced_ori_shape, to_frac_z_axis


@VLD.add_format(DF.FRAC_NZ)
@VLD.add_format(DF.DEFAULT)
@VLD.check_attrs('axis')
class SquareSumV1(Expander):
    """Square expander"""

    def _expand(self, graph_builder):
        x = self.inputs[0]
        axis = self.attrs['axis']

        ori_shape = x.shape
        if x.data_format == DF.FRAC_NZ:
            ori_shape = infer_shape_from_fractalnz(ori_shape)
        if not axis:
            axis = []
            for i, _ in enumerate(ori_shape):
                axis.append(i)
        else:
            if isinstance(axis, int):
                axis = [axis]
            for i, _ in enumerate(list(axis)):
                if axis[i] < 0:
                    axis[i] += len(ori_shape)

        ori_reduced_shape = ori_shape
        if x.data_format == DF.FRAC_NZ:
            ori_reduced_shape = get_reduced_ori_shape(ori_shape, axis)
            axis = to_frac_z_axis(ori_shape, axis)

        square_res = graph_builder.emit('Mul', [x, x])
        result = graph_builder.emit('ReduceSum', [square_res], attrs={'reduce_axis': axis, 'keep_dims': False})
        if x.data_format == DF.FRAC_NZ:
            result = graph_builder.emit('Reshape', [result], attrs={'shape': ori_reduced_shape})
        return result
