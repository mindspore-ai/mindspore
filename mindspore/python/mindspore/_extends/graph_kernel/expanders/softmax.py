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
"""generate json desc for softmax"""
from mindspore._extends.graph_kernel.model.model import DataFormat as DF
from ._utils import Expander, ExpanderInfoValidator as VLD
from ._utils import infer_shape_from_fractalnz, get_reduced_ori_shape, to_frac_z_axis


@VLD.add_format(DF.FRAC_NZ)
@VLD.add_format(DF.DEFAULT)
@VLD.check_attrs('axis')
class Softmax(Expander):
    """Softmax expander"""

    def _expand(self, graph_builder):
        input_x = self.inputs[0]
        processor = self.processor
        axis = self.attrs['axis']

        ori_shape = input_x.shape
        if input_x.data_format == DF.FRAC_NZ:
            ori_shape = infer_shape_from_fractalnz(input_x.shape)

        for i, _ in enumerate(list(axis)):
            if axis[i] < 0:
                axis[i] += len(ori_shape)

        ori_reduced_shape = get_reduced_ori_shape(ori_shape, axis)

        if input_x.data_format == DF.FRAC_NZ:
            axis = to_frac_z_axis(ori_shape, axis)

        ori_dtype = input_x.dtype
        if ori_dtype != "float16" and processor == "aicore":
            input_x_f16 = graph_builder.emit('Cast', [input_x], attrs={'dst_type': 'float16'})
            max_x_f16 = graph_builder.emit('ReduceMax', [input_x_f16], attrs={'reduce_axis': axis, 'keep_dims': True})
            max_x = graph_builder.emit('Cast', [max_x_f16], attrs={'dst_type': ori_dtype})
        else:
            max_x = graph_builder.emit('ReduceMax', [input_x], attrs={'reduce_axis': axis, 'keep_dims': True})

        if ori_dtype == "float16" and processor == "aicore":
            max_x = graph_builder.emit('Cast', [max_x], attrs={'dst_type': "float32"})
            input_x = graph_builder.emit('Cast', [input_x], attrs={'dst_type': "float32"})

        if input_x.data_format == DF.FRAC_NZ:
            max_x = graph_builder.emit('Reshape', [max_x], attrs={'shape': ori_reduced_shape})
        data_sub = graph_builder.emit('Sub', [input_x, max_x])
        data_exp = graph_builder.emit('Exp', [data_sub])
        data_expsum = graph_builder.emit('ReduceSum', [data_exp], attrs={'reduce_axis': axis, 'keep_dims': True})
        if input_x.data_format == DF.FRAC_NZ:
            data_expsum = graph_builder.emit('Reshape', [data_expsum], attrs={'shape': ori_reduced_shape})
        result = graph_builder.emit('RealDiv', [data_exp, data_expsum])
        if ori_dtype == "float16" and processor == "aicore":
            result = graph_builder.emit('Cast', [result], attrs={'dst_type': ori_dtype})

        return result
