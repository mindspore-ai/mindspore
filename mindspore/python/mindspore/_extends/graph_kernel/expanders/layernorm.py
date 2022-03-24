# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""generate json desc for LayerNorm"""
from mindspore._extends.graph_kernel.model.model import DataFormat as DF
from ._utils import Expander, ExpanderInfoValidator as VLD
from ._utils import infer_shape_from_fractalnz, get_reduced_ori_shape, to_frac_z_axis


@VLD.add_format(DF.FRAC_NZ, DF.DEFAULT, DF.DEFAULT)
@VLD.add_format(DF.DEFAULT, DF.DEFAULT, DF.DEFAULT)
@VLD.check_attrs('begin_norm_axis', 'begin_params_axis', 'epsilon')
class LayerNorm(Expander):
    """LayerNorm expander"""

    def _expand(self, graph_builder):
        input_x, input_gamma, input_beta = self.inputs
        processor = self.processor
        begin_norm_axis = self.attrs['begin_norm_axis']
        epsilon = self.attrs['epsilon']

        ori_dtype = input_x.dtype
        if processor == 'aicore' and ori_dtype == 'float16':
            input_x = graph_builder.emit('Cast', [input_x], attrs={'dst_type': 'float32'})
            input_gamma = graph_builder.emit('Cast', [input_gamma], attrs={'dst_type': 'float32'})
            input_beta = graph_builder.emit('Cast', [input_beta], attrs={'dst_type': 'float32'})

        ori_shape_x = input_x.shape
        if input_x.data_format == DF.FRAC_NZ:
            ori_shape_x = infer_shape_from_fractalnz(input_x.shape)

        # Calculate the scaling ratio of the average
        if begin_norm_axis < 0:
            begin_norm_axis += len(ori_shape_x)

        reduce_axis = ()
        for i, _ in enumerate(ori_shape_x):
            if i > begin_norm_axis or i == begin_norm_axis:
                reduce_axis = reduce_axis + (i,)

        reduce_elts = 1.0
        for i in reduce_axis:
            reduce_elts *= ori_shape_x[i]
        # after reduced
        ori_reduced_shape_x = get_reduced_ori_shape(ori_shape_x, reduce_axis)

        axis = reduce_axis
        if input_x.data_format == DF.FRAC_NZ:
            axis = to_frac_z_axis(ori_shape_x, reduce_axis)

        mean_cof_v = graph_builder.value(input_x.dtype, 1.0 / reduce_elts)

        # Calculate mean
        mean_red = graph_builder.emit('ReduceSum', [input_x], attrs={'reduce_axis': axis, 'keep_dims': True})
        mean = graph_builder.emit('Mul', [mean_red, mean_cof_v])
        if input_x.data_format == DF.FRAC_NZ:
            mean = graph_builder.emit('Reshape', [mean], attrs={'shape': ori_reduced_shape_x})

        # Calculate variance
        variance_sub = graph_builder.emit('Sub', [input_x, mean])
        variance_mul = graph_builder.emit('Mul', [variance_sub, variance_sub])
        variance_red = graph_builder.emit('ReduceSum', [variance_mul], attrs={'reduce_axis': axis, 'keep_dims': True})
        variance = graph_builder.emit('Mul', [variance_red, mean_cof_v])
        if input_x.data_format == DF.FRAC_NZ:
            variance = graph_builder.emit('Reshape', [variance], attrs={'shape': ori_reduced_shape_x})

        # Calculate normalize
        normalize_sub = graph_builder.emit('Sub', [input_x, mean])
        epsilon_v = graph_builder.value(input_x.dtype, epsilon)
        normalize_add = graph_builder.emit('Add', [variance, epsilon_v])
        normlize_rsqrt = graph_builder.emit('Rsqrt', [normalize_add])
        normalize_mul = graph_builder.emit('Mul', [normalize_sub, normlize_rsqrt])

        # Calculate scale and translate
        scale_mul = graph_builder.emit('Mul', [normalize_mul, input_gamma])
        res = graph_builder.emit('Add', [scale_mul, input_beta])

        if processor == 'aicore' and ori_dtype == 'float16':
            res = graph_builder.emit('Cast', [res], attrs={'dst_type': 'float16'})
            mean = graph_builder.emit('Cast', [mean], attrs={'dst_type': 'float16'})
            variance = graph_builder.emit('Cast', [variance], attrs={'dst_type': 'float16'})
        return res, mean, variance
