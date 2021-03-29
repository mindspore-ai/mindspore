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
"""generate json desc for LayerNorm"""
from mindspore._extends.graph_kernel.model.model import DataFormat as DF
from ._utils import Expander, ExpanderInfoValidator as VLD


@VLD.add_format(DF.FRAC_NZ, DF.DEFAULT, DF.DEFAULT)
@VLD.add_format(DF.DEFAULT, DF.DEFAULT, DF.DEFAULT)
@VLD.check_attrs('begin_norm_axis', 'begin_params_axis', 'epsilon')
class LayerNorm(Expander):
    """LayerNorm expander"""

    def to_frac_z_axis(self, ori_shape, ori_axis):
        """
        judge the format is fractal NZ

        Parameters
        ----------
        ori_shape: list or tuple
            original shape of input
        ori_axis: list or tuple
            original axis of original shape to operate

        Returns
        -------
        output: list
            axis of the fractal Nz shape
        """

        frac_z_axis = list(ori_axis)
        shape_len = len(ori_shape)
        axis_count = len(frac_z_axis)
        axis_negative_1 = shape_len - 1
        axis_negative_2 = shape_len - 2

        for i in range(axis_count):
            axis_index = (frac_z_axis[i] + shape_len) % shape_len
            if axis_index == axis_negative_1:
                if frac_z_axis[i] > shape_len - 2: # akg:[2,3] [1,4] tbe:[2,4] [1,3]
                    frac_z_axis[i] = axis_index - 1
                    frac_z_axis.append(axis_index + 2)
                else: # no case cover this branch now
                    frac_z_axis[i] = axis_index - 1
                    frac_z_axis.append(axis_index + 2)
            elif axis_index == axis_negative_2:
                frac_z_axis[i] = axis_index + 1
                frac_z_axis.append(axis_index + 2)
            else:
                frac_z_axis[i] = axis_index
        return frac_z_axis

    def infer_shape_from_fractalNz(self, fractal):
        "get original shape from fractalNz shape"
        shape = []
        dims = len(fractal)
        batch = dims - 4
        for i in range(batch):
            shape.append(fractal[i])

        m = fractal[dims - 3] * fractal[dims - 2]
        n = fractal[dims - 4] * fractal[dims - 1]
        shape.append(m)
        shape.append(n)

        return shape

    def get_reduced_ori_shape(self, shape, axis):
        "get shape after reduced which is based on original shape"
        reduced_ori_shape = []
        for i, value in enumerate(shape):
            if i in axis:
                reduced_ori_shape.append(1)
            else:
                reduced_ori_shape.append(value)
        return reduced_ori_shape

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
            ori_shape_x = self.infer_shape_from_fractalNz(input_x.shape)

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

        if input_x.data_format == DF.FRAC_NZ:
            reduce_axis = self.to_frac_z_axis(ori_shape_x, reduce_axis)
            ori_shape_x = self.get_reduced_ori_shape(ori_shape_x, reduce_axis) # after reduced

        mean_cof = 1.0 / reduce_elts
        mean_cof_v = graph_builder.value(input_x.dtype, mean_cof)

        # Calculate mean
        mean_red = graph_builder.emit('ReduceSum', [input_x],
                                      attrs={'reduce_axis': reduce_axis, 'keep_dims': True})
        mean = graph_builder.emit('Mul', [mean_red, mean_cof_v])
        if input_x.data_format == DF.FRAC_NZ:
            mean = graph_builder.emit('Reshape', [mean], attrs={'shape': ori_shape_x})

        # Calculate variance
        variance_sub = graph_builder.emit('Sub', [input_x, mean])
        variance_mul = graph_builder.emit('Mul', [variance_sub, variance_sub])
        variance_red = graph_builder.emit('ReduceSum', [variance_mul],
                                          attrs={'reduce_axis': reduce_axis, 'keep_dims': True})
        variance = graph_builder.emit('Mul', [variance_red, mean_cof_v])
        if input_x.data_format == DF.FRAC_NZ:
            variance = graph_builder.emit('Reshape', [variance], attrs={'shape': ori_shape_x})

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
