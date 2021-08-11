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
"""generate json desc for BatchNorm"""
from mindspore._extends.graph_kernel.model.model import DataFormat as DF
from ._utils import Expander, ExpanderInfoValidator as VLD
from .expand_dims import ExpandDims


@VLD.add_format(DF.NHWC, DF.DEFAULT, DF.DEFAULT, DF.DEFAULT, DF.DEFAULT)
@VLD.add_format(DF.NCHW, DF.DEFAULT, DF.DEFAULT, DF.DEFAULT, DF.DEFAULT)
@VLD.add_format(DF.DEFAULT, DF.DEFAULT, DF.DEFAULT, DF.DEFAULT, DF.DEFAULT)
@VLD.check_attrs('is_training', 'momentum', 'epsilon')
class BatchNorm(Expander):
    """BatchNorm expander"""

    def _expand(self, graph_builder):
        # get op info
        input_x = self.inputs[0]
        input_scale = self.inputs[1]
        input_offset = self.inputs[2]
        input_mean = self.inputs[3]
        input_variance = self.inputs[4]
        epsilon_v = graph_builder.value(input_scale.dtype, self.attrs['epsilon'])

        input_x_ori_type = input_x.dtype
        input_x_new_type = input_x.dtype
        if input_x.dtype == "float16" and input_scale.dtype == "float32" and input_offset.dtype == "float32" and \
                input_mean.dtype == "float32" and input_variance.dtype == "float32":
            input_x_new_type = "float32"
        if input_x_new_type != input_x_ori_type:
            input_x = graph_builder.emit('Cast', [input_x], attrs={'dst_type': input_x_new_type})

        if self.attrs['is_training']:
            self.inputs[0] = input_x
            res_y, mean_res, variance_res, mean_muls, y_sqrt_rec = self._bn_train(graph_builder)
            if input_x_new_type != input_x_ori_type:
                res_y = graph_builder.emit('Cast', [res_y], attrs={'dst_type': input_x_ori_type})
            return res_y, mean_res, variance_res, mean_muls, y_sqrt_rec
        # infer mode
        if input_x.data_format in (DF.DEFAULT, DF.NCHW):
            input_mean = graph_builder.emit(
                'Reshape', [input_mean], attrs={'shape': ExpandDims.infer_shape(input_mean.shape, [-1, -1])})
            input_scale = graph_builder.emit(
                'Reshape', [input_scale], attrs={'shape': ExpandDims.infer_shape(input_scale.shape, [-1, -1])})
            input_offset = graph_builder.emit(
                'Reshape', [input_offset], attrs={'shape': ExpandDims.infer_shape(input_offset.shape, [-1, -1])})
        x_sub = graph_builder.emit('Sub', [input_x, input_mean])
        x_sub_mul = graph_builder.emit('Mul', [input_scale, x_sub])
        var_add = graph_builder.emit('Add', [epsilon_v, input_variance])
        var_add_sqrt = graph_builder.emit('Sqrt', [var_add])
        if input_x.data_format in (DF.DEFAULT, DF.NCHW):
            var_add_sqrt = graph_builder.emit(
                'Reshape', [var_add_sqrt], attrs={'shape': ExpandDims.infer_shape(var_add_sqrt.shape, [-1, -1])})
        x_div = graph_builder.emit('RealDiv', [x_sub_mul, var_add_sqrt])
        res_y = graph_builder.emit('Add', [input_offset, x_div])
        if input_x_new_type != input_x_ori_type:
            res_y = graph_builder.emit('Cast', [res_y], attrs={'dst_type': input_x_ori_type})
        return res_y, var_add, var_add, var_add, var_add

    def _bn_train(self, graph_builder):
        """expand BatchNorm for training mode"""
        input_x = self.inputs[0]
        input_scale = self.inputs[1]
        input_offset = self.inputs[2]
        input_mean = self.inputs[3]
        input_variance = self.inputs[4]
        epsilon_v = graph_builder.value(input_scale.dtype, self.attrs['epsilon'])
        reduce_axis = ()
        shape_x = input_x.shape
        if input_x.data_format == DF.NHWC:
            reduce_axis = (0, 1, 2)
            num = shape_x[0] * shape_x[1] * shape_x[2]
        else:
            reduce_axis = (0, 2, 3)
            num = shape_x[0] * shape_x[2] * shape_x[3]
        num_rec = 1.0 / num
        num_rec_v = graph_builder.value(input_scale.dtype, num_rec)

        # compute mean value of input_x
        mean_sum = graph_builder.emit(
            'ReduceSum', [input_x], attrs={'reduce_axis': reduce_axis, 'keep_dims': False})
        mean_muls = graph_builder.emit('Mul', [mean_sum, num_rec_v])

        # compute variance of input_x
        if input_x.data_format in (DF.DEFAULT, DF.NCHW):
            mean_muls_expand = graph_builder.emit(
                'Reshape', [mean_muls], attrs={'shape': ExpandDims.infer_shape(mean_muls.shape, [-1, -1])})
        else:
            mean_muls_expand = mean_muls
        var_sub = graph_builder.emit('Sub', [input_x, mean_muls_expand])
        var_mul = graph_builder.emit('Mul', [var_sub, var_sub])
        var_sum = graph_builder.emit('ReduceSum', [var_mul], attrs={'reduce_axis': reduce_axis, 'keep_dims': False})
        var_mul = graph_builder.emit('Mul', [var_sum, num_rec_v])

        # y_sqrt_rec means 1 / sqrt(variance + epsilon), which is calculated in backward pass
        scalar_one = 1.0
        scalar_one_v = graph_builder.value(input_scale.dtype, scalar_one)
        y_add = graph_builder.emit('Add', [var_mul, epsilon_v])
        y_sqrt = graph_builder.emit('Sqrt', [y_add])
        y_sqrt_rec = graph_builder.emit('RealDiv', [scalar_one_v, y_sqrt])

        # compute res_y
        tmp_sub = graph_builder.emit('Sub', [input_x, mean_muls_expand])
        if input_x.data_format in (DF.DEFAULT, DF.NCHW):
            y_sqrt_rec_expand = graph_builder.emit(
                'Reshape', [y_sqrt_rec], attrs={'shape': ExpandDims.infer_shape(y_sqrt_rec.shape, [-1, -1])})
        else:
            y_sqrt_rec_expand = y_sqrt_rec
        y_norm = graph_builder.emit('Mul', [tmp_sub, y_sqrt_rec_expand])
        if input_x.data_format in (DF.DEFAULT, DF.NCHW):
            input_scale_expand = graph_builder.emit(
                'Reshape', [input_scale], attrs={'shape': ExpandDims.infer_shape(input_scale.shape, [-1, -1])})
        else:
            input_scale_expand = input_scale
        res_y_mul = graph_builder.emit('Mul', [input_scale_expand, y_norm])
        if input_x.data_format in (DF.DEFAULT, DF.NCHW):
            input_offset_expand = graph_builder.emit(
                'Reshape', [input_offset], attrs={'shape': ExpandDims.infer_shape(input_offset.shape, [-1, -1])})
        else:
            input_offset_expand = input_offset
        res_y = graph_builder.emit('Add', [res_y_mul, input_offset_expand])

        # compute mean_res
        momentum_sub = scalar_one - self.attrs['momentum']
        momentum_v_sub = graph_builder.value(input_scale.dtype, momentum_sub)
        new_running_mean_tmp = graph_builder.emit('Mul', [momentum_v_sub, input_mean])
        momentum_v = graph_builder.value(input_scale.dtype, self.attrs['momentum'])
        current_mean_tmp = graph_builder.emit('Mul', [momentum_v, mean_muls])
        updated_moving_mean = graph_builder.emit('Add', [new_running_mean_tmp, current_mean_tmp])
        mean_res = graph_builder.emit(
            'InplaceAssign', [input_mean, updated_moving_mean, updated_moving_mean], attrs={'fake_output': True})

        # variance_res is calculated by sample variance, and need to multiply by num / (num - 1)
        var_num = float(num) / (num - 1)
        var_num_v = graph_builder.value(input_scale.dtype, var_num)
        var_mul_update = graph_builder.emit('Mul', [var_num_v, var_mul])
        new_running_var_tmp = graph_builder.emit('Mul', [momentum_v_sub, input_variance])
        current_var_tmp = graph_builder.emit('Mul', [momentum_v, var_mul_update])
        updated_moving_variance = graph_builder.emit('Add', [new_running_var_tmp, current_var_tmp])
        variance_res = graph_builder.emit(
            'InplaceAssign', [input_variance, updated_moving_variance, updated_moving_variance],
            attrs={'fake_output': True})
        return res_y, mean_res, variance_res, mean_muls, y_sqrt_rec
