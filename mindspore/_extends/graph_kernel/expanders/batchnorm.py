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
        epsilon_v = graph_builder.value(input_scale.dtype, self.attrs['epsilon'], input_scale.data_format)

        if self.attrs['is_training']:
            reduce_axis = ()
            shape_x = input_x.shape
            if input_x.data_format == "NHWC":
                reduce_axis = (0, 1, 2)
                num = shape_x[0] * shape_x[1] * shape_x[2]
            else:
                reduce_axis = (0, 2, 3)
                num = shape_x[0] * shape_x[2] * shape_x[3]
            num_rec = 1.0 / num
            num_rec_v = graph_builder.value(input_scale.dtype, num_rec, input_scale.data_format)

            # compute mean value of input_x
            mean_sum = graph_builder.emit(
                'ReduceSum', [input_x], attrs={'reduce_axis': reduce_axis, 'keep_dims': False})
            mean_muls = graph_builder.emit('Mul', [mean_sum, num_rec_v])

            # compute variance of input_x
            if not input_x.data_format == "NHWC":
                mean_muls_expand = graph_builder.emit('ExpandDims', [mean_muls], attrs={'axis': 1})
                mean_muls_expand = graph_builder.emit('ExpandDims', [mean_muls_expand], attrs={'axis': 2})
            else:
                mean_muls_expand = mean_muls
            var_sub = graph_builder.emit('Sub', [input_x, mean_muls_expand])
            var_mul = graph_builder.emit('Mul', [var_sub, var_sub])
            var_sum = graph_builder.emit('ReduceSum', [var_mul], attrs={'reduce_axis': reduce_axis, 'keep_dims': False})
            var_mul = graph_builder.emit('Mul', [var_sum, num_rec_v])

            # y_sqrt_rec means 1 / sqrt(variance + epsilon), which is calculated in backward pass
            scalar_one = 1.0
            scalar_one_v = graph_builder.value(input_scale.dtype, scalar_one, input_scale.data_format)
            y_add = graph_builder.emit('Add', [var_mul, epsilon_v])
            y_sqrt = graph_builder.emit('Sqrt', [y_add])
            y_sqrt_rec = graph_builder.emit('RealDiv', [scalar_one_v, y_sqrt])

            # compute res_y
            tmp_sub = graph_builder.emit('Sub', [input_x, mean_muls_expand])
            if not input_x.data_format == "NHWC":
                y_sqrt_rec_expand = graph_builder.emit('ExpandDims', [y_sqrt_rec], attrs={'axis': 1})
                y_sqrt_rec_expand = graph_builder.emit('ExpandDims', [y_sqrt_rec_expand], attrs={'axis': 2})
            else:
                y_sqrt_rec_expand = y_sqrt_rec
            y_norm = graph_builder.emit('Mul', [tmp_sub, y_sqrt_rec_expand])
            if not input_x.data_format == "NHWC":
                input_scale_expand = graph_builder.emit('ExpandDims', [input_scale], attrs={'axis': 1})
                input_scale_expand = graph_builder.emit('ExpandDims', [input_scale_expand], attrs={'axis': 2})
            else:
                input_scale_expand = input_scale
            res_y_mul = graph_builder.emit('Mul', [input_scale_expand, y_norm])
            if not input_x.data_format == "NHWC":
                input_offset_expand = graph_builder.emit('ExpandDims', [input_offset], attrs={'axis': 1})
                input_offset_expand = graph_builder.emit('ExpandDims', [input_offset_expand], attrs={'axis': 2})
            else:
                input_offset_expand = input_offset
            res_y = graph_builder.emit('Add', [res_y_mul, input_offset_expand])

            # compute mean_res
            momentum_sub = scalar_one - self.attrs['momentum']
            momentum_v_sub = graph_builder.value(input_scale.dtype, momentum_sub, input_scale.data_format)
            new_running_mean_tmp = graph_builder.emit('Mul', [momentum_v_sub, input_mean])
            momentum_v = graph_builder.value(input_scale.dtype, self.attrs['momentum'], input_scale.data_format)
            current_mean_tmp = graph_builder.emit('Mul', [momentum_v, mean_muls])
            updated_moving_mean = graph_builder.emit('Add', [new_running_mean_tmp, current_mean_tmp])
            mean_res = graph_builder.emit(
                'InplaceAssign', [input_mean, updated_moving_mean, updated_moving_mean], attrs={'fake_output': True})

            # variance_res is calculated by sample variance, and need to multiply by num / (num - 1)
            var_num = float(num) / (num - 1)
            var_num_v = graph_builder.value(input_scale.dtype, var_num, input_scale.data_format)
            var_mul_update = graph_builder.emit('Mul', [var_num_v, var_mul])
            new_running_var_tmp = graph_builder.emit('Mul', [momentum_v_sub, input_variance])
            current_var_tmp = graph_builder.emit('Mul', [momentum_v, var_mul_update])
            updated_moving_variance = graph_builder.emit('Add', [new_running_var_tmp, current_var_tmp])
            variance_res = graph_builder.emit(
                'InplaceAssign', [input_variance, updated_moving_variance, updated_moving_variance],
                attrs={'fake_output': True})

            # compute reverse, just return a C shape tensor
            reserve = graph_builder.emit('Add', [input_offset, scalar_one_v])
            return res_y, mean_res, variance_res, mean_muls, y_sqrt_rec, reserve
        # infer mode
        if not input_x.data_format == "NHWC":
            input_mean = graph_builder.emit('ExpandDims', [input_mean], attrs={'axis': 1})
            input_mean = graph_builder.emit('ExpandDims', [input_mean], attrs={'axis': 2})
            input_scale = graph_builder.emit('ExpandDims', [input_scale], attrs={'axis': 1})
            input_scale = graph_builder.emit('ExpandDims', [input_scale], attrs={'axis': 2})
            input_offset = graph_builder.emit('ExpandDims', [input_offset], attrs={'axis': 1})
            input_offset = graph_builder.emit('ExpandDims', [input_offset], attrs={'axis': 2})
        x_sub = graph_builder.emit('Sub', [input_x, input_mean])
        x_sub_mul = graph_builder.emit('Mul', [input_scale, x_sub])
        var_add = graph_builder.emit('Add', [epsilon_v, input_variance])
        var_add_sqrt = graph_builder.emit('Sqrt', [var_add])
        if not input_x.data_format == "NHWC":
            var_add_sqrt = graph_builder.emit('ExpandDims', [var_add_sqrt], attrs={'axis': 1})
            var_add_sqrt = graph_builder.emit('ExpandDims', [var_add_sqrt], attrs={'axis': 2})
        x_div = graph_builder.emit('RealDiv', [x_sub_mul, var_add_sqrt])
        res_y = graph_builder.emit('Add', [input_offset, x_div])
        return res_y, var_add, var_add, var_add, var_add
