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
"""generate json desc for BatchNormGrad"""
from mindspore._extends.graph_kernel.model.model import DataFormat as DF
from ._utils import Expander, ExpanderInfoValidator as VLD
from .expand_dims import ExpandDims


@VLD.add_format(DF.NHWC, DF.NHWC, DF.DEFAULT, DF.DEFAULT, DF.DEFAULT, DF.DEFAULT)
@VLD.add_format(DF.NCHW, DF.NCHW, DF.DEFAULT, DF.DEFAULT, DF.DEFAULT, DF.DEFAULT)
@VLD.add_format(DF.DEFAULT, DF.DEFAULT, DF.DEFAULT, DF.DEFAULT, DF.DEFAULT, DF.DEFAULT)
@VLD.check_attrs('is_training', 'epsilon')
class BatchNormGrad(Expander):
    """BatchNormGrad expander"""

    def _expand(self, graph_builder):
        # get op info
        input_dy = self.inputs[0]
        input_x = self.inputs[1]
        input_scale = self.inputs[2]
        input_save_mean = self.inputs[3]
        input_save_inv_variance = self.inputs[4]

        reduce_axis = ()
        shape_x = input_x.shape
        if input_x.data_format == DF.NHWC:
            reduce_axis = (0, 1, 2)
            num = shape_x[0] * shape_x[1] * shape_x[2]
        else:
            reduce_axis = (0, 2, 3)
            num = shape_x[0] * shape_x[2] * shape_x[3]
        ori_type = input_x.dtype
        if ori_type == 'float16':
            input_x = graph_builder.emit('Cast', [input_x], attrs={'dst_type': 'float32'})
        if input_dy.dtype == 'float16':
            input_dy = graph_builder.emit('Cast', [input_dy], attrs={'dst_type': 'float32'})
        num_rec = -1.0 / num
        num_rec_v = graph_builder.value(input_scale.dtype, num_rec)
        dbeta = graph_builder.emit('ReduceSum', [input_dy], attrs={'reduce_axis': reduce_axis, 'keep_dims': False})

        # in training input_save_inv_variance means 1 / sqrt(variance + epsilon), which is calculated in forward pass
        if self.attrs['is_training']:
            inv_variance = input_save_inv_variance
        else:
            epsilon_v = graph_builder.value(input_scale.dtype, self.attrs['epsilon'])
            var_add = graph_builder.emit('Add', [input_save_inv_variance, epsilon_v])
            sqrt_var_eps = graph_builder.emit('Sqrt', [var_add])
            scalar_one = 1.0
            scalar_one_v = graph_builder.value(input_scale.dtype, scalar_one)
            inv_variance = graph_builder.emit('RealDiv', [scalar_one_v, sqrt_var_eps])

        # compute dgamma
        if input_x.data_format in (DF.DEFAULT, DF.NCHW):
            input_save_mean = graph_builder.emit(
                'Reshape', [input_save_mean], attrs={'shape': ExpandDims.infer_shape(input_save_mean.shape, [-1, -1])})
            inv_variance = graph_builder.emit(
                'Reshape', [inv_variance], attrs={'shape': ExpandDims.infer_shape(inv_variance.shape, [-1, -1])})
            input_scale = graph_builder.emit(
                'Reshape', [input_scale], attrs={'shape': ExpandDims.infer_shape(input_scale.shape, [-1, -1])})
        x_sub_mean = graph_builder.emit('Sub', [input_x, input_save_mean])
        x_div = graph_builder.emit('Mul', [x_sub_mean, inv_variance])
        dgamma_param = graph_builder.emit('Mul', [input_dy, x_div])
        dgamma = graph_builder.emit(
            'ReduceSum', [dgamma_param], attrs={'reduce_axis': reduce_axis, 'keep_dims': False})

        # compute dx
        if self.attrs['is_training']:
            tmp_b = graph_builder.emit('Mul', [num_rec_v, dbeta])
            if input_x.data_format in (DF.DEFAULT, DF.NCHW):
                dgamma_expand = graph_builder.emit(
                    'Reshape', [dgamma], attrs={'shape': ExpandDims.infer_shape(dgamma.shape, [-1, -1])})
                tmp_b = graph_builder.emit(
                    'Reshape', [tmp_b], attrs={'shape': ExpandDims.infer_shape(tmp_b.shape, [-1, -1])})
            else:
                dgamma_expand = dgamma
            x_sub_mean_dgamma_mul = graph_builder.emit('Mul', [x_div, dgamma_expand])
            tmp_c = graph_builder.emit('Mul', [num_rec_v, x_sub_mean_dgamma_mul])
            tmp_ab_add = graph_builder.emit('Add', [input_dy, tmp_b])
            tmp_abc_add = graph_builder.emit('Add', [tmp_ab_add, tmp_c])
            gamma_mul = graph_builder.emit('Mul', [input_scale, tmp_abc_add])
            dx = graph_builder.emit('Mul', [inv_variance, gamma_mul])
        else:
            y_scale = graph_builder.emit('Mul', [input_scale, input_dy])
            dx = graph_builder.emit('Mul', [inv_variance, y_scale])
        if ori_type == 'float16':
            dx = graph_builder.emit('Cast', [dx], attrs={'dst_type': 'float16'})

        # set output tensors' data_format
        dx.data_format = self.outputs[0]['format']
        dgamma.data_format = self.outputs[1]['format']
        dbeta.data_format = self.outputs[2]['format']

        return dx, dgamma, dbeta
