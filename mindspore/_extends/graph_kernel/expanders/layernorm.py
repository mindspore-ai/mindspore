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


@VLD.add_format(DF.DEFAULT, DF.DEFAULT, DF.DEFAULT)
@VLD.check_attrs('begin_norm_axis', 'begin_params_axis', 'epsilon')
class LayerNorm(Expander):
    """LayerNorm expander"""

    def _expand(self, graph_builder):
        input_x, input_gamma, input_beta = self.inputs
        begin_norm_axis = self.attrs['begin_norm_axis']
        epsilon = self.attrs['epsilon']

        # Calculate the scaling ratio of the average
        if begin_norm_axis < 0:
            begin_norm_axis += len(input_x.shape)
        reduce_axis = ()
        for i, _ in enumerate(input_x.shape):
            if i > begin_norm_axis or i == begin_norm_axis:
                reduce_axis = reduce_axis + (i,)

        reduce_elts = 1.0
        for i in reduce_axis:
            reduce_elts *= input_x.shape[i]
        mean_cof = 1.0 / reduce_elts
        mean_cof_v = graph_builder.value(input_x.dtype, mean_cof)

        # Calculate mean
        mean_red = graph_builder.emit('ReduceSum', [input_x], attrs={'reduce_axis': reduce_axis, 'keep_dims': True})
        mean = graph_builder.emit('Mul', [mean_red, mean_cof_v])

        # Calculate variance
        variance_sub = graph_builder.emit('Sub', [input_x, mean])
        variance_mul = graph_builder.emit('Mul', [variance_sub, variance_sub])
        variance_red = graph_builder.emit('ReduceSum', [variance_mul],
                                          attrs={'reduce_axis': reduce_axis, 'keep_dims': True})
        variance = graph_builder.emit('Mul', [variance_red, mean_cof_v])

        # Calculate normalize
        normalize_sub = graph_builder.emit('Sub', [input_x, mean])
        epsilon_v = graph_builder.value(input_x.dtype, epsilon)
        normalize_add = graph_builder.emit('Add', [variance, epsilon_v])
        normlize_rsqrt = graph_builder.emit('Rsqrt', [normalize_add])
        normalize_mul = graph_builder.emit('Mul', [normalize_sub, normlize_rsqrt])

        # Calculate scale and translate
        scale_mul = graph_builder.emit('Mul', [input_gamma, normalize_mul])
        res = graph_builder.emit('Add', [scale_mul, input_beta])

        return res, mean, variance
