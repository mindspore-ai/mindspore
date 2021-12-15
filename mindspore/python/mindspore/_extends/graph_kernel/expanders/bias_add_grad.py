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
"""generate json desc for bias_add"""
from mindspore._extends.graph_kernel.model.model import DataFormat as DF
from ._utils import Expander, ExpanderInfoValidator as VLD


@VLD.add_format(DF.DEFAULT)
@VLD.add_format(DF.NHWC)
@VLD.add_format(DF.NCHW)
@VLD.add_format(DF.FRAC_NZ)
class BiasAddGrad(Expander):
    """BiasAddGrad expander"""

    def _expand(self, graph_builder):
        x = self.inputs[0]

        reduce_axis = ()
        if x.data_format == DF.NHWC:
            reduce_axis = (0, 1, 2)
        elif x.data_format == DF.NCHW:
            reduce_axis = (0, 2, 3)
        elif x.data_format == DF.FRAC_NZ:
            reduce_axis = (-2, -3)
        else:
            # DefaultFormat shape's length should be from 2 to 4
            if len(x.shape) == 2:
                reduce_axis = (0,)
            elif len(x.shape) == 3:
                reduce_axis = (0, 1)
            else:
                reduce_axis = (0, 2, 3)
        result = graph_builder.emit('ReduceSum', [x], attrs={'reduce_axis': reduce_axis, 'keep_dims': False})
        if x.data_format == DF.FRAC_NZ:
            out_shape = x.shape[:-4] + [x.shape[-1] * x.shape[-4]]
            result = graph_builder.emit('Reshape', [result], attrs={'shape': out_shape})
        return result
