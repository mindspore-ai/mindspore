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


@VLD.add_format(DF.DEFAULT, DF.DEFAULT)
@VLD.add_format(DF.NCHW, DF.DEFAULT)
@VLD.add_format(DF.NHWC, DF.DEFAULT)
class BiasAdd(Expander):
    """BiasAdd expander"""

    def _expand(self, graph_builder):
        input_x, input_y = self.inputs

        if input_x.data_format == DF.NCHW:
            input_y_expand = graph_builder.emit('ExpandDims', [input_y], attrs={'axis': 1})
            input_y_expand = graph_builder.emit('ExpandDims', [input_y_expand], attrs={'axis': 2})
            result = graph_builder.emit('Add', [input_x, input_y_expand])
        elif input_x.data_format == DF.DEFAULT:
            if len(input_x.shape) == 2:
                result = graph_builder.emit('Add', [input_x, input_y])
            elif len(input_x.shape) == 3:
                input_y_expand = graph_builder.emit('ExpandDims', [input_y], attrs={'axis': 1})
                result = graph_builder.emit('Add', [input_x, input_y_expand])
            else:  # len == 4
                input_y_expand = graph_builder.emit('ExpandDims', [input_y], attrs={'axis': 1})
                input_y_expand = graph_builder.emit('ExpandDims', [input_y_expand], attrs={'axis': 2})
                result = graph_builder.emit('Add', [input_x, input_y_expand])
        else:  # NHWC
            result = graph_builder.emit('Add', [input_x, input_y])

        return result
