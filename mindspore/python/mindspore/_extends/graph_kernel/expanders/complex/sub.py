# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""generate json desc for csub"""
from mindspore._extends.graph_kernel.model.model import DataFormat as DF
from mindspore._extends.graph_kernel.expanders._utils import Expander, ExpanderInfoValidator as VLD


@VLD.add_format(DF.DEFAULT, DF.DEFAULT)
class CSub(Expander):
    """CSub expander"""

    def _expand(self, graph_builder):
        input_x, input_y = self.inputs
        if input_x.dtype == input_y.dtype:
            x_real = graph_builder.emit('CReal', [input_x])
            y_real = graph_builder.emit('CReal', [input_y])
            x_imag = graph_builder.emit('CImag', [input_x])
            y_imag = graph_builder.emit('CImag', [input_y])
            result_real = graph_builder.emit('Sub', [x_real, y_real])
            result_imag = graph_builder.emit('Sub', [x_imag, y_imag])
            result = graph_builder.emit('Complex', [result_real, result_imag])
        elif input_x.dtype == "complex64" or input_x.dtype == "complex128":
            x_real = graph_builder.emit('CReal', [input_x])
            x_imag = graph_builder.emit('CImag', [input_x])
            x_real_sub_y = graph_builder.emit('Sub', [x_real, input_y])
            result = graph_builder.emit('Complex', [x_real_sub_y, x_imag])
        elif input_y.dtype == "complex64" or input_y.dtype == "complex128":
            y_real = graph_builder.emit('CReal', [input_y])
            y_imag = graph_builder.emit('CImag', [input_y])
            x_sub_y_real = graph_builder.emit('Sub', [input_x, y_real])
            y_imag = graph_builder.emit('Neg', [y_imag])
            result = graph_builder.emit('Complex', [x_sub_y_real, y_imag])
        return result
