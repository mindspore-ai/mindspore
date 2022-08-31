# Copyright 2022 Huawei Technologies Co., Ltd
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
"""generate json desc for crealdiv"""
from mindspore._extends.graph_kernel.model.model import DataFormat as DF
from mindspore._extends.graph_kernel.expanders._utils import Expander, ExpanderInfoValidator as VLD


@VLD.add_format(DF.DEFAULT, DF.DEFAULT)
class CRealDiv(Expander):
    """CRealDiv expander"""

    def _expand(self, graph_builder):
        """CRealDiv Implementation"""
        input_x, input_y = self.inputs
        if input_x.dtype == input_y.dtype:
            x_real = graph_builder.emit('CReal', [input_x])
            y_real = graph_builder.emit('CReal', [input_y])
            x_imag = graph_builder.emit('CImag', [input_x])
            y_imag = graph_builder.emit('CImag', [input_y])
            squre_y_real = graph_builder.emit('Mul', [y_real, y_real])
            squre_y_imag = graph_builder.emit('Mul', [y_imag, y_imag])
            final_denominator = graph_builder.emit('Add', [squre_y_real, squre_y_imag])
            x_real_mul_y_real = graph_builder.emit('Mul', [x_real, y_real])
            x_imag_mul_y_imag = graph_builder.emit('Mul', [x_imag, y_imag])
            x_real_mul_y_imag = graph_builder.emit('Mul', [x_real, y_imag])
            x_imag_mul_y_real = graph_builder.emit('Mul', [x_imag, y_real])
            final_numerator_real = graph_builder.emit('Add', [x_real_mul_y_real, x_imag_mul_y_imag])
            final_numerator_imag = graph_builder.emit('Sub', [x_imag_mul_y_real, x_real_mul_y_imag])
            result_real = graph_builder.emit('RealDiv', [final_numerator_real, final_denominator])
            result_imag = graph_builder.emit('RealDiv', [final_numerator_imag, final_denominator])
            result = graph_builder.emit('Complex', [result_real, result_imag])
        elif input_x.dtype == "complex64" or input_x.dtype == "complex128":
            x_real = graph_builder.emit('CReal', [input_x])
            x_imag = graph_builder.emit('CImag', [input_x])
            x_real_div_y = graph_builder.emit('RealDiv', [x_real, input_y])
            x_imag_div_y = graph_builder.emit('RealDiv', [x_imag, input_y])
            result = graph_builder.emit('Complex', [x_real_div_y, x_imag_div_y])
        elif input_y.dtype == "complex64" or input_y.dtype == "complex128":
            y_real = graph_builder.emit('CReal', [input_y])
            y_imag = graph_builder.emit('CImag', [input_y])
            neg_y_imag = graph_builder.emit('Neg', [y_imag])
            squre_y_real = graph_builder.emit('Mul', [y_real, y_real])
            squre_y_imag = graph_builder.emit('Mul', [y_imag, y_imag])
            final_denominator = graph_builder.emit('Add', [squre_y_real, squre_y_imag])
            x_mul_y_real = graph_builder.emit('Mul', [input_x, y_real])
            x_mul_neg_y_imag = graph_builder.emit('Mul', [input_x, neg_y_imag])
            y_real_div_x = graph_builder.emit('RealDiv', [x_mul_y_real, final_denominator])
            y_imag_div_x = graph_builder.emit('RealDiv', [x_mul_neg_y_imag, final_denominator])
            result = graph_builder.emit('Complex', [y_real_div_x, y_imag_div_x])
        return result
