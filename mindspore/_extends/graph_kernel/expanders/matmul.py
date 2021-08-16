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
"""generate json desc for BatchMatMul and MatMul"""
from mindspore._extends.graph_kernel.model.model import DataFormat as DF
from mindspore._extends.graph_kernel.model.model import GraphKernelUnsupportedException as GKException
from ._utils import Expander, ExpanderInfoValidator as VLD


@VLD.check_attrs('transpose_a', 'transpose_b', 'left_format', 'right_format')
class MatMul(Expander):
    """
    MatMul expander
    """

    def __init__(self, expand_info):
        super(MatMul, self).__init__(expand_info)
        self.transpose_a = self.attrs['transpose_a']
        self.transpose_b = self.attrs['transpose_b']
        self.left_format = self.attrs['left_format']
        self.right_format = self.attrs['right_format']
        self.shape_a = self.inputs[0]['shape']
        self.shape_b = self.inputs[1]['shape']

    def _optimize_to_mul(self):
        """check if matmul can be replace by mul"""
        if self.processor != 'aicore' or self.left_format != DF.DEFAULT or self.right_format != DF.DEFAULT:
            return False
        k_a = self.shape_a[-2] if self.transpose_a else self.shape_a[-1]
        k_b = self.shape_b[-1] if self.transpose_b else self.shape_b[-2]
        if k_a != 1 or k_b != 1:
            return False
        return True

    def _check(self):
        input_num = len(self.inputs)
        if input_num < 2:
            raise GKException("matul inputs number should bigger than 1, but got {}.".format(input_num))

    def _expand(self, graph_builder):
        def transpose(shape):
            trans_shape = list(shape)
            trans_shape[-2] = shape[-1]
            trans_shape[-1] = shape[-2]
            return trans_shape
        if not self._optimize_to_mul():
            raise GKException("MatMul/BatchMatMul do not need to be replaced by Mul")
        # Matmul is replaced by Mul([b m k], [b k n]) when k==1
        input_a = self.inputs[0]
        input_b = self.inputs[1]
        if self.transpose_a:
            shape_a_trans = transpose(self.shape_a)
            input_a = graph_builder.emit('Reshape', [input_a], attrs={'shape': shape_a_trans})
        if self.transpose_b:
            shape_b_trans = transpose(self.shape_b)
            input_b = graph_builder.emit('Reshape', [input_b], attrs={'shape': shape_b_trans})
        result = graph_builder.emit('Mul', [input_a, input_b])
        if 'dst_type' in self.attrs and self.inputs[0].dtype != self.attrs['dst_type']:
            result = graph_builder.emit('Cast', [result], attrs={'dst_type': self.attrs['dst_type']})
        return result


class BatchMatMul(MatMul):
    """BatchMatMul expander"""
