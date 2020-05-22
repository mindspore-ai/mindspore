# Copyright 2020 Huawei Technologies Co., Ltd
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
# ============================================================================
import mindspore as ms
from mindspore.ops import prim_attr_register, PrimitiveWithInfer
from mindspore.ops.composite import multitype_ops as C


# y = x^2
class CusMatMulCube(PrimitiveWithInfer):
    """CusMatMulCube definition"""

    @prim_attr_register
    def __init__(self, transpose_a=False, transpose_b=False):
        """init CusMatMulCube"""
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['y'])
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b

    def get_bprop(self):
        def bprop(x1, x2, out, dout):
            return (C.zeros_like(x1), C.zeros_like(x2))

        return bprop

    def infer_shape(self, data1_shape, data2_shape):
        # shape = [1, data1_shape[1], data2_shape[2], 16, 16]
        # return shape
        if self.transpose_a == True:
            k1, m = data1_shape
        else:
            m, k1 = data1_shape
        if self.transpose_b == True:
            n, k2 = data2_shape
        else:
            k2, n = data2_shape
        assert k1 == k2
        shape = [m, n]
        return shape

    def infer_dtype(self, data1_dtype, data2_dtype):
        return ms.common.dtype.tensor_type(getattr(ms, "float32"))
