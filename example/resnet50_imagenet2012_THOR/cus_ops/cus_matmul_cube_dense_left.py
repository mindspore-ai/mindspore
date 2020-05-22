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
"""CusMatMulCubeDenseLeft"""
import mindspore as ms
from mindspore.ops import prim_attr_register, PrimitiveWithInfer
from mindspore.ops.composite import multitype_ops as C


# y = x^2
class CusMatMulCubeDenseLeft(PrimitiveWithInfer):
    """CusMatMulCube definition"""

    @prim_attr_register
    def __init__(self):
        """init CusMatMulCube"""
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['y'])

    def get_bprop(self):
        def bprop(x1, x2, out, dout):
            return (C.zeros_like(x1), C.zeros_like(x2))

        return bprop

    def infer_shape(self, data1_shape, data2_shape):
        return data2_shape

    def infer_dtype(self, data1_dtype, data2_dtype):
        return ms.common.dtype.tensor_type(getattr(ms, "float16"))
