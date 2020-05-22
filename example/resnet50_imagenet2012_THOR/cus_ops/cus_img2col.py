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
"""CusImg2Col"""
from mindspore.ops import prim_attr_register, PrimitiveWithInfer
from mindspore.ops.composite import multitype_ops as C


class CusImg2Col(PrimitiveWithInfer):
    """CusImg2Col definition"""

    @prim_attr_register
    def __init__(self, ksizes, strides, dilates=(1, 1, 1, 1), mode="NC1HWC0"):
        """init CusImg2Col"""
        self.init_prim_io_names(inputs=['x1'], outputs=['y'])
        self.ksizes = ksizes
        self.strides = strides
        self.dilates = dilates
        self.mode = mode

    def get_bprop(self):
        def bprop(x, out, dout):
            return (C.zeros_like(x),)

        return bprop

    def infer_shape(self, data1_shape):
        bs, c, h, w = data1_shape
        _, stride_h, stride_w, _ = self.strides
        _, k_w, k_h, _ = self.ksizes
        # assert m == n
        c0 = 16
        c1 = c // 16
        if c1 == 0:
            c1 = 1
        shape = [bs * int(h // stride_h) * int(w // stride_w), k_w * k_h * c1 * c0]
        return shape

    def infer_dtype(self, data1_dtype):
        return data1_dtype
