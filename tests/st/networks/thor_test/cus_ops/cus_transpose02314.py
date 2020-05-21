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
 
 
import numpy as np
from mindspore.ops import prim_attr_register, PrimitiveWithInfer
from mindspore import Tensor
from mindspore.ops.composite import multitype_ops as C
 
class CusTranspose02314(PrimitiveWithInfer):
    """CusTranspose02314 definition"""
    @prim_attr_register
    def __init__(self):
        """init CusTranspose02314"""
        self.init_prim_io_names(inputs=['x1'], outputs=['y'])
        from .transpose02314_impl import CusTranspose02314
 
    def get_bprop(self):
        def bprop(x, out, dout):
            return (C.zeros_like(x),)
        return bprop
 
    def infer_shape(self, data1_shape):
        assert len(data1_shape) == 4
        n, c, h, w = data1_shape
        c0 = 16
        c1 = c // 16
        shape = (n * h * w, c1 * c0)
        # axis_0, axis_1, axis_2, axis_3, axis_4 = data1_shape
        # shape = (axis_0, axis_2, axis_3, axis_1, axis_4)
        return shape
 
    def infer_dtype(self, data1_dtype):
        return data1_dtype
