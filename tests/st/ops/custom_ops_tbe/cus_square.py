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
from mindspore import Tensor
from mindspore.ops import prim_attr_register, PrimitiveWithInfer
from mindspore.ops import operations as P

# y = x^2
class CusSquare(PrimitiveWithInfer):
    """CusSquare definition"""
    from square_impl import CusSquareImpl

    @prim_attr_register
    def __init__(self):
        """init CusSquare"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def vm_impl(self, x):
        x = x.asnumpy()
        return Tensor(np.multiply(x, x))

    def infer_shape(self, data_shape):
        return data_shape

    def infer_dtype(self, data_dtype):
        return data_dtype
    
    def get_bprop(self):
        def bprop(data, out, dout):
            gradient = data * 2
            dx = gradient * dout
            return (dx, )
        return bprop
