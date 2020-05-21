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
 
class CusCholeskyTrsm(PrimitiveWithInfer):
    """CusCholeskyTrsm definition"""
    @prim_attr_register
    def __init__(self):
        """init CusCholeskyTrsm"""
        self.init_prim_io_names(inputs=['x1'], outputs=['y'])
        from .cholesky_trsm import CusCholeskyTrsm
 
    def infer_shape(self, data1_shape):
        m,n = data1_shape
        if m >= 128:
            return [m//128,128,128]
        else:
            return [1,64,64]
 
    def infer_dtype(self, data1_dtype):
        return data1_dtype
