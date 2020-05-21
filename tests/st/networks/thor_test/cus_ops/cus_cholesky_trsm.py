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
