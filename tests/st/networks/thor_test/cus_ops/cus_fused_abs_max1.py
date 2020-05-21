import numpy as np
from mindspore.ops import prim_attr_register, PrimitiveWithInfer
from mindspore import Tensor
from mindspore.ops.composite import multitype_ops as C
 
class CusFusedAbsMax1(PrimitiveWithInfer):
    """CusCholeskyTrsm definition"""
    @prim_attr_register
    def __init__(self, origin_shape = [-1,-1]):
        """init CusCholeskyTrsm"""
        self.init_prim_io_names(inputs=['x1'], outputs=['y'])
        from .fused_abs_max1 import CusFusedAbsMax1
        self.origin_shape = origin_shape
 
    def get_bprop(self):
        def bprop(x, out, dout):
            return (C.zeros_like(x),)
        return bprop
 
    def infer_shape(self, data1_shape):
        if len(data1_shape) == 2:
            return [1,]
        else:
            return [32, 64]
        # return [128,128]
 
    def infer_dtype(self, data1_dtype):
        return data1_dtype
