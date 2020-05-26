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
"""thor_ops"""
import mindspore as ms
from mindspore.ops import prim_attr_register, PrimitiveWithInfer
from mindspore.ops.composite import multitype_ops as C

__all__ = ["CusBatchMatMul",
           "CusCholeskyTrsm",
           "CusFusedAbsMax1",
           "CusImg2Col",
           "CusMatMulCubeDenseLeft",
           "CusMatMulCubeFraczRightMul",
           "CusMatMulCube",
           "CusMatrixCombine",
           "CusTranspose02314",
           "CusMatMulCubeDenseRight",
           "CusMatMulCubeFraczLeftCast",
           ]


class CusBatchMatMul(PrimitiveWithInfer):
    """CusBatchMatMul definition"""

    @prim_attr_register
    def __init__(self):
        """init CusBatchMatMul"""
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['y'])

    def get_bprop(self):
        def bprop(x1, x2, out, dout):
            return (C.zeros_like(x1), C.zeros_like(x2))

        return bprop

    def infer_shape(self, data1_shape, data2_shape):
        return data1_shape

    def infer_dtype(self, data1_dtype, data2_dtype):
        return data1_dtype


class CusCholeskyTrsm(PrimitiveWithInfer):
    """CusCholeskyTrsm definition"""

    @prim_attr_register
    def __init__(self):
        """init CusCholeskyTrsm"""
        self.init_prim_io_names(inputs=['x1'], outputs=['y'])

    def infer_shape(self, data1_shape):
        ll = []
        m, _ = data1_shape
        if m >= 128:
            ll = [m // 128, 128, 128]
        else:
            ll = [1, 64, 64]
        return ll

    def infer_dtype(self, data1_dtype):
        return data1_dtype


class CusFusedAbsMax1(PrimitiveWithInfer):
    """CusFusedAbsMax1 definition"""

    @prim_attr_register
    def __init__(self, origin_shape=[-1, -1]):
        """init CusFusedAbsMax1"""
        self.init_prim_io_names(inputs=['x1'], outputs=['y'])
        self.origin_shape = origin_shape

    def get_bprop(self):
        def bprop(x, out, dout):
            return (C.zeros_like(x),)

        return bprop

    def infer_shape(self, data1_shape):
        ll = []
        if len(data1_shape) == 2:
            ll = [1,]
        else:
            ll = [32, 64]
        return ll

    def infer_dtype(self, data1_dtype):
        return data1_dtype


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


class CusMatMulCubeDenseLeft(PrimitiveWithInfer):
    """CusMatMulCube definition"""

    @prim_attr_register
    def __init__(self):
        """init CusMatMulCubeDenseLeft"""
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['y'])

    def get_bprop(self):
        def bprop(x1, x2, out, dout):
            return (C.zeros_like(x1), C.zeros_like(x2))

        return bprop

    def infer_shape(self, data1_shape, data2_shape):
        return data2_shape

    def infer_dtype(self, data1_dtype, data2_dtype):
        return ms.common.dtype.tensor_type(getattr(ms, "float16"))


class CusMatMulCubeFraczRightMul(PrimitiveWithInfer):
    """CusMatMulCubeFraczRightMul definition"""

    @prim_attr_register
    def __init__(self):
        """init CusMatMulCubeFraczRightMul"""
        self.init_prim_io_names(inputs=['x1', 'x2', 'x3'], outputs=['y'])

    def get_bprop(self):
        def bprop(x1, x2, x3, out, dout):
            return (C.zeros_like(x1), C.zeros_like(x2), C.zeros_like(x3))

        return bprop

    def infer_shape(self, data1_shape, data2_shape, data3_shape):
        return data1_shape

    def infer_dtype(self, data1_dtype, data2_dtype, data3_dtype):
        return ms.common.dtype.tensor_type(getattr(ms, "float32"))


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
        if self.transpose_a:
            k1, m = data1_shape
        else:
            m, k1 = data1_shape
        if self.transpose_b:
            n, k2 = data2_shape
        else:
            k2, n = data2_shape
        assert k1 == k2
        shape = [m, n]
        return shape

    def infer_dtype(self, data1_dtype, data2_dtype):
        return ms.common.dtype.tensor_type(getattr(ms, "float32"))


class CusMatrixCombine(PrimitiveWithInfer):
    """CusMatrixCombine definition"""

    @prim_attr_register
    def __init__(self):
        """init CusMatrixCombine"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def get_bprop(self):
        def bprop(x, out, dout):
            return (C.zeros_like(x),)

        return bprop

    def infer_shape(self, data_shape):
        a, b, c = data_shape
        shape = [a * b, a * c]

        return shape

    def infer_dtype(self, data_dtype):
        return data_dtype


class CusTranspose02314(PrimitiveWithInfer):
    """CusTranspose02314 definition"""

    @prim_attr_register
    def __init__(self):
        """init CusTranspose02314"""
        self.init_prim_io_names(inputs=['x1'], outputs=['y'])

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
        return shape

    def infer_dtype(self, data1_dtype):
        return data1_dtype


class CusMatMulCubeDenseRight(PrimitiveWithInfer):
    """CusMatMulCubeDenseRight definition"""

    @prim_attr_register
    def __init__(self):
        """init CusMatMulCubeDenseRight"""
        self.init_prim_io_names(inputs=['x1', 'x2', 'x3'], outputs=['y'])

    def get_bprop(self):
        def bprop(x1, x2, x3, out, dout):
            return (C.zeros_like(x1), C.zeros_like(x2), C.zeros_like(x3))

        return bprop

    def infer_shape(self, data1_shape, data2_shape, data3_shape):
        return data1_shape

    def infer_dtype(self, data1_dtype, data2_dtype, data3_dtype):
        return ms.common.dtype.tensor_type(getattr(ms, "float32"))


class CusMatMulCubeFraczLeftCast(PrimitiveWithInfer):
    """CusMatMulCubeFraczLeftCast definition"""

    @prim_attr_register
    def __init__(self):
        """init CusMatMulCubeFraczLeftCast"""
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['y'])

    def get_bprop(self):
        def bprop(x1, x2, out, dout):
            return (C.zeros_like(x1), C.zeros_like(x2))

        return bprop

    def infer_shape(self, data1_shape, data2_shape):
        return data2_shape

    def infer_dtype(self, data1_dtype, data2_dtype):
        return ms.common.dtype.tensor_type(getattr(ms, "float16"))
