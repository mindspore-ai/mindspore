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
"""GraphKernel Op Infer"""

import copy
import sys
from functools import reduce as prod_reduce
from .model import GraphKernelUnsupportedException as GKException
from .model import PrimLib, DataFormat as DF


def infer(op_name, inputs, attrs):
    """infer shape dtype and format"""

    def _create_opinfer():
        if hasattr(sys.modules[__name__], op_name):
            op_cls = getattr(sys.modules[__name__], op_name)
            return op_cls(op_name, inputs, attrs)
        # common infer
        class_name_map = {
            PrimLib.ELEMWISE: "_Elemwise",
            PrimLib.REDUCE: "_Reduce",
        }
        cls_name = class_name_map.get(PrimLib.primtives.get(op_name, PrimLib.default_primtive).iter_type, None)
        if not cls_name:
            raise GKException("OpInfo does not support op {}".format(op_name))
        op_cls = getattr(sys.modules[__name__], cls_name)
        return op_cls(op_name, inputs, attrs)

    return _create_opinfer().infer()


class OpInfer:
    """
    OpInfer is the base class for inferring operator info in GraphKernel model builder.

    There are three methods should be overridden to define the infer logic of the operator:
    _infer_shape(), _infer_type() and _infer_format().
    """

    def __init__(self, name, inputs, attrs):
        self.name = name
        self.inputs = inputs
        self.attrs = attrs

    def infer(self):
        """Infer shape, type and format by op inputs"""
        self._check()
        return self._infer_shape(), self._infer_type(), self._infer_format()

    def _infer_shape(self):
        return self.inputs[0].shape

    def _infer_type(self):
        return self.inputs[0].dtype

    def _infer_format(self):
        return self.inputs[0].data_format

    def _check(self):
        self._check_shape()
        self._check_type()
        self._check_format()

    def _check_shape(self):
        pass

    def _check_type(self):
        """check all dtypes are same"""
        dtype = self.inputs[0].dtype
        for i, t in enumerate(self.inputs[1:]):
            if t.dtype != dtype:
                raise GKException(
                    "Incompatible dtype between input {}({}) and {}({})".format(0, dtype, i + 1, t.dtype))

    def _check_format(self):
        """check formats are compatible. only DefaultFormat is compatible with others"""
        result = self.inputs[0].data_format
        i = 0
        for j, t in enumerate(self.inputs[1:]):
            if t.data_format != result:
                if DF.DEFAULT not in (result, t.data_format):
                    raise GKException("Incompatible format between input {}({}) and {}({})".format(
                        i, result, j + 1, t.data_format))
                if result == DF.DEFAULT:
                    result = t.data_format
                    i = j + 1


class _Elemwise(OpInfer):
    """Common infer for elementwise operators"""

    @staticmethod
    def broadcast_shape(shapes):
        """deduce broadcast shape using same rules as numpy"""
        dim_size = max([len(shape) for shape in shapes])
        align_shapes = [[1] * (dim_size - len(shape)) + shape for shape in shapes]
        out_shape = [1] * dim_size
        for i in range(dim_size):
            for align_shape in align_shapes:
                if align_shape[i] == 1:
                    continue
                if out_shape[i] == 1:
                    out_shape[i] = align_shape[i]
                elif out_shape[i] != align_shape[i]:
                    raise GKException("shape broadcast failed!")
        return out_shape

    @staticmethod
    def defaultformat_to_nz(default_shape):
        """default format shape to fractal_Nz format shape"""
        more_two_d_shape, two_d_shape = default_shape[:-2], default_shape[-2:]
        # (32) or (1, 32) -> (2, 1, 1, 16)
        if len(two_d_shape) == 1 or (len(two_d_shape) == 2 and two_d_shape[0] == 1):
            shape = [two_d_shape[-1] // 16, 1, 1, 16]
            if two_d_shape[-1] % 16 != 0:
                raise GKException("should be multiplies of 16")
            return more_two_d_shape + shape
        # (32, 1) -> (1, 2, 16, 1)
        if len(two_d_shape) == 2 and two_d_shape[1] == 1:
            shape = [1, two_d_shape[0] // 16, 16, 1]
            if two_d_shape[0] % 16 != 0:
                raise GKException("should be multiples of 16")
            return more_two_d_shape + shape
        # (32, 48) -> (3, 2, 16, 16)
        shape = [two_d_shape[1] // 16, two_d_shape[0] // 16, 16, 16]
        if two_d_shape[0] % 16 != 0 or two_d_shape[1] % 16 != 0:
            raise GKException("should be multiples of 16")
        return more_two_d_shape + shape

    def _infer_shape(self):
        """returns the output shape with broadcast"""

        # in case all inputs are default format/NHWC/NCHW
        is_default = [op_input.data_format in (DF.DEFAULT, DF.NHWC, DF.NCHW) for op_input in self.inputs]
        if all(is_default):
            return self.broadcast_shape([op_input.shape for op_input in self.inputs])

        # in case formats are fractal_nz, default_fromat/NHWC/HCHW(optional)
        is_default_frac_nz = [op_input.data_format in (DF.DEFAULT, DF.NHWC, DF.NCHW, DF.FRAC_NZ)
                              for op_input in self.inputs]
        if all(is_default_frac_nz):
            nz_shapes = [self.defaultformat_to_nz(op_input.shape) if op_input.data_format != DF.FRAC_NZ
                         else op_input.shape for op_input in self.inputs]
            return self.broadcast_shape(nz_shapes)

        raise GKException("Only support default and fractal_nz")

    def _infer_format(self):
        for tensor in self.inputs:
            if tensor.data_format != DF.DEFAULT:
                return tensor.data_format
        return DF.DEFAULT


class _Reduce(OpInfer):
    """Common infer for reduction operators"""

    def _check(self):
        super(_Reduce, self)._check()
        # check reduce axis in the range [-len, len)
        shape_len = len(self.inputs[0].shape)
        axis = self.attrs['reduce_axis']
        if isinstance(axis, int):
            axis = [axis]
        if not all([(-shape_len <= i < shape_len) for i in axis]):
            raise GKException(
                "reduce_axis should be in range [{},{}) but got {}".format(-shape_len, shape_len, axis))

    def _infer_shape(self):
        shape = copy.deepcopy(self.inputs[0].shape)
        axis = self.attrs['reduce_axis']

        if isinstance(axis, int):
            axis = [axis]
        if any([i < 0 for i in axis]):
            # change the axis to non-negative number.
            axis = list(map(lambda i: i + len(shape) if i < 0 else i, axis))
        self.attrs['reduce_axis'] = sorted(axis)

        if self.attrs['keep_dims']:
            for i in axis:
                shape[i] = 1
            return shape

        real_shape = []
        for i, s in enumerate(shape):
            if i not in axis:
                real_shape.append(s)
        return real_shape

    def _infer_format(self):
        return DF.DEFAULT


class _Reshape(OpInfer):
    """Common infer for reshape operators, should not be instantiated"""

    def _infer_shape(self):
        raise GKException("_infer_shape should be implemented by subclass")

    def _infer_format(self):
        return DF.DEFAULT if "format" not in self.attrs else self.attrs["format"]


class Reshape(_Reshape):
    """Reshape op infer"""

    def _check_shape(self):
        size_before_reshape = prod_reduce(lambda x, y: x * y, self.inputs[0].shape)
        size_after_reshape = prod_reduce(lambda x, y: x * y, self.attrs["shape"])
        if size_before_reshape != size_after_reshape:
            raise GKException("The shape product before and after reshaping should be equal")

    def _infer_shape(self):
        return self.attrs["shape"]


class Cast(_Elemwise):
    """Cast op infer"""

    def _infer_type(self):
        return self.attrs["dst_type"]


class InplaceAssign(_Elemwise):
    """InplaceAssign op infer"""

    def _infer_shape(self):
        return self.inputs[2].shape

    def _infer_type(self):
        return self.inputs[2].dtype

    def _infer_format(self):
        return self.inputs[2].data_format


class BroadcastTo(OpInfer):
    """BroadcastTo op infer"""

    def _infer_shape(self):
        return self.attrs["shape"]

    def _infer_format(self):
        return self.inputs[0].data_format


class _CompareOp(_Elemwise):
    """Compare operators"""

    def _infer_type(self):
        return "bool"


class CImag(OpInfer):
    """CImag op infer"""

    def _check_type(self):
        if self.inputs[0].dtype != "complex64":
            raise GKException(
                "CImag's input[0] should be a complex64 condition but got {}".format(self.inputs[0].dtype))

    def _infer_type(self):
        return "float32"


class CReal(OpInfer):
    """CReal op infer"""

    def _check_type(self):
        if self.inputs[0].dtype != "complex64":
            raise GKException(
                "CReal's input[0] should be a complex64 condition but got {}".format(self.inputs[0].dtype))

    def _infer_type(self):
        return "float32"


class Complex(OpInfer):
    """Complex op infer"""

    def _check_type(self):
        if self.inputs[0].dtype != "float32":
            raise GKException(
                "Complex's input[0] should be a float32 condition but got {}".format(self.inputs[0].dtype))
        if self.inputs[0].dtype != self.inputs[1].dtype:
            raise GKException("Complex's input mismatch ({} vs {})".format(self.inputs[0].dtype, self.inputs[1].dtype))

    def _infer_type(self):
        return "complex64"


class Less(_CompareOp):
    """Less op infer"""


class LessEqual(_CompareOp):
    """LessEqual op infer"""


class Equal(_CompareOp):
    """Equal op infer"""


class Greater(_CompareOp):
    """Greater op infer"""


class GreaterEqual(_CompareOp):
    """GreaterEqual op infer"""


class Select(_Elemwise):
    """Select op infer"""

    def _check_type(self):
        if self.inputs[0].dtype != "bool":
            raise GKException("Select's input[0] should be a bool condition but got {}".format(self.inputs[0].dtype))
        if self.inputs[1].dtype != self.inputs[2].dtype:
            raise GKException("Select's input mismatch ({} vs {})".format(self.inputs[1].dtype, self.inputs[2].dtype))

    def _infer_type(self):
        return self.inputs[1].dtype


def check_format_any(formats, checked_format):
    """Check whether input format in formats list"""
    if not isinstance(formats, (list, tuple)):
        raise GKException("formats {} should be list or tuple, but got {}.".format(formats, type(formats)))
    if checked_format not in formats:
        raise GKException("Check {} failed in {}".format(checked_format, formats))


def check_nd(data, nd):
    """Check whether data are nd format"""
    if not isinstance(data, (list, tuple)) or len(data) != nd:
        raise GKException("input should be {}D list or tuple, but got {}.".format(nd, data))


def conv_had_pad(pad_list, pad_mode):
    """Check whether conv need to add pad"""
    if not isinstance(pad_list, (list, tuple)) or len(pad_list) != 4:
        raise GKException("pad_list should be 4D list or tuple, but got {}".format(pad_list))
    if pad_list[0] != pad_list[1] or pad_list[2] != pad_list[3]:
        return True
    if pad_mode not in ["VALID", "valid"]:
        for _, pad in enumerate(pad_list):
            if pad != 0:
                return True
    return False


class Conv2D(OpInfer):
    """Conv2D infer"""

    def _infer_type(self):
        if isinstance(self.attrs, dict) and "dst_type" in self.attrs:
            return self.attrs["dst_type"]
        return self.inputs[0].dtype

    def _infer_shape(self):
        shape_0 = list(self.inputs[0].shape)
        shape_1 = list(self.inputs[1].shape)
        check_nd(shape_0, 4)
        check_nd(shape_1, 4)

        formats = [self.inputs[0].data_format, self.inputs[1].data_format, self.attrs["format"]]
        check_format_any(formats, DF.NHWC)

        n, h, w, out_channel = shape_0[0], shape_0[1], shape_0[2], shape_1[0]
        pad_list = self.attrs["pad_list"]
        pad_mode = self.attrs["pad_mode"]
        kernel_size = self.attrs["kernel_size"]
        stride = self.attrs["stride"]
        dilation = self.attrs["dilation"]
        check_nd(pad_list, 4)
        check_nd(kernel_size, 2)
        check_nd(stride, 4)
        check_nd(dilation, 4)

        has_pad = conv_had_pad(pad_list, pad_mode)
        if not has_pad:
            pad_list = [0, 0, 0, 0]

        k_h = (kernel_size[0] - 1) * dilation[-2] + 1
        k_w = (kernel_size[1] - 1) * dilation[-1] + 1
        out_h = (h + pad_list[0] + pad_list[1] - k_h) // stride[-2] + 1
        out_w = (w + pad_list[2] + pad_list[3] - k_w) // stride[-1] + 1
        return [n, out_h, out_w, out_channel]


class MatMul(OpInfer):
    """MatMul infer"""

    def _infer_type(self):
        if isinstance(self.attrs, dict) and "dst_type" in self.attrs:
            return self.attrs["dst_type"]
        return self.inputs[0].dtype

    def _infer_shape(self):
        shape_0 = list(self.inputs[0].shape)
        shape_1 = list(self.inputs[1].shape)
        if len(shape_0) != 2 or len(shape_1) != 2:
            raise GKException("MatMul's inputs shape must be 2D, but got {}, {}".format(len(shape_0), len(shape_1)))
        transpose_a = self.attrs["transpose_a"]
        transpose_b = self.attrs["transpose_b"]
        m, k1 = (shape_0[-1], shape_0[-2]) if transpose_a else (shape_0[-2], shape_0[-1])
        k2, n = (shape_1[-1], shape_1[-2]) if transpose_b else (shape_1[-2], shape_1[-1])
        if k1 != k2:
            raise GKException("MatMul's inputs have different k value: {} vs {}".format(k1, k2))
        output_shape = [m, n]
        return output_shape


class PadAkg(OpInfer):
    """PadAkg infer"""

    def _infer_shape(self):
        shape = list(self.inputs[0].shape)
        n = len(shape)
        pad_before = list(self.attrs["head"])
        pad_after = list(self.attrs["tail"])
        if len(pad_before) != n or len(pad_after) != n:
            raise GKException("Input dimension and pad mismatch: {}d vs {}d vs {}d"
                              .format(n, len(pad_before), len(pad_after)))
        out_shape = [shape[i] + pad_before[i] + pad_after[i] for i in range(n)]
        return out_shape


class UnPadAkg(OpInfer):
    """UnPadAkg infer"""

    def _infer_shape(self):
        shape = list(self.inputs[0].shape)
        n = len(shape)
        unpad_after = list(self.attrs["tail"])
        if len(unpad_after) != n:
            raise GKException("Input dimension and pad mismatch: {}d vs {}d".format(n, len(unpad_after)))
        out_shape = [shape[i] - unpad_after[i] for i in range(n)]
        return out_shape


class Gather(OpInfer):
    """Gather infer"""

    def _infer_shape(self):
        input_shape = self.inputs[0].shape
        indices_shape = self.inputs[1].shape
        axis = self.attrs['axis']
        output_shape = input_shape
        indices_shape_one_dim = 1
        for dim in indices_shape:
            indices_shape_one_dim *= dim
        output_shape[axis] = indices_shape_one_dim
        return output_shape

    def _infer_type(self):
        return self.inputs[0].dtype

    def _infer_format(self):
        return self.inputs[0].data_format

    def _check_type(self):
        if self.inputs[1].dtype != "int32":
            raise GKException("Indices dtype must be int32!")
