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
from functools import reduce
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

    def _infer_shape(self):
        """returns the input shape with largest flatten size"""
        shape = (1,)
        max_flatten_size = 1
        for t in self.inputs:
            flatten_size = reduce(lambda x, y: x * y, t.shape)
            if flatten_size >= max_flatten_size:
                max_flatten_size = flatten_size
                shape = t.shape
        return shape

    def _infer_format(self):
        for tensor in self.inputs:
            if tensor.data_format != DF.DEFAULT:
                return tensor.data_format
        return DF.DEFAULT


class _Reduce(OpInfer):
    """Common infer for reduction operators"""

    def _check(self):
        super()._check()
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
        return DF.DEFAULT


class Reshape(_Reshape):
    def _infer_shape(self):
        return self.attrs["shape"]


class ExpandDims(_Reshape):
    def _infer_shape(self):
        return list(self.inputs[0].shape).insert(self.attrs["axis"], 1)


class Cast(_Elemwise):
    def _infer_type(self):
        return self.attrs["dst_type"]


class InplaceAssign(_Elemwise):
    def _infer_shape(self):
        return [1] if self.attrs["fake_output"] else self.inputs[2].shape

    def _infer_type(self):
        return self.inputs[2].dtype

    def _infer_format(self):
        return DF.DEFAULT if self.attrs["fake_output"] else self.inputs[2].data_format


class BroadcastTo(OpInfer):
    def _infer_shape(self):
        return self.attrs["shape"]

    def _infer_format(self):
        return self.inputs[0].data_format


class Tile(OpInfer):
    """Op Tile"""

    def __init__(self, op_name, inputs, attrs):
        super().__init__(op_name, inputs, attrs)
        self.input_reshape = None
        self.output_reshape = None
        self.broadcast_compatible = True

    def _infer_shape(self):
        shape = self.inputs[0].shape
        multiples = self.attrs["multiples"]

        shape = list(shape)
        multiples = list(multiples)
        diff_len = len(multiples) - len(shape)
        if diff_len < 0:
            raise ValueError("Dimensions of multiples{} < dimensions of input{} in Tile".format(multiples, shape))
        if diff_len > 0:
            for _ in range(diff_len):
                shape.insert(0, 1)

        self.broadcast_compatible = True
        output_shape = []
        self.input_reshape = []
        self.output_reshape = []
        for sh, mul in list(zip(shape, multiples)):
            dim = sh * mul
            output_shape.append(dim)
            if sh == 1 or mul == 1:
                self.input_reshape.append(sh)
                self.output_reshape.append(dim)
            else:
                self.broadcast_compatible = False
                self.input_reshape.append(1)
                self.input_reshape.append(sh)
                self.output_reshape.append(mul)
                self.output_reshape.append(sh)

        return output_shape

    def _infer_format(self):
        return DF.DEFAULT


class _CompareOp(_Elemwise):
    """Compare operators"""

    def _infer_type(self):
        return "bool"


class Less(_CompareOp):
    pass


class LessEqual(_CompareOp):
    pass


class Equal(_CompareOp):
    pass


class Greater(_CompareOp):
    pass


class GreaterEqual(_CompareOp):
    pass


class Select(_Elemwise):
    def _check_type(self):
        if self.inputs[0].dtype != "bool":
            raise GKException("Select's input[0] should be a bool condition but got {}".format(self.inputs[0].dtype))
        if self.inputs[1].dtype != self.inputs[2].dtype:
            raise GKException("Select's input mismatch ({} vs {})".format(self.inputs[1].dtype, self.inputs[2].dtype))

    def _infer_type(self):
        return self.inputs[1].dtype
