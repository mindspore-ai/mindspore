# Copyright 2023 Huawei Technologies Co., Ltd
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

"""Custom op dsl file, used for dynamic format/data type select, update akg info and compile akg info"""
from __future__ import absolute_import
import os
import sys
import json
import copy
import functools
import subprocess
import shutil

from tbe.common.buildcfg import get_current_build_config
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json

BLOCK = 16
FP16_MAX = 65504
OP = "op"
STR = "str"
NAME = "name"
TENSOR_NAME = "tensor_name"
ATTR = "attr"
VALUE = "value"
SHAPE = "shape"
FORMAT = "format"
DATA_TYPE = "data_type"
ORI_SHAPE = "ori_shape"
ORI_FORMAT = "ori_format"
ORI_DATA_TYPE = "ori_data_type"
OP_DESC = "op_desc"
INPUT_DESC = "input_desc"
OUTPUT_DESC = "output_desc"
FRACTAL_NZ = "FRACTAL_NZ"
DEFAULT_FORMAT = "DefaultFormat"
FLOAT16 = "float16"
FLOAT32 = "float32"
O_SUFFIX = ".o"
JSON_SUFFIX = ".json"


def copy_shape(shape):
    """Deep copy shape"""
    res = []
    if isinstance(shape, int):
        shape = [shape]
    for _, s in enumerate(shape):
        res.append(s)
    return res

# InfoGlobalConfig is used to store global configuration for info files.
# It can be accessed or modified internally in custom.py using InfoGlobalConfig.xxx.


class InfoGlobalConfig:
    # whether enable akg cce lib
    enable_cce_lib = False
    # ascend arch type, for 910B and 910A
    ascend_arch = ""


class OpInfer:
    """Base infer class, used to provide supported formats and data type of each op and update each of"""

    def __init__(self, op_desc):
        self.name = op_desc[NAME]
        self.op_desc = op_desc
        self.input_desc = []
        self.output_desc = []
        self.attr = {}
        if isinstance(op_desc.get(INPUT_DESC), list):
            for desc in op_desc[INPUT_DESC]:
                for item in desc:
                    self.input_desc.append(item)
        if isinstance(op_desc.get(ATTR), list):
            for item in op_desc[ATTR]:
                self.attr[item[NAME]] = item
        if isinstance(op_desc.get(OUTPUT_DESC), list):
            for item in op_desc[OUTPUT_DESC]:
                self.output_desc.append(item)

    @staticmethod
    def is_nz(shape):
        """check if shape can be converted to FRACTAL_NZ"""
        if len(shape) >= 2 and shape[-2] % BLOCK == 0 and shape[-1] % BLOCK == 0:
            return True
        return False

    @staticmethod
    def update_format(formats, new_format):
        """combine new_format to formats"""
        new_formats = [new_format] if not isinstance(new_format, (list, tuple)) else new_format
        for f in new_formats:
            if f not in formats:
                formats.append(f)

    def get_attr(self, key):
        """get the value of attr"""
        if key not in self.attr:
            raise KeyError("Can not find attr '{}' in op '{}'".format(key, self.name))
        return self.attr.get(key)[VALUE]

    def set_attr(self, key, value):
        """set the value of attr"""
        if key not in self.attr:
            raise KeyError("Can not find attr '{}' in op '{}'".format(key, self.name))
        self.attr.get(key)[VALUE] = value

    def supported_type(self):
        """get the supported data type of current op"""
        keep_fp32 = False
        for item in self.input_desc:
            # check if type can reduce precision
            value = item.get(VALUE, None)
            if item[DATA_TYPE] == FLOAT32 and value is not None and abs(value) > FP16_MAX:
                keep_fp32 = True
                break
        io_type = ",".join([t[DATA_TYPE] for t in self.input_desc] + [t[DATA_TYPE] for t in self.output_desc])
        fp32_type = io_type.replace(FLOAT16, FLOAT32)
        fp16_type = io_type.replace(FLOAT32, FLOAT16)
        supported_types = [io_type]
        if fp32_type not in supported_types:
            supported_types.append(fp32_type)
        if not keep_fp32 and fp16_type not in supported_types:
            supported_types.append(fp16_type)
        return supported_types

    def supported_format(self):
        """get the supported format of current op"""
        io_num = len(self.input_desc) + len(self.output_desc)
        nd = ["ND"] * io_num
        return [",".join(nd)]

    def infer_type(self):
        """infer data type"""
        fixed_out_type_ops = ["Equal", "Less", "LessEqual", "Greater", "GreaterEqual"]
        if self.name not in fixed_out_type_ops:
            self.output_desc[0][DATA_TYPE] = self.input_desc[0][DATA_TYPE]

    def infer_format(self):
        """infer format"""
        self.output_desc[0][FORMAT] = self.input_desc[0][FORMAT]

    def infer_shape(self):
        """infer shape"""
        self.output_desc[0][SHAPE] = copy_shape(self.input_desc[0][SHAPE])

    def infer_ori_shape(self):
        """infer original shape"""
        for _, desc in enumerate(self.output_desc):
            desc[ORI_SHAPE] = copy_shape(desc[SHAPE])

    def infer(self):
        """infer shape, format and data type"""
        self.infer_type()
        self.infer_format()
        self.infer_shape()

    def post_process(self):
        """post process after infer"""

    def update(self):
        """update each of"""
        for _, desc in enumerate(self.output_desc):
            desc[ORI_DATA_TYPE] = desc[DATA_TYPE]
            desc[ORI_FORMAT] = desc[FORMAT]
        self.infer_ori_shape()
        self.infer()
        self.post_process()


class Elemwise(OpInfer):
    """Elemwise op with one input and one output."""

    def supported_format(self):
        if self.name == "Reciprocal":
            supported_formats = ["ND,ND"]
            # pad will cause 'divided by 0'
            if self.is_nz(self.input_desc[0][SHAPE]):
                self.update_format(supported_formats, "FRACTAL_NZ,FRACTAL_NZ")
            return supported_formats
        return ["ND,ND", "FRACTAL_NZ,FRACTAL_NZ", "NC1HWC0,NC1HWC0", "FRACTAL_Z,FRACTAL_Z"]

    def infer_ori_shape(self):
        self.output_desc[0][ORI_SHAPE] = self.input_desc[0][ORI_SHAPE]


class Cast(Elemwise):
    """Cast op."""

    def supported_type(self):
        in_type = self.input_desc[0][DATA_TYPE]
        out_type = self.output_desc[0][DATA_TYPE]
        io_type = ",".join([in_type, out_type])
        return [io_type]

    def infer_type(self):
        self.output_desc[0][DATA_TYPE] = self.output_desc[0][DATA_TYPE]


class ElemwiseBinaryNoBroadcast(OpInfer):
    """Elemwise op with two inputs and one output, not supports broadcast."""

    def supported_format(self):
        return ["ND,ND,ND", "FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ", "NC1HWC0,NC1HWC0,NC1HWC0",
                "FRACTAL_Z,FRACTAL_Z,FRACTAL_Z"]

    def infer_ori_shape(self):
        self.output_desc[0][ORI_SHAPE] = self.input_desc[0][ORI_SHAPE]


class ElemwiseBinary(OpInfer):
    """Elemwise op with two inputs and one output, supports broadcast."""

    @staticmethod
    def nd2fractal_nz(shape):
        """convert ND shape to FRACTAL_NZ shape"""
        if len(shape) == 1:
            if shape[-1] == 1:
                return [1, 1, 1, 1]
            if shape[-1] % BLOCK == 0:
                return [shape[-1] // BLOCK, 1, 1, BLOCK]
        elif len(shape) >= 2:
            if shape[-2] == 1 and shape[-1] == 1:
                return shape[:-2] + [1, 1, 1, 1]
            if shape[-2] == 1 and shape[-1] % BLOCK == 0:
                return shape[:-2] + [shape[-1] // BLOCK, 1, 1, BLOCK]
            if shape[-2] % BLOCK == 0 and shape[-1] == 1:
                return shape[:-2] + [1, shape[-2] // BLOCK, BLOCK, 1]
        return []

    def broadcast_shape(self, sh0, sh1):
        """calculate broadcast shape"""
        out_shape = []
        max_len = max(len(sh0), len(sh1))
        pad_sh0 = [1] * (max_len - len(sh0)) + sh0
        pad_sh1 = [1] * (max_len - len(sh1)) + sh1
        for i in range(max_len):
            a, b = pad_sh0[i], pad_sh1[i]
            if a == 1:
                out_shape.append(b)
            elif b in [1, a]:
                out_shape.append(a)
            else:
                raise ValueError("For '{}', input shapes {} and {} can not broadcast".format(self.name, sh0, sh1))
        return pad_sh0, pad_sh1, out_shape

    def supported_format(self):
        sh0, sh1 = self.input_desc[0][SHAPE], self.input_desc[1][SHAPE]
        supported_formats = ["ND,ND,ND"]
        is_const_0 = (VALUE in self.input_desc[0])
        is_const_1 = (VALUE in self.input_desc[1])
        if sh0 == sh1 or is_const_0 or is_const_1:
            # No broadcast case
            self.update_format(supported_formats, ["FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ", "NC1HWC0,NC1HWC0,NC1HWC0",
                                                   "FRACTAL_Z,FRACTAL_Z,FRACTAL_Z"])
        else:
            # note: (1, 640), (640)  "FRACTAL_NZ,ND,FRACTAL_NZ", (1, 640) comes from MatMul
            if len(sh0) == 2 and len(sh1) == 1:
                if sh0[-1] == sh1[-1] and sh1[-1] % BLOCK == 0:
                    self.update_format(supported_formats, "FRACTAL_NZ,ND,FRACTAL_NZ")
            elif len(sh0) == 1 and len(sh1) == 2:
                if sh0[-1] == sh1[-1] and sh0[-1] % BLOCK == 0:
                    self.update_format(supported_formats, "ND,FRACTAL_NZ,FRACTAL_NZ")
            # Broadcast case
            pad_sh0, pad_sh1, _ = self.broadcast_shape(sh0, sh1)
            # 1D with broadcast only supports "ND,ND,ND"
            if len(pad_sh0) > 1:
                nz0, nz1 = self.is_nz(pad_sh0), self.is_nz(pad_sh1)
                if nz0 and nz1:
                    self.update_format(supported_formats, "FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ")
                elif nz0:
                    self.update_format(supported_formats, "FRACTAL_NZ,ND,FRACTAL_NZ")
                elif nz1:
                    self.update_format(supported_formats, "ND,FRACTAL_NZ,FRACTAL_NZ")
        # note: ND,ND,FRACTAL_NZ? e.g. (1024, 1), (1, 5120)
        return supported_formats

    def infer_format(self):
        # select special format
        special_formats = ["FRACTAL", "C0"]
        format0, format1 = self.input_desc[0][FORMAT], self.input_desc[1][FORMAT]
        for f in special_formats:
            if format0.find(f) != -1:
                self.output_desc[0][FORMAT] = format0
                return
        self.output_desc[0][FORMAT] = format1

    def infer_shape(self):
        sh0, sh1 = self.input_desc[0][SHAPE], self.input_desc[1][SHAPE]
        if sh0 == sh1:
            self.output_desc[0][SHAPE] = copy_shape(sh0)
        format0, format1 = self.input_desc[0][FORMAT], self.input_desc[1][FORMAT]
        if format0 != format1:
            new_sh0 = self.nd2fractal_nz(sh0)
            new_sh1 = self.nd2fractal_nz(sh1)
            if format0 == FRACTAL_NZ and new_sh1:
                _, _, out_shape = self.broadcast_shape(sh0, new_sh1)
                self.output_desc[0][SHAPE] = out_shape
                return
            if format1 == FRACTAL_NZ and new_sh0:
                _, _, out_shape = self.broadcast_shape(new_sh0, sh1)
                self.output_desc[0][SHAPE] = out_shape
                return
        _, _, out_shape = self.broadcast_shape(sh0, sh1)
        self.output_desc[0][SHAPE] = out_shape

    def infer_ori_shape(self):
        sh0, sh1 = self.input_desc[0][ORI_SHAPE], self.input_desc[1][ORI_SHAPE]
        _, _, out_shape = self.broadcast_shape(sh0, sh1)
        self.output_desc[0][ORI_SHAPE] = out_shape


class MatMul(OpInfer):
    """MatMul op."""

    def supported_format(self):
        input_num = len(self.input_desc)
        # MatMul cce only support ND
        if InfoGlobalConfig.enable_cce_lib and input_num == 2:
            return ["ND,ND,ND"]
        if InfoGlobalConfig.enable_cce_lib and input_num == 3:
            return ["ND,ND,ND,ND"]
        if input_num == 2:
            return ["FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ"]
        if input_num == 3:
            bias_shape = self.input_desc[2][SHAPE]
            if len(bias_shape) == 1 and (bias_shape[-1] == 1 or bias_shape[-1] % BLOCK == 0):
                return ["FRACTAL_NZ,FRACTAL_NZ,ND,FRACTAL_NZ"]
            return ["ND,ND,ND,ND"]
        raise ValueError("MatMul only supports 2 or 3 input tensors, but got {} input tensors".format(input_num))

    def nd_infer(self, sh0, sh1, trans_a, trans_b):
        """infer shape with nd format"""
        if len(sh0) != len(sh1):
            raise ValueError("For '{}', input shape '{}' and '{}' are not supported".format(self.name, sh0, sh1))
        m = sh0[-2] if not trans_a else sh0[-1]
        n = sh1[-1] if not trans_b else sh1[-2]
        res = sh0[:-2] + [m, n]
        return res

    def infer_shape(self):
        sh0, sh1 = self.input_desc[0][SHAPE], self.input_desc[1][SHAPE]
        format0, format1 = self.input_desc[0][FORMAT], self.input_desc[1][FORMAT]
        trans_a, trans_b = self.get_attr("transpose_a"), self.get_attr("transpose_b")
        if format0 != format1 or len(sh0) != len(sh1):
            raise ValueError("For '{}', input '{}' and '{}' are not supported"
                             .format(self.name, self.input_desc[0], self.input_desc[1]))
        if format0 != FRACTAL_NZ and len(sh0) >= 2:
            self.output_desc[0][SHAPE] = self.nd_infer(sh0, sh1, trans_a, trans_b)
        elif format0 == FRACTAL_NZ and len(sh0) >= 4:
            m1, m0 = sh0[-3], sh0[-2]
            if trans_a:
                m1, m0 = sh0[-4], sh0[-1]
            n1, n0 = sh1[-4], sh1[-1]
            if trans_b:
                n1, n0 = sh1[-3], sh1[-2]
            self.output_desc[0][SHAPE] = sh0[:-4] + [n1, m1, m0, n0]
        else:
            raise ValueError("For '{}', input '{}' and '{}' are not supported"
                             .format(self.name, self.input_desc[0], self.input_desc[1]))

    def infer_ori_shape(self):
        sh0, sh1 = self.input_desc[0][ORI_SHAPE], self.input_desc[1][ORI_SHAPE]
        trans_a, trans_b = self.get_attr("transpose_a"), self.get_attr("transpose_b")
        self.output_desc[0][ORI_SHAPE] = self.nd_infer(sh0, sh1, trans_a, trans_b)

    def post_process(self):
        self.op_desc[ATTR].append({DATA_TYPE: STR, NAME: "left_format", VALUE: self.input_desc[0][FORMAT]})
        self.op_desc[ATTR].append({DATA_TYPE: STR, NAME: "right_format", VALUE: self.input_desc[1][FORMAT]})
        self.op_desc[ATTR].append({DATA_TYPE: STR, NAME: "dst_type", VALUE: self.output_desc[0][DATA_TYPE]})

    def infer_type(self):
        """infer data type"""
        if "910B" in InfoGlobalConfig.ascend_arch and not InfoGlobalConfig.enable_cce_lib:
            self.output_desc[0][DATA_TYPE] = "float32"
        else:
            super().infer_type()

    def supported_type(self):
        if "910B" in InfoGlobalConfig.ascend_arch and not InfoGlobalConfig.enable_cce_lib:
            support_types = "float16,float16,float32"
            return [support_types]
        return super().supported_type()

class BatchMatMul(MatMul):
    """BatchMatMul op. Only support cce lib"""
    def __init__(self, op_desc):
        super().__init__(op_desc)
        if "910B" not in InfoGlobalConfig.ascend_arch or not InfoGlobalConfig.enable_cce_lib:
            raise ValueError("BatchMatMul only support 910B cce lib")

    def infer_shape(self):
        sh0, sh1 = self.input_desc[0][SHAPE], self.input_desc[1][SHAPE]
        format0, format1 = self.input_desc[0][FORMAT], self.input_desc[1][FORMAT]
        trans_a, trans_b = self.get_attr("transpose_a"), self.get_attr("transpose_b")
        # only support nd
        if (format0 != FRACTAL_NZ and format1 != FRACTAL_NZ):
            self.output_desc[0][SHAPE] = self.nd_infer(sh0, sh1, trans_a, trans_b)
        else:
            raise ValueError("For '{}', input '{}' and '{}' are not supported"
                             .format(self.name, self.input_desc[0], self.input_desc[1]))

    def nd_infer(self, sh0, sh1, trans_a, trans_b):
        """infer shape with nd format"""
        m = sh0[-2] if not trans_a else sh0[-1]
        n = sh1[-1] if not trans_b else sh1[-2]
        res = sh0[:-2] + [m, n]
        return res

    def infer_type(self):
        """infer data type"""
        self.output_desc[0][DATA_TYPE] = "float16"

    def supported_type(self):
        """supported type"""
        return ["float16,float16,float16"]

class Reduce(OpInfer):
    """Reduce op."""

    @staticmethod
    def _out_nz(rank, axis):
        """check if output remains FRACTAL_NZ"""
        if rank - 2 not in axis and rank - 1 not in axis:
            return True
        return False

    @staticmethod
    def _reduced_shape(shape, axis, keep_dims):
        """calc reduced shape"""
        out_shape = []
        for i, s in enumerate(shape):
            if i in axis:
                if keep_dims:
                    out_shape.append(1)
            else:
                out_shape.append(s)
        return out_shape

    def _get_axis(self, rank):
        axis_input = self.input_desc[1][VALUE]
        axis = []
        if isinstance(axis_input, int):
            axis = [axis_input + rank if axis_input < 0 else axis_input]
        else:
            axis = [i + rank if i < 0 else i for i in axis_input]
        return axis

    def supported_type(self):
        in_type = self.input_desc[0][DATA_TYPE]
        if in_type == FLOAT16:
            return ["float16,int64,float16", "float32,int64,float32"]
        if in_type == FLOAT32:
            return ["float32,int64,float32"]
        io_type = ",".join([in_type, "int64", in_type])
        return [io_type]

    def supported_format(self):
        supported_formats = ["ND,DefaultFormat,ND"]
        shape = self.input_desc[0][SHAPE]
        rank = len(shape)
        axis = self._get_axis(rank)
        if self.is_nz(shape):
            if self._out_nz(rank, axis):
                supported_formats.append("FRACTAL_NZ,DefaultFormat,FRACTAL_NZ")
        return supported_formats

    def infer_shape(self):
        ori_format, cur_format = self.input_desc[0][ORI_FORMAT], self.input_desc[0][FORMAT]
        if cur_format == FRACTAL_NZ and cur_format != ori_format:
            ori_shape, cur_shape = self.input_desc[0][ORI_SHAPE], self.input_desc[0][SHAPE]
            ori_rank = len(ori_shape)
            rank = len(cur_shape)
            axis = self._get_axis(ori_rank)
            new_axis = []
            for i in axis:
                if i == ori_rank - 1:
                    new_axis.extend([rank - 4, rank - 1])
                elif i == ori_rank - 2:
                    new_axis.extend([rank - 3, rank - 2])
                else:
                    new_axis.append(i)
            self.input_desc[1][VALUE] = new_axis
            self.input_desc[1][SHAPE] = [len(new_axis)]
            self.output_desc[0][SHAPE] = self._reduced_shape(cur_shape, new_axis, self.get_attr("keep_dims"))
        else:
            self.output_desc[0][SHAPE] = self.output_desc[0][ORI_SHAPE]

    def infer_ori_shape(self):
        shape = self.input_desc[0][ORI_SHAPE]
        rank = len(shape)
        axis = self._get_axis(rank)
        self.output_desc[0][ORI_SHAPE] = self._reduced_shape(shape, axis, self.get_attr("keep_dims"))


class Reshape(OpInfer):
    """Reshape op."""

    def supported_format(self):
        return ["ND,DefaultFormat,ND"]

    def infer_shape(self):
        """Reshape keeps ND format, so the output shape will not be changed"""
        self.output_desc[0][SHAPE] = self.output_desc[0][ORI_SHAPE]

    def infer_ori_shape(self):
        shape = self.input_desc[0][ORI_SHAPE]
        out_shape = copy_shape(self.input_desc[1][VALUE])
        if -1 in out_shape:
            idx = out_shape.index(-1)
            tmp = []
            for _, s in enumerate(out_shape):
                if s != -1:
                    tmp.append(s)
            if len(tmp) + 1 != len(out_shape):
                raise ValueError("Find multiple -1 in attr 'shape' {}".format(out_shape))
            tmp_sz = functools.reduce(lambda x, y: x * y, tmp, 1)
            out_shape[idx] = functools.reduce(lambda x, y: x * y, shape, 1) // tmp_sz
        self.output_desc[0][ORI_SHAPE] = out_shape

    def post_process(self):
        self.input_desc[1]["ori_value"] = self.input_desc[1][VALUE]
        self.input_desc[1][VALUE] = self.output_desc[0][SHAPE]


class ExpandDimAndSqueeze(Reshape):
    def copy_axis(self, axis):
        out_axis = []
        if isinstance(axis, int):
            out_axis.append(axis)
        else:
            out_axis = copy.deepcopy(axis)
        return out_axis


class Squeeze(ExpandDimAndSqueeze):
    def infer_ori_shape(self):
        axis = self.copy_axis(self.input_desc[1][VALUE])
        input_shape = copy_shape(self.input_desc[0][SHAPE])
        for idx in axis:
            if input_shape[idx] != 1:
                raise ValueError("The value of attr 'axis' is wrong , the squeezed axis must be 1, but got {}. 'axis': "
                                 "{}, input shape: {}".format(input_shape[idx], axis, input_shape))
            input_shape.pop(idx)
        self.output_desc[0][ORI_SHAPE] = input_shape


class ExpandDim(ExpandDimAndSqueeze):
    def infer_ori_shape(self):
        axis = self.copy_axis(self.input_desc[1][VALUE])
        input_shape = copy_shape(self.input_desc[0][SHAPE])
        for idx in axis:
            input_shape.insert(idx, 1)
        self.output_desc[0][ORI_SHAPE] = input_shape


class BroadcastTo(OpInfer):
    """BroadcastTo op."""

    def supported_format(self):
        io_format = ["ND"] * len(self.input_desc)
        return [",".join(io_format)]

    def infer_shape(self):
        """Broadcast op keeps ND format, so the output shape will not be changed"""
        self.output_desc[0][SHAPE] = self.output_desc[0][ORI_SHAPE]

    def infer_ori_shape(self):
        shape = self.input_desc[0][ORI_SHAPE]
        broad_shape = self.get_attr(SHAPE) if SHAPE in self.attr else self.input_desc[1][VALUE]
        if len(broad_shape) < len(shape):
            raise ValueError("The length of attr 'shape' must be >= the length of input shape, but got attr 'shape': "
                             "{}, input shape: {}".format(broad_shape, shape))
        pad_shape = [1] * (len(broad_shape) - len(shape)) + shape
        out_shape = []
        for i, b in enumerate(broad_shape):
            if b == -1:
                out_shape.append(pad_shape[i])
            else:
                out_shape.append(b)
        self.output_desc[0][ORI_SHAPE] = out_shape

    def post_process(self):
        if not isinstance(self.op_desc.get(ATTR), list):
            return
        for item in self.op_desc[ATTR]:
            if item[NAME] == SHAPE:
                item["ori_value"] = item[VALUE]
                item[VALUE] = self.output_desc[0][SHAPE]


class Tile(OpInfer):
    """BroadcastTo op."""

    def supported_format(self):
        return ["ND,ND"]

    def infer_shape(self):
        """Tile op keeps ND format, so the output shape will not be changed"""
        self.output_desc[0][SHAPE] = self.output_desc[0][ORI_SHAPE]

    def infer_ori_shape(self):
        shape = self.input_desc[0][ORI_SHAPE]
        multiples = self.input_desc[1][VALUE]
        if len(multiples) < len(shape):
            raise ValueError("The length of attr 'multiples' must be >= the length of input shape, but got attr "
                             "'multiples': {}, input shape: {}".format(multiples, shape))
        pad_shape = [1] * (len(multiples) - len(shape)) + shape
        out_shape = []
        for i, m in enumerate(multiples):
            out_shape.append(m * pad_shape[i])
        self.output_desc[0][ORI_SHAPE] = out_shape


class PagedAttention(OpInfer):
    """PagedAttention"""

    def supported_format(self):
        return ["ND,ND,ND,ND,ND,ND"]

    def infer_shape(self):
        """PagedAttention op keeps ND format, so the output shape will not be changed"""
        self.output_desc[0]["shape"] = self.output_desc[0]["ori_shape"]

    def infer_ori_shape(self):
        self.output_desc[0]["ori_shape"] = self.input_desc[0]["ori_shape"]


class ReshapeAndCache(OpInfer):
    """ReshapeAndCache"""

    def supported_format(self):
        return ["ND,ND,ND,ND,ND,ND"]

    def infer_shape(self):
        """ReshapeAndCache op keeps ND format, so the output shape will not be changed"""
        self.output_desc[0]["shape"] = self.output_desc[0]["ori_shape"]

    def infer_ori_shape(self):
        self.output_desc[0]["ori_shape"] = self.input_desc[0]["ori_shape"]


class PagedAttentionMask(PagedAttention):
    """PagedAttentionMask"""

    def supported_format(self):
        return ["ND,ND,ND,ND,ND,ND,ND"]


# Ge will convert dtype bool to int8, and ReLU will be expand to Greater op in expander,
# and the dtype of Greater op is bool, which is incompatible with bool.
# As a result akg will rise error when parsing Greater op with dtype int8.
# Expand And Sequeeze op will be expanded into Reshape op in expander,
# but in dynamic shape scenario, the meaning of -1 in Reshape op different from -1 in Expand And Sequeeze op.
# So this will lead to infer shape error.
# To solve these problems we need to cluster these ops in to subgraph and update info file here.
prims = {
    "Abs": Elemwise,
    "Neg": Elemwise,
    "Sqrt": Elemwise,
    "Rsqrt": Elemwise,
    "Reciprocal": Elemwise,
    "FastGeLU": Elemwise,
    "Round": Elemwise,
    "Assign": ElemwiseBinaryNoBroadcast,
    "Add": ElemwiseBinary,
    "Sub": ElemwiseBinary,
    "Mul": ElemwiseBinary,
    "Div": ElemwiseBinary,
    "Mod": ElemwiseBinary,
    "RealDiv": ElemwiseBinary,
    "Maximum": ElemwiseBinary,
    "Minimum": ElemwiseBinary,
    "MatMul": MatMul,
    "BatchMatMul": BatchMatMul,
    "ReduceSum": Reduce,
    "Reshape": Reshape,
    "ExpandDims": ExpandDim,
    "Squeeze": Squeeze,
    "BroadcastTo": BroadcastTo,
    "Tile": Tile,
    "Log": Elemwise,
    "Exp": Elemwise,
    "Pow": Elemwise,
    "Sign": Elemwise,
    "ReLU": Elemwise,
    "Tanh": Elemwise,
    "ReduceMax": Reduce,
    "ReduceMin": Reduce,
    "Cast": Cast,
    "PagedAttention": PagedAttention,
    "PagedAttentionMask": PagedAttentionMask,
    "ReshapeAndCache": ReshapeAndCache,
}


def convert_to_default_format(desc):
    """Convert to DefaultFormat"""
    default_format = ["ND", "NCHW", "NHWC", "HWCN", DEFAULT_FORMAT]
    for _, input_desc in enumerate(desc[INPUT_DESC]):
        if input_desc[0][FORMAT] in default_format:
            input_desc[0][FORMAT] = DEFAULT_FORMAT
        if not input_desc[0][SHAPE]:
            input_desc[0][SHAPE] = [1]
    for _, op_desc in enumerate(desc[OP_DESC]):
        for _, input_desc in enumerate(op_desc[INPUT_DESC]):
            if input_desc[0][FORMAT] in default_format:
                input_desc[0][FORMAT] = DEFAULT_FORMAT
            if not input_desc[0][SHAPE]:
                input_desc[0][SHAPE] = [1]
        for _, output_desc in enumerate(op_desc[OUTPUT_DESC]):
            if output_desc[FORMAT] in default_format:
                output_desc[FORMAT] = DEFAULT_FORMAT
            if not output_desc[SHAPE]:
                output_desc[SHAPE] = [1]
    for _, output_desc in enumerate(desc[OUTPUT_DESC]):
        if output_desc[FORMAT] in default_format:
            output_desc[FORMAT] = DEFAULT_FORMAT
        if not output_desc[SHAPE]:
            output_desc[SHAPE] = [1]


def update_global_input_desc(info_desc, args):
    """Update the global input of the fused info file"""

    def _convert_tbe_type(tbe_type, ori_type):
        if tbe_type == "float":
            return FLOAT32
        if tbe_type == "int8" and ori_type == "bool":
            # GE pass int8 here if data type is bool, but we must return bool back to GE, otherwise GE will
            # raise an error "current op does not support bool"
            return ori_type
        return tbe_type

    def _covert_tbe_shape(tbe_shape):
        if not tbe_shape:
            return [1]
        return copy_shape(tbe_shape)

    if isinstance(info_desc.get(INPUT_DESC), list):
        for i, desc in enumerate(info_desc[INPUT_DESC]):
            desc[0][ORI_DATA_TYPE] = desc[0][DATA_TYPE]
            desc[0][DATA_TYPE] = _convert_tbe_type(args[i]["dtype"], desc[0][ORI_DATA_TYPE])
            desc[0][ORI_FORMAT] = args[i].get(ORI_FORMAT, desc[0][FORMAT])
            desc[0][FORMAT] = args[i][FORMAT]
            desc[0][ORI_SHAPE] = _covert_tbe_shape(args[i].get(ORI_SHAPE, desc[0][SHAPE]))
            desc[0][SHAPE] = list(args[i][SHAPE])


def update_global_output_desc(info_desc, tensor_desc):
    """Update the global output of the fused info file"""
    for i, desc in enumerate(info_desc[OUTPUT_DESC]):
        tensor_name = desc[TENSOR_NAME]
        if tensor_name not in tensor_desc:
            raise RuntimeError("tensor '{}' not exist in op_desc".format(tensor_name))
        info_desc[OUTPUT_DESC][i] = tensor_desc[tensor_name]


def update_op_input_desc(op_desc, tensor_desc):
    """Update the input of operator"""
    if not isinstance(op_desc.get(INPUT_DESC), list):
        return
    inputs_type_orig = []
    inputs_type = []
    const_inputs_idx = []
    for i, desc in enumerate(op_desc[INPUT_DESC]):
        for j, item in enumerate(desc):
            if VALUE in item:
                inputs_type_orig.append(None)
                inputs_type.append(None)
                const_inputs_idx.append(i)
                item[ORI_DATA_TYPE] = item[DATA_TYPE]
                item[ORI_FORMAT] = item[FORMAT]
                item[ORI_SHAPE] = copy_shape(item[SHAPE])
            else:
                inputs_type_orig.append(item[DATA_TYPE])
                tensor_name = item[TENSOR_NAME]
                if tensor_name not in tensor_desc:
                    raise RuntimeError("tensor '{}' used without initialization".format(tensor_name))
                # update op input
                desc[j] = tensor_desc[tensor_name]
                inputs_type.append(tensor_desc[tensor_name][DATA_TYPE])
    # update op const input's data type
    for _, idx in enumerate(const_inputs_idx):
        const_value_type = op_desc[INPUT_DESC][idx][0][DATA_TYPE]
        if const_value_type in inputs_type_orig:
            op_desc[INPUT_DESC][idx][0][DATA_TYPE] = inputs_type[inputs_type_orig.index(const_value_type)]
        # cache op const input
        tensor_desc[op_desc[INPUT_DESC][idx][0][TENSOR_NAME]] = op_desc[INPUT_DESC][idx][0]


def cache_input_tensors(tensor_desc, input_desc):
    """Cache input tensor desc"""
    if isinstance(input_desc, list):
        for desc in input_desc:
            for item in desc:
                tensor_desc[item[TENSOR_NAME]] = item


def cache_output_tensors(tensor_desc, output_desc):
    """Cache output tensor desc"""
    for item in output_desc:
        tensor_desc[item[TENSOR_NAME]] = item


def save(filename, contents):
    """Save to file"""
    with os.fdopen(os.open(filename, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o660), 'w') as f:
        f.write(contents)


def update_akg_info(args, info_path, kernel_name=None):
    """Update akg info base on the current inputs provided by GE"""
    with open(info_path, 'r') as f:
        info_str = f.read()
        desc = json.loads(info_str)
        desc["op_ori"] = desc[OP]
        desc[OP] = kernel_name if kernel_name else desc[OP]
        tensor_desc = {}  # {tensor_name: tensor_desc}

        # Update input_desc
        update_global_input_desc(desc, args)
        # cache global input
        cache_input_tensors(tensor_desc, desc.get(INPUT_DESC))
        # Update info global config
        InfoGlobalConfig.enable_cce_lib = desc.get("enable_cce_lib")
        target_info = desc.get("target_info")
        if target_info is not None:
            InfoGlobalConfig.ascend_arch = target_info.get("arch")

        # Update op_desc
        for _, op_desc in enumerate(desc[OP_DESC]):
            update_op_input_desc(op_desc, tensor_desc)
            op_name = op_desc[NAME]
            if op_name not in prims:
                raise KeyError("Not supported op: {}".format(op_name))
            prim = prims.get(op_name)(op_desc)
            prim.update()
            # cache op output
            cache_output_tensors(tensor_desc, op_desc[OUTPUT_DESC])

        # Update output_desc
        update_global_output_desc(desc, tensor_desc)

        # Update data format to DefaultFormat
        convert_to_default_format(desc)

        # GE backend must use old CCE
        desc["backend"] = "GE"

        return desc


def save_updated_akg_info(*args):
    """Save the updated akg info."""
    info_path = args[-2]
    kernel_name = args[-1]
    if not isinstance(info_path, str):
        # in this case, kernel_name is not passed by GE, skip compiling
        return ""
    updated_desc = update_akg_info(args, info_path, kernel_name)
    real_info_path = os.path.join(os.path.realpath(os.path.dirname(info_path)), kernel_name + ".info")
    # Save the updated info file
    save(real_info_path, json.dumps(updated_desc))
    return real_info_path


def create_dirs(*dirs):
    """Create directories."""
    for d in dirs:
        if not os.path.isdir(d):
            try:
                os.makedirs(d)
            except OSError as err:
                # File exists
                if err.errno == 17:
                    pass
                else:
                    raise err


def copy_file(src_path, dst_path):
    """Copy file to dst."""
    try:
        if os.path.isfile(dst_path):
            os.remove(dst_path)
    except OSError:
        pass

    try:
        shutil.copy(src_path, dst_path)
    except PermissionError:
        # If dst_path already exits and only has READ permission
        pass


def _compile_subprocess(kernel_meta_dirs, info_path, is_lite=True, compile_backend=None, attrs=None):
    """Use a new process to compile info."""
    kernel_meta_parent_dir, kernel_meta_dir = kernel_meta_dirs
    my_env = os.environ
    my_env["MS_COMPILER_CACHE_PATH"] = kernel_meta_parent_dir
    my_env["KERNEL_META_DIR"] = kernel_meta_dir
    compiler = os.path.join(os.path.split(os.path.realpath(__file__))[0], "compiler.py")
    if is_lite:
        run_args = [sys.executable, compiler, info_path]
    else:
        run_args = [sys.executable, compiler, info_path, compile_backend, attrs, kernel_meta_parent_dir]
    compile_result = subprocess.run(run_args, text=True, check=False, capture_output=True, env=my_env)
    return compile_result


def search_supported_types_formats(info):
    """Get the supported data types and formats of the fused info file"""

    class DfsSearcher:
        """Use DFS"""

        def __init__(self, top_io_names, ops_desc):
            self.supported_types = []
            self.supported_formats = []
            self.top_io_names = top_io_names
            self.tensor_types = {}
            self.tensor_formats = {}
            self.ops_desc = ops_desc
            self.cache = []

        def set_current_format(self, cur_format, io_names):
            """set tensor format"""
            for i, fmt in enumerate(cur_format):
                if self.tensor_formats.get(io_names[i], fmt) != fmt:
                    return False
                self.tensor_formats[io_names[i]] = fmt
            return True

        def set_current_type(self, cur_type, io_names):
            """set tensor data type"""
            for i, data_type in enumerate(cur_type):
                if self.tensor_types.get(io_names[i], data_type) != data_type:
                    return False
                self.tensor_types[io_names[i]] = data_type
            return True

        def get_desc(self, opid):
            """get desc"""
            if opid < len(self.cache):
                return self.cache[opid]
            desc = self.ops_desc[opid]
            io_names = [item[TENSOR_NAME] for desc in desc[INPUT_DESC] for item in desc]
            io_names.append(desc[OUTPUT_DESC][0][TENSOR_NAME])
            op_name = desc[NAME]
            if op_name not in prims:
                raise KeyError("Not supported op: {}".format(op_name))
            prim = prims.get(op_name)(desc)
            io_formats = [f.split(",") for f in prim.supported_format()]
            io_types = [t.split(",") for t in prim.supported_type()]
            self.cache.append((io_formats, io_types, tuple(io_names)))
            return self.cache[-1]

        def search_types(self, opid):
            """search the supported types"""
            if opid == len(self.ops_desc):
                top_tensor_types = tuple(self.tensor_types.get(t) for t in self.top_io_names)
                self.supported_types.append(top_tensor_types)
                return
            _, op_io_types, io_names = self.get_desc(opid)
            for cur_type in op_io_types:
                bak_tensor_types = copy.deepcopy(self.tensor_types)
                if self.set_current_type(cur_type, io_names):
                    self.search_types(opid + 1)
                self.tensor_types = bak_tensor_types

        def search_formats(self, opid):
            """search the supported formats"""
            if opid == len(self.ops_desc):
                top_tensor_formats = tuple(self.tensor_formats.get(t) for t in self.top_io_names)
                self.supported_formats.append(top_tensor_formats)
                return
            op_io_formats, _, io_names = self.get_desc(opid)
            for cur_format in op_io_formats:
                bak_tensor_formats = copy.deepcopy(self.tensor_formats)
                if self.set_current_format(cur_format, io_names):
                    self.search_formats(opid + 1)
                self.tensor_formats = bak_tensor_formats

    def remove_dup(data):
        res = []
        data_str = []
        for _, t in enumerate(data):
            t_str = ",".join(t)
            if t_str not in data_str:
                data_str.append(t_str)
                res.append(t)
        return res

    top_io_names = [t[0][TENSOR_NAME] for t in info[INPUT_DESC]] + [t[TENSOR_NAME] for t in info[OUTPUT_DESC]]
    handle = DfsSearcher(top_io_names, info[OP_DESC])
    handle.search_types(0)
    handle.search_formats(0)
    return remove_dup(handle.supported_types), remove_dup(handle.supported_formats)


def op_select_format(*args, **kwags):
    """Entrance for format/data type selection, will invoked by GE"""
    info_path = args[-1]
    desc = update_akg_info(args, info_path)
    supported_io_type, supported_io_format = search_supported_types_formats(desc)
    if not supported_io_type or not supported_io_format:
        raise RuntimeError("Select format failed for info: {}".format(info_path))
    input_num = len(desc[INPUT_DESC])
    output_num = len(desc[OUTPUT_DESC])
    param_list = []
    for i in range(input_num + output_num):
        dtype_list = [item[i] for item in supported_io_type] * len(supported_io_format)
        format_list = functools.reduce(lambda x, y: x + y,
                                       [[item[i]] * len(supported_io_type) for item in supported_io_format])
        classify = "input" + str(i) if i < input_num else "output" + str(i - input_num)
        name = "x" + str(i) if i < input_num else "y" + str(i - input_num)
        param = gen_param(classify=classify,
                          name=name,
                          datatype=",".join(dtype_list),
                          format=",".join(format_list))
        param_list.append(param)
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def custom(*args, **kwags):
    """Entrance for akg info compiling, will invoked by GE"""
    kernel_name = args[-1]
    real_info_path = save_updated_akg_info(*args)
    if not real_info_path:
        return
    kernel_meta_parent_dir = get_current_build_config("kernel_meta_parent_dir")
    kernel_meta_dir = "kernel_meta"
    compile_result = _compile_subprocess([kernel_meta_parent_dir, kernel_meta_dir], real_info_path, is_lite=True)
    json_path = os.path.join(kernel_meta_parent_dir, kernel_meta_dir, kernel_name + JSON_SUFFIX)
    if compile_result.returncode or not os.path.exists(json_path):
        raise RuntimeError("Compile {} failed! Detailed compile message: {}, {}"
                           .format(kernel_name, compile_result.stdout.strip(), compile_result.stderr.strip()))


def custom_train(*args, **kwags):
    """Entrance for akg info compiling, will invoked by GE"""

    def _get_optimized_info_path():
        """Get the info optimized by akg."""
        target_info = "target_info"
        file_path = os.path.join(composite_graph_dir, kernel_name + ".info")
        if not os.path.isfile(file_path):
            return real_info_path
        with open(real_info_path, 'r') as f:
            desc = json.loads(f.read())
            if target_info in desc:
                with open(file_path, 'r') as fo:
                    info_desc = json.loads(fo.read())
                    info_desc[target_info] = desc[target_info]
                save(file_path, json.dumps(info_desc))
        return file_path

    info_path = args[-2]
    kernel_name = args[-1]
    real_info_path = save_updated_akg_info(*args)
    if not real_info_path:
        return
    info_dir = os.path.realpath(os.path.dirname(info_path))
    kernel_meta_parent_dir = get_current_build_config("kernel_meta_parent_dir")
    kernel_meta = "kernel_meta"
    kernel_meta_dir = os.path.join(kernel_meta_parent_dir, kernel_meta)
    akg_compile_dir = os.path.join(info_dir, "akg")
    tbe_compile_dir = os.path.join(info_dir, "tbe")
    composite_graph_dir = os.path.join(info_dir, "composite")  # save akg optimized info
    akg_kernel_meta_dir = os.path.join(akg_compile_dir, kernel_meta)  # save akg compile result
    tbe_kernel_meta_dir = os.path.join(tbe_compile_dir, kernel_meta)  # save tbe compile result
    create_dirs(kernel_meta_dir, composite_graph_dir, akg_kernel_meta_dir, tbe_kernel_meta_dir)
    # Compile with AKG
    attr = {"dump_composite_graph": composite_graph_dir, "optimize_for_tbe": True}
    attrs = json.dumps(attr)
    akg_compile_result = _compile_subprocess([akg_compile_dir, kernel_meta], real_info_path,
                                             is_lite=False, compile_backend="AKG", attrs=attrs)
    json_path = os.path.join(akg_kernel_meta_dir, kernel_name + JSON_SUFFIX)
    o_path = os.path.join(akg_kernel_meta_dir, kernel_name + O_SUFFIX)
    if not os.path.exists(json_path):
        # Compile with TBE
        optimized_info_path = _get_optimized_info_path()
        tbe_compile_result = _compile_subprocess([tbe_compile_dir, kernel_meta], optimized_info_path,
                                                 is_lite=False, compile_backend="TBE", attrs=attrs)
        json_path = os.path.join(tbe_kernel_meta_dir, kernel_name + JSON_SUFFIX)
        o_path = os.path.join(tbe_kernel_meta_dir, kernel_name + O_SUFFIX)
        if not os.path.exists(json_path):
            raise RuntimeError("Compile {} failed! Detailed akg compile message: {}, {}\n"
                               "Detailed tbe compile message: {}, {}"
                               .format(kernel_name,
                                       akg_compile_result.stdout.strip(), akg_compile_result.stderr.strip(),
                                       tbe_compile_result.stdout.strip(), tbe_compile_result.stderr.strip()))
    copy_file(json_path, os.path.join(kernel_meta_dir, kernel_name + JSON_SUFFIX))
    copy_file(o_path, os.path.join(kernel_meta_dir, kernel_name + O_SUFFIX))
