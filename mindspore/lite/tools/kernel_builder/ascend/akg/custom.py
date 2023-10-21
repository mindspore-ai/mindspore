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

from tbe.common.buildcfg import get_current_build_config
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json

BLOCK = 16
FP16_MAX = 65504


def copy_shape(shape):
    """Deep copy shape"""
    res = []
    if isinstance(shape, int):
        shape = [shape]
    for _, s in enumerate(shape):
        res.append(s)
    return res


class OpInfer:
    """Base infer class, used to provide supported formats and data type of each op and update each of"""

    def __init__(self, op_desc):
        self.name = op_desc["name"]
        self.op_desc = op_desc
        self.input_desc = []
        self.output_desc = []
        self.attr = {}
        if isinstance(op_desc.get("input_desc"), list):
            for desc in op_desc["input_desc"]:
                for item in desc:
                    self.input_desc.append(item)
        if isinstance(op_desc.get("attr"), list):
            for item in op_desc["attr"]:
                self.attr[item["name"]] = item
        if isinstance(op_desc.get("output_desc"), list):
            for item in op_desc["output_desc"]:
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
        return self.attr.get(key)["value"]

    def set_attr(self, key, value):
        """set the value of attr"""
        if key not in self.attr:
            raise KeyError("Can not find attr '{}' in op '{}'".format(key, self.name))
        self.attr.get(key)["value"] = value

    def supported_type(self):
        """get the supported data type of current op"""
        keep_fp32 = False
        for item in self.input_desc:
            # check if type can reduce precision
            value = item.get("value", None)
            if item["data_type"] == "float32" and value is not None and abs(value) > FP16_MAX:
                keep_fp32 = True
                break
        io_type = ",".join([t["data_type"] for t in self.input_desc] + [t["data_type"] for t in self.output_desc])
        fp32_type = io_type.replace("float16", "float32")
        fp16_type = io_type.replace("float32", "float16")
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
        self.output_desc[0]["data_type"] = self.input_desc[0]["data_type"]

    def infer_format(self):
        """infer format"""
        self.output_desc[0]["format"] = self.input_desc[0]["format"]

    def infer_shape(self):
        """infer shape"""
        self.output_desc[0]["shape"] = copy_shape(self.input_desc[0]["shape"])

    def infer_ori_shape(self):
        """infer original shape"""
        for _, desc in enumerate(self.output_desc):
            desc["ori_shape"] = copy_shape(desc["shape"])

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
            desc["ori_data_type"] = desc["data_type"]
            desc["ori_format"] = desc["format"]
        self.infer_ori_shape()
        self.infer()
        self.post_process()


class Elemwise(OpInfer):
    """Elemwise op with one input and one output."""

    def supported_format(self):
        if self.name == "Reciprocal":
            supported_formats = ["ND,ND"]
            # pad will cause 'divided by 0'
            if self.is_nz(self.input_desc[0]["shape"]):
                self.update_format(supported_formats, "FRACTAL_NZ,FRACTAL_NZ")
            return supported_formats
        return ["ND,ND", "FRACTAL_NZ,FRACTAL_NZ", "NC1HWC0,NC1HWC0", "FRACTAL_Z,FRACTAL_Z"]

    def infer_ori_shape(self):
        self.output_desc[0]["ori_shape"] = self.input_desc[0]["ori_shape"]


class ElemwiseBinaryNoBroadcast(OpInfer):
    """Elemwise op with two inputs and one output, not supports broadcast."""

    def supported_format(self):
        return ["ND,ND,ND", "FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ", "NC1HWC0,NC1HWC0,NC1HWC0",
                "FRACTAL_Z,FRACTAL_Z,FRACTAL_Z"]

    def infer_ori_shape(self):
        self.output_desc[0]["ori_shape"] = self.input_desc[0]["ori_shape"]


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
        sh0, sh1 = self.input_desc[0]["shape"], self.input_desc[1]["shape"]
        supported_formats = ["ND,ND,ND"]
        is_const_0 = ("value" in self.input_desc[0])
        is_const_1 = ("value" in self.input_desc[1])
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
        format0, format1 = self.input_desc[0]["format"], self.input_desc[1]["format"]
        for f in special_formats:
            if format0.find(f) != -1:
                self.output_desc[0]["format"] = format0
                return
        self.output_desc[0]["format"] = format1

    def infer_shape(self):
        sh0, sh1 = self.input_desc[0]["shape"], self.input_desc[1]["shape"]
        if sh0 == sh1:
            self.output_desc[0]["shape"] = copy_shape(sh0)
        format0, format1 = self.input_desc[0]["format"], self.input_desc[1]["format"]
        if format0 != format1:
            new_sh0 = self.nd2fractal_nz(sh0)
            new_sh1 = self.nd2fractal_nz(sh1)
            if format0 == "FRACTAL_NZ" and new_sh1:
                _, _, out_shape = self.broadcast_shape(sh0, new_sh1)
                self.output_desc[0]["shape"] = out_shape
                return
            if format1 == "FRACTAL_NZ" and new_sh0:
                _, _, out_shape = self.broadcast_shape(new_sh0, sh1)
                self.output_desc[0]["shape"] = out_shape
                return
        _, _, out_shape = self.broadcast_shape(sh0, sh1)
        self.output_desc[0]["shape"] = out_shape

    def infer_ori_shape(self):
        sh0, sh1 = self.input_desc[0]["ori_shape"], self.input_desc[1]["ori_shape"]
        _, _, out_shape = self.broadcast_shape(sh0, sh1)
        self.output_desc[0]["ori_shape"] = out_shape


class MatMul(OpInfer):
    """MatMul op."""

    def supported_format(self):
        input_num = len(self.input_desc)
        if input_num == 2:
            return ["FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ"]
        if input_num == 3:
            bias_shape = self.input_desc[2]["shape"]
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
        sh0, sh1 = self.input_desc[0]["shape"], self.input_desc[1]["shape"]
        format0, format1 = self.input_desc[0]["format"], self.input_desc[1]["format"]
        trans_a, trans_b = self.get_attr("transpose_a"), self.get_attr("transpose_b")
        if format0 != format1 or len(sh0) != len(sh1):
            raise ValueError("For '{}', input '{}' and '{}' are not supported"
                             .format(self.name, self.input_desc[0], self.input_desc[1]))
        if format0 != "FRACTAL_NZ" and len(sh0) >= 2:
            self.output_desc[0]["shape"] = self.nd_infer(sh0, sh1, trans_a, trans_b)
        elif format0 == "FRACTAL_NZ" and len(sh0) >= 4:
            m1, m0 = sh0[-3], sh0[-2]
            if trans_a:
                m1, m0 = sh0[-4], sh0[-1]
            n1, n0 = sh1[-4], sh1[-1]
            if trans_b:
                n1, n0 = sh1[-3], sh1[-2]
            self.output_desc[0]["shape"] = sh0[:-4] + [n1, m1, m0, n0]
        else:
            raise ValueError("For '{}', input '{}' and '{}' are not supported"
                             .format(self.name, self.input_desc[0], self.input_desc[1]))

    def infer_ori_shape(self):
        sh0, sh1 = self.input_desc[0]["ori_shape"], self.input_desc[1]["ori_shape"]
        trans_a, trans_b = self.get_attr("transpose_a"), self.get_attr("transpose_b")
        self.output_desc[0]["ori_shape"] = self.nd_infer(sh0, sh1, trans_a, trans_b)

    def post_process(self):
        self.op_desc["attr"].append({"data_type": "str", "name": "left_format", "value": self.input_desc[0]["format"]})
        self.op_desc["attr"].append({"data_type": "str", "name": "right_format", "value": self.input_desc[1]["format"]})
        self.op_desc["attr"].append({"data_type": "str", "name": "dst_type", "value": self.output_desc[0]["data_type"]})


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
        axis_input = self.input_desc[1]["value"]
        axis = []
        if isinstance(axis_input, int):
            axis = [axis_input + rank if axis_input < 0 else axis_input]
        else:
            axis = [i + rank if i < 0 else i for i in axis_input]
        return axis

    def supported_type(self):
        in_type = self.input_desc[0]["data_type"]
        if in_type == "float16":
            return ["float16,int64,float16", "float32,int64,float32"]
        if in_type == "float32":
            return ["float32,int64,float32"]
        io_type = ",".join([in_type, "int64", in_type])
        return [io_type]

    def supported_format(self):
        supported_formats = ["ND,DefaultFormat,ND"]
        shape = self.input_desc[0]["shape"]
        rank = len(shape)
        axis = self._get_axis(rank)
        if self.is_nz(shape):
            if self._out_nz(rank, axis):
                supported_formats.append("FRACTAL_NZ,DefaultFormat,FRACTAL_NZ")
        return supported_formats

    def infer_shape(self):
        ori_format, cur_format = self.input_desc[0]["ori_format"], self.input_desc[0]["format"]
        if cur_format == "FRACTAL_NZ" and cur_format != ori_format:
            ori_shape, cur_shape = self.input_desc[0]["ori_shape"], self.input_desc[0]["shape"]
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
            self.input_desc[1]["value"] = new_axis
            self.input_desc[1]["shape"] = [len(new_axis)]
            self.output_desc[0]["shape"] = self._reduced_shape(cur_shape, new_axis, self.get_attr("keep_dims"))
        else:
            self.output_desc[0]["shape"] = self.output_desc[0]["ori_shape"]

    def infer_ori_shape(self):
        shape = self.input_desc[0]["ori_shape"]
        rank = len(shape)
        axis = self._get_axis(rank)
        self.output_desc[0]["ori_shape"] = self._reduced_shape(shape, axis, self.get_attr("keep_dims"))


class Reshape(OpInfer):
    """Reshape op."""

    def supported_format(self):
        return ["ND,DefaultFormat,ND"]

    def infer_shape(self):
        """Reshape keeps ND format, so the output shape will not be changed"""
        self.output_desc[0]["shape"] = self.output_desc[0]["ori_shape"]

    def infer_ori_shape(self):
        shape = self.input_desc[0]["ori_shape"]
        out_shape = copy_shape(self.input_desc[1]["value"])
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
        self.output_desc[0]["ori_shape"] = out_shape

    def post_process(self):
        self.input_desc[1]["ori_value"] = self.input_desc[1]["value"]
        self.input_desc[1]["value"] = self.output_desc[0]["shape"]


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
        axis = self.copy_axis(self.input_desc[1]["value"])
        input_shape = copy_shape(self.input_desc[0]["shape"])
        for idx in axis:
            if input_shape[idx] != 1:
                raise ValueError("The value of attr 'axis' is wrong , the squeezed axis must be 1, but got {}. 'axis': "
                                 "{}, input shape: {}".format(input_shape[idx], axis, input_shape))
            input_shape.pop(idx)
        self.output_desc[0]["ori_shape"] = input_shape


class ExpandDim(ExpandDimAndSqueeze):
    def infer_ori_shape(self):
        axis = self.copy_axis(self.input_desc[1]["value"])
        input_shape = copy_shape(self.input_desc[0]["shape"])
        for idx in axis:
            input_shape.insert(idx, 1)
        self.output_desc[0]["ori_shape"] = input_shape


class BroadcastTo(OpInfer):
    """BroadcastTo op."""

    def supported_format(self):
        return ["ND,ND"]

    def infer_shape(self):
        """Broadcast op keeps ND format, so the output shape will not be changed"""
        self.output_desc[0]["shape"] = self.output_desc[0]["ori_shape"]

    def infer_ori_shape(self):
        shape = self.input_desc[0]["ori_shape"]
        broad_shape = self.get_attr("shape")
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
        self.output_desc[0]["ori_shape"] = out_shape

    def post_process(self):
        for item in self.op_desc["attr"]:
            if item["name"] == "shape":
                item["ori_value"] = item["value"]
                item["value"] = self.output_desc[0]["shape"]


class Tile(OpInfer):
    """BroadcastTo op."""

    def supported_format(self):
        return ["ND,ND"]

    def infer_shape(self):
        """Tile op keeps ND format, so the output shape will not be changed"""
        self.output_desc[0]["shape"] = self.output_desc[0]["ori_shape"]

    def infer_ori_shape(self):
        shape = self.input_desc[0]["ori_shape"]
        multiples = self.input_desc[1]["value"]
        if len(multiples) < len(shape):
            raise ValueError("The length of attr 'multiples' must be >= the length of input shape, but got attr "
                             "'multiples': {}, input shape: {}".format(multiples, shape))
        pad_shape = [1] * (len(multiples) - len(shape)) + shape
        out_shape = []
        for i, m in enumerate(multiples):
            out_shape.append(m * pad_shape[i])
        self.output_desc[0]["ori_shape"] = out_shape


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
    "BatchMatMul": MatMul,
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
}


def convert_to_default_format(desc):
    """Convert to DefaultFormat"""
    default_format = ["ND", "NCHW", "NHWC", "HWCN", "DefaultFormat"]
    for _, input_desc in enumerate(desc["input_desc"]):
        if input_desc[0]["format"] in default_format:
            input_desc[0]["format"] = "DefaultFormat"
    for _, op_desc in enumerate(desc["op_desc"]):
        for _, input_desc in enumerate(op_desc["input_desc"]):
            if input_desc[0]["format"] in default_format:
                input_desc[0]["format"] = "DefaultFormat"
        for _, output_desc in enumerate(op_desc["output_desc"]):
            if output_desc["format"] in default_format:
                output_desc["format"] = "DefaultFormat"
    for _, output_desc in enumerate(desc["output_desc"]):
        if output_desc["format"] in default_format:
            output_desc["format"] = "DefaultFormat"


def update_global_input_desc(info_desc, args):
    """Update the global input of the fused info file"""

    def _convert_tbe_type(tbe_type):
        if tbe_type == "float":
            return "float32"
        return tbe_type

    def _covert_tbe_shape(tbe_shape):
        if not tbe_shape:
            return [1]
        return copy_shape(tbe_shape)

    if isinstance(info_desc.get("input_desc"), list):
        for i, desc in enumerate(info_desc["input_desc"]):
            desc[0]["ori_data_type"] = desc[0]["data_type"]
            desc[0]["data_type"] = _convert_tbe_type(args[i]["dtype"])
            desc[0]["ori_format"] = args[i].get("ori_format", desc[0]["format"])
            desc[0]["format"] = args[i]["format"]
            desc[0]["ori_shape"] = _covert_tbe_shape(args[i].get("ori_shape", desc[0]["shape"]))
            desc[0]["shape"] = list(args[i]["shape"])


def update_global_output_desc(info_desc, tensor_desc):
    """Update the global output of the fused info file"""
    for i, desc in enumerate(info_desc["output_desc"]):
        tensor_name = desc["tensor_name"]
        if tensor_name not in tensor_desc:
            raise RuntimeError("tensor '{}' not exist in op_desc".format(tensor_name))
        info_desc["output_desc"][i] = tensor_desc[tensor_name]


def update_op_input_desc(op_desc, tensor_desc):
    """Update the input of operator"""
    if not isinstance(op_desc.get("input_desc"), list):
        return
    inputs_type_orig = []
    inputs_type = []
    const_inputs_idx = []
    for i, desc in enumerate(op_desc["input_desc"]):
        for j, item in enumerate(desc):
            if "value" in item:
                inputs_type_orig.append(None)
                inputs_type.append(None)
                const_inputs_idx.append(i)
                item["ori_data_type"] = item["data_type"]
                item["ori_format"] = item["format"]
                item["ori_shape"] = copy_shape(item["shape"])
            else:
                inputs_type_orig.append(item["data_type"])
                tensor_name = item["tensor_name"]
                if tensor_name not in tensor_desc:
                    raise RuntimeError("tensor '{}' used without initialization".format(tensor_name))
                # update op input
                desc[j] = tensor_desc[tensor_name]
                inputs_type.append(tensor_desc[tensor_name]["data_type"])
    # update op const input's data type
    for _, idx in enumerate(const_inputs_idx):
        const_value_type = op_desc["input_desc"][idx][0]["data_type"]
        if const_value_type in inputs_type_orig:
            op_desc["input_desc"][idx][0]["data_type"] = inputs_type[inputs_type_orig.index(const_value_type)]
        # cache op const input
        tensor_desc[op_desc["input_desc"][idx][0]["tensor_name"]] = op_desc["input_desc"][idx][0]


def cache_input_tensors(tensor_desc, input_desc):
    """Cache input tensor desc"""
    if isinstance(input_desc, list):
        for desc in input_desc:
            for item in desc:
                tensor_desc[item["tensor_name"]] = item


def cache_output_tensors(tensor_desc, output_desc):
    """Cache output tensor desc"""
    for item in output_desc:
        tensor_desc[item["tensor_name"]] = item


def save(filename, contents):
    """Save to file"""
    with os.fdopen(os.open(filename, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o660), 'w') as f:
        f.write(contents)


def update_akg_info(args, info_path, kernel_name=None):
    """Update akg info base on the current inputs provided by GE"""
    with open(info_path, 'r') as f:
        info_str = f.read()
        desc = json.loads(info_str)
        desc["op_ori"] = desc["op"]
        desc["op"] = kernel_name if kernel_name else desc["op"]
        tensor_desc = {}  # {tensor_name: tensor_desc}

        # Update input_desc
        update_global_input_desc(desc, args)
        # cache global input
        cache_input_tensors(tensor_desc, desc.get("input_desc"))

        # Update op_desc
        for _, op_desc in enumerate(desc["op_desc"]):
            update_op_input_desc(op_desc, tensor_desc)
            op_name = op_desc["name"]
            if op_name not in prims:
                raise KeyError("Not supported op: {}".format(op_name))
            prim = prims.get(op_name)(op_desc)
            prim.update()
            # cache op output
            cache_output_tensors(tensor_desc, op_desc["output_desc"])

        # Update output_desc
        update_global_output_desc(desc, tensor_desc)

        # Update data format to DefaultFormat
        convert_to_default_format(desc)

        return desc


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
            io_names = [item["tensor_name"] for desc in desc["input_desc"] for item in desc]
            io_names.append(desc["output_desc"][0]["tensor_name"])
            op_name = desc["name"]
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

    top_io_names = [t[0]["tensor_name"] for t in info["input_desc"]] + [t["tensor_name"] for t in info["output_desc"]]
    handle = DfsSearcher(top_io_names, info["op_desc"])
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
    input_num = len(desc["input_desc"])
    output_num = len(desc["output_desc"])
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
    info_path = args[-2]
    kernel_name = args[-1]
    if not isinstance(info_path, str):
        # in this case, kernel_name is not passed by GE, skip compiling
        return
    updated_desc = update_akg_info(args, info_path, kernel_name)
    real_info_path = os.path.join(os.path.realpath(os.path.dirname(info_path)), kernel_name + ".info")
    # Save the updated info file which will be compiled by AKG
    save(real_info_path, json.dumps(updated_desc))
    my_env = os.environ
    my_env["MS_COMPILER_CACHE_PATH"] = get_current_build_config("kernel_meta_parent_dir")
    my_env["KERNEL_META_DIR"] = "kernel_meta"
    compiler = os.path.join(os.path.split(os.path.realpath(__file__))[0], "compiler.py")
    compile_result = subprocess.run([sys.executable, compiler, real_info_path], text=True, check=False,
                                    capture_output=True, env=my_env)
    json_path = os.path.join(my_env["MS_COMPILER_CACHE_PATH"], my_env["KERNEL_META_DIR"], kernel_name + ".json")
    if compile_result.returncode or not os.path.exists(json_path):
        raise RuntimeError("Compile {} failed! Detailed compile message: {}, {}"
                           .format(kernel_name, compile_result.stdout.strip(), compile_result.stderr.strip()))
