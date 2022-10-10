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

"""build tbe kernel"""

from __future__ import absolute_import
import os
import json
import functools
from tbe_topi import OpPattern, get_op_reg_info
from te import tvm
from te.utils import shape_util
from te.platform.cce_conf import te_set_version
from tbe.common.buildcfg import build_config
from tbe.dsl import auto_schedule
from tbe.dsl import build as tbe_build
import tbe.common.context.op_context as op_context


def initialize(kernel_meta_parent_dir):
    """Initialize the TBE compile environment."""
    os.environ["CONTEXT_MODELCOMPILING"] = "TRUE"
    # socVersion, coreType, coreNum, l1Fusion, l2Mode, l2Fusion
    soc_info = ["Ascend910A", "", "", "false", "2", "false",
                {"op_impl_mode": "",
                 "op_debug_level": "3",
                 "op_impl_mode_list": [],
                 "op_debug_dir": kernel_meta_parent_dir,
                 "vector_fp_ceiling": "",
                 "mdl_bank_path": "",
                 "op_bank_path": ""}]
    if not te_set_version(*soc_info):
        raise RuntimeError("Set version failed")


def update_config(config, op_names):
    """Update build config."""
    bool_storage_as_1bit_oplist = ["Asinh", "Atanh", "Acosh", "Asin", "Atan2", "Acos", "Pow", "Elu", "Select"]
    change_type_dict = {"MatMul": (True, False),
                        "BatchMatMul": (True, False)}
    config["bool_storage_as_1bit"] = True
    config["enable_group_inplace"] = False
    config["enable_vector_2x"] = True
    for op in op_names:
        if op in bool_storage_as_1bit_oplist:
            config["bool_storage_as_1bit"] = False
        enable_group_inplace, enable_vector_2x = change_type_dict.get(op, (False, True))
        config["enable_group_inplace"] = config["enable_group_inplace"] or enable_group_inplace
        config["enable_vector_2x"] = config["enable_vector_2x"] and enable_vector_2x


def add_new_shape(names, shapes, new_shapes, inputs):
    """Add new shape for input tensors."""
    if not isinstance(names, list):
        names = [names]
        shapes = [shapes]
        new_shapes = [new_shapes]
    for i, name in enumerate(names):
        if shapes[i] == new_shapes[i]:
            continue
        if name not in inputs:
            raise RuntimeError("Can not support reshape on output tensor {}".format(name))
        if "new_shape" not in inputs[name]:
            inputs[name]["new_shape"] = new_shapes[i]
        elif new_shapes[i] != inputs[name]["new_shape"]:
            raise RuntimeError("Find different new_shape {} and {} for {}"
                               .format(inputs[name]["new_shape"], new_shapes[i], name))


class TransShape:
    """TransShape"""

    @classmethod
    def trans_elemwise_shape(cls, op_inputs, inputs):
        """deal with elemwise shape."""
        names = []
        shapes = []
        ori_shapes = []
        formats = []
        ori_formats = []
        for k, v in op_inputs.items():
            # scalar not included
            if v.get("value") is not None:
                continue
            names.append(k)
            shapes.append(v["shape"])
            ori_shapes.append(v["ori_shape"] if v.get("ori_shape") else None)
            formats.append(v["format"])
            ori_formats.append(v["ori_format"])
        if len(shapes) == 2 and len(shapes[0]) != len(shapes[1]):
            from impl.add import _add_check_format, _infer_shape
            format_pattern = _add_check_format({"shape": shapes[0], "format": formats[0]},
                                               {"shape": shapes[1], "format": formats[1]})

            ori_shape0 = ori_shapes[0]
            if ori_shape0 is None:
                ori_shape0 = infer_ori_shape(shapes[0], formats[0], ori_formats[0])
            ori_shape1 = ori_shapes[1]
            if ori_shape1 is None:
                ori_shape1 = infer_ori_shape(shapes[1], formats[1], ori_formats[1])
            new_shapes = [None, None]
            new_shapes[0], new_shapes[1] = _infer_shape(format_pattern,
                                                        {"shape": shapes[0], "ori_shape": ori_shape0},
                                                        {"shape": shapes[1], "ori_shape": ori_shape1})
            new_shapes[0], new_shapes[1], _ = shape_util.broadcast_shapes(new_shapes[0], new_shapes[1],
                                                                          param_name_input1="input0",
                                                                          param_name_input2="input1")
            add_new_shape(names, shapes, new_shapes, inputs)

    @classmethod
    def trans_batch_matmul_shape(cls, op_inputs, inputs):
        """deal with batch_matmul."""
        for k, v in op_inputs.items():
            # batch dimension of BatchMatMul must be fused to 1D
            shape = v["shape"]
            if len(shape) > 5:
                new_shape = [functools.reduce(lambda x, y: x * y, shape[:-4])] + shape[-4:]
                add_new_shape(k, shape, new_shape, inputs)

    @classmethod
    def run(cls, op_name, pattern, op_inputs, inputs):
        """entry function."""
        if pattern == OpPattern.ELEMWISE:
            TransShape.trans_elemwise_shape(op_inputs, inputs)
        elif op_name == "BatchMatMul":
            TransShape.trans_batch_matmul_shape(op_inputs, inputs)


def infer_ori_shape(shape, cur_format, ori_format):
    """Given current format and shape, infer the shape with ori_format."""
    if cur_format == ori_format:
        return shape
    default_formats = ["DefaultFormat", "ND", "NCHW"]

    if cur_format in default_formats and ori_format in default_formats:
        return shape

    if cur_format == "FRACTAL_NZ" and ori_format in default_formats:
        dims = len(shape)
        if dims < 4:
            raise ValueError("Invalid shape {} for format {}".format(shape, cur_format))
        ori_shape = shape[:dims - 4]
        m = shape[-3] * shape[-2]
        n = shape[-4] * shape[-1]
        ori_shape.append(m)
        ori_shape.append(n)
        return ori_shape

    if cur_format == "NC1HWC0" and ori_format in default_formats:
        if len(shape) != 5:
            raise ValueError("Invalid shape {} for format {}".format(shape, cur_format))
        ori_shape = [shape[0], shape[1] * shape[4], shape[2], shape[3]]
        return ori_shape

    if cur_format == "NHWC" and ori_format in default_formats:
        if len(shape) != 4:
            raise ValueError("Invalid shape {} for format {}".format(shape, cur_format))
        ori_shape = [shape[0], shape[3], shape[1], shape[2]]
        return ori_shape

    raise ValueError("Not support infer shape from format {} to {}".format(cur_format, ori_format))


def get_inputs_name(input_desc):
    """Get input tensor names."""
    inputs_name = []
    for desc in input_desc:
        for item in desc:
            inputs_name.append(item["tensor_name"])
    return inputs_name


def get_outputs_name(output_desc):
    """Get output tensor names."""
    outputs_name = []
    for item in output_desc:
        outputs_name.append(item["tensor_name"])
    return outputs_name


def get_all_op_name(op_desc):
    """Get op names."""
    op_names = []
    for op in op_desc:
        op_names.append(op["name"])
    return op_names


def get_input_desc(input_desc):
    """Get input desc."""
    res = {}
    for desc in input_desc:
        for item in desc:
            item["shape"] = [1] if not item["shape"] else item["shape"]
            res[item["tensor_name"]] = item
    return res


def get_inputs_tensor(input_desc, all_tensors):
    """Get input placeholders."""
    inputs = []
    for desc in input_desc:
        for item in desc:
            name = item["tensor_name"]
            if item.get("value") is not None:
                # const value
                all_tensors[name] = tvm.const(item["value"], item["data_type"])
            if all_tensors.get(name) is None:
                raise ValueError("Tensor [{}] not found.".format(name))
            inputs.append(all_tensors[name])
    return inputs


def get_op_attrs(op, fusion_op_name):
    """Get op attrs."""

    def _get_attrs(attr_desc):
        attrs = {}
        if not isinstance(attr_desc, list):
            return attrs
        for attr in attr_desc:
            attrs[attr["name"]] = attr["value"]
        return attrs

    op_name = op["name"]
    op_attrs = _get_attrs(op.get("attr"))
    if op_name == "BatchMatMul":
        op_attrs["dst_type"] = op["output_desc"][0]["data_type"]
        op_attrs["dst_ori_shape"] = op["output_desc"][0].get("ori_shape")
        if op_attrs.get("dst_ori_shape") is None:
            op_attrs["dst_ori_shape"] = infer_ori_shape(op["output_desc"][0]["shape"],
                                                        op["output_desc"][0]["format"],
                                                        op["output_desc"][0]["ori_format"])
    elif op_name == "MatMul":
        op_attrs["dst_type"] = op["output_desc"][0]["data_type"]
        op_attrs["dst_format"] = op["output_desc"][0]["format"]
    elif op_name == "Cast":
        op_attrs["dst_type"] = op["output_desc"][0]["data_type"]
    op_attrs["fusion_op_name"] = fusion_op_name
    return op_attrs


def create_placeholders(inputs):
    """Create placeholders."""
    tensors = {}
    for k, v in inputs.items():
        dtype = v["data_type"]
        if dtype == "bool":
            dtype = "int8"
        shape = v["shape"]
        if "new_shape" in v:
            shape = v["new_shape"]
            print("[{}] shape: {} --> new shape: {}".format(k, v["shape"], v["new_shape"]))
        attr = {
            "format": v.get("format"),
            "sub_format": v.get("sub_format", ""),
            "ori_shape": v.get("ori_shape"),
            "ori_format": v.get("ori_format"),
            "addr_type": v.get("addr_type", 0),
            "valid_shape": v.get("valid_shape", []),
            "slice_offset": v.get("slice_offset", []),
            "L1_fusion_type": v.get("L1_fusion_type", -1),
            "L1_addr_flag": v.get("L1_addr_flag", -1),
            "L1_addr_offset": v.get("L1_addr_offset", -1),
            "L1_valid_size": v.get("L1_valid_size", -1),
            "range": v.get("range", [])
        }
        if attr.get("ori_shape") is None:
            attr["ori_shape"] = infer_ori_shape(v.get("shape"), v.get("format"), attr.get("ori_format"))
        tensors[k] = tvm.placeholder(shape=shape, name=k, dtype=dtype, attrs=attr)
    return tensors


def same_shape(inputs):
    """Check if all inputs have same shape."""
    if not inputs:
        return True
    base_shape = -1
    for _, v in inputs.items():
        if base_shape == -1:
            base_shape = v["shape"]
        if v["shape"] != base_shape:
            return False
    return True


def create_input_tensors(json_dict):
    """Create input placeholders."""
    fold_dim = True
    inputs = get_input_desc(json_dict.get("input_desc", []))
    for op in json_dict["op_desc"]:
        op_name = op["name"]
        pattern = get_op_reg_info(op_name, "pattern")
        op_inputs = get_input_desc(op.get("input_desc", []))
        TransShape.run(op_name, pattern, op_inputs, inputs)
        if pattern != OpPattern.ELEMWISE or not same_shape(op_inputs):
            fold_dim = False
    if fold_dim:
        for k, v in inputs.items():
            shape = v["shape"]
            new_shape = [functools.reduce(lambda x, y: x * y, shape[:])]
            add_new_shape(k, shape, new_shape, inputs)
    return create_placeholders(inputs)


def create_fusion_op_name(op_names):
    """Create fusion op name, which is used in same op compute process."""
    fusion_op_name = "te_fusion" if len(op_names) > 1 else ""
    for op_name in op_names:
        kernel_name = get_op_reg_info(op_name, "kernel_name")
        fusion_op_name = "{}_{}".format(fusion_op_name, kernel_name)
    return fusion_op_name


def update_format(json_dict):
    """Some format like DefaultFormat is not recognized in TBE, need to covert these formats."""

    def _update_input_format(input_desc):
        for desc in input_desc:
            for item in desc:
                if item["format"] == "DefaultFormat":
                    item["format"] = "ND"
                if item.get("ori_format") is None or item["ori_format"] == "DefaultFormat":
                    item["ori_format"] = "NCHW"

    def _update_output_format(output_desc):
        for item in output_desc:
            if item["format"] == "DefaultFormat":
                item["format"] = "ND"
            if item.get("ori_format") is None or item["ori_format"] == "DefaultFormat":
                item["ori_format"] = "NCHW"

    _update_input_format(json_dict.get("input_desc", []))
    _update_output_format(json_dict["output_desc"])
    for op in json_dict["op_desc"]:
        _update_input_format(op.get("input_desc", []))
        _update_output_format(op["output_desc"])


def build(json_str):
    """Build kernel."""
    json_dict = json.loads(json_str)
    update_format(json_dict)
    inputs_name = get_inputs_name(json_dict.get("input_desc", []))
    outputs_name = get_outputs_name(json_dict["output_desc"])
    op_names = get_all_op_name(json_dict["op_desc"])
    fusion_op_name = create_fusion_op_name(op_names)

    # Create input placeholder
    all_tensors = create_input_tensors(json_dict)

    with op_context.OpContext("pre_static"):
        context = op_context.get_context()
        context.add_addition("op_bank_path", "")
        context.add_addition("mdl_bank_path", "")

        # Emit op
        for op in json_dict["op_desc"]:
            op_name = op["name"]
            # get op input tensor
            op_inputs = get_inputs_tensor(op.get("input_desc", []), all_tensors)
            # get op attrs
            op_attrs = get_op_attrs(op, fusion_op_name)
            # op compute
            op_outputs = get_op_reg_info(op_name, "func")(*op_inputs, attrs=op_attrs)
            # update op output tensor
            if not isinstance(op_outputs, (list, tuple)):
                op_outputs = [op_outputs]
            if len(op["output_desc"]) != len(op_outputs):
                raise ValueError("len(op[\"output_desc\"] is not equal to the number of real output tensors in op[{}]: "
                                 "{} vs {}".format(op_name, len(op["output_desc"]), len(op_outputs)))
            for i, desc in enumerate(op["output_desc"]):
                all_tensors[desc["tensor_name"]] = op_outputs[i]

        # Collect input, output tensors
        io_tensors = []
        output_tensors = []
        for name in inputs_name:
            io_tensors.append(all_tensors.get(name))
        for name in outputs_name:
            output_tensors.append(all_tensors.get(name))
            io_tensors.append(all_tensors.get(name))

        # Schedule and build
        with tvm.target.cce():
            sch = auto_schedule(output_tensors)
        config = {"name": json_dict["op"],
                  "tensor_list": io_tensors}
        update_config(config, op_names)
        tbe_build(sch, config)


def build_tbe_kernel(json_str, kernel_meta_parent_dir):
    """Build TBE kernel."""
    initialize(kernel_meta_parent_dir)
    with build_config(kernel_meta_parent_dir=kernel_meta_parent_dir):
        build(json_str)
