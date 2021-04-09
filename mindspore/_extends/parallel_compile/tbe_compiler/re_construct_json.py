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
# ============================================================================
"""re construct json"""
import json


def common_op_info(json_file):
    """
    Create more detail info
    :param json_file: origin json file
    :return: origin json file
    """
    json_file["L1_addr_offset"] = 0
    json_file["L1_fusion_type"] = -1
    json_file["L1_workspace_size"] = -1
    json_file["addr_type"] = 0
    json_file["slice_offset"] = []
    json_file["split_index"] = 0
    json_file["total_shape"] = []
    json_file["valid_shape"] = []
    return json_file


def create_input(json_info):
    """
    Create input, type is "Data"
    :param json_info: json file
    :return: ops list
    """
    ops = []
    if "inputs" in json_info and json_info["inputs"] is not None:
        ori_inputs = json_info["inputs"]
        for _, item in enumerate(ori_inputs):
            op_info = {
                "name": item[0]["name"],
                "output_desc": [common_op_info(item[0])],
                "type": "Data"
            }
            ops.append(op_info)
    return ops


def create_inout_desc(ori_json):
    """
    Create input or output, insert "data_type" attr and other detail infos
    :param ori_json: input or output list, the item in list is a dict
    :return: list
    """
    if ori_json is None:
        return "null"
    out_list = []
    for _, item in enumerate(ori_json):
        item[0]["data_type"] = item[0]["dtype"] if "dtype" in item[0] else 0
        if "ori_format" in item[0] or "ori_shape" in item[0]:
            item[0]["L1_addr_offset"] = 0
            item[0]["L1_fusion_type"] = -1
            item[0]["L1_workspace_size"] = -1
            item[0]["addr_type"] = 0
            item[0]["slice_offset"] = []
            item[0]["split_index"] = 0
            item[0]["total_shape"] = []
            item[0]["valid_shape"] = []
        else:
            item[0]["shape"] = "NULL"
        out_list.append(item[0])
    return out_list


def create_pre_build_attr(ori_json):
    """
    Create prebuild_outs_attrs
    :param ori_json: origin json file
    :return: dict
    """
    args = [create_inout_desc(ori_json["outputs"])[0]]
    if "attrs" in ori_json and ori_json["attrs"] is not None:
        ori_attrs = ori_json["attrs"]
        for item in ori_attrs:
            if "value" in item:
                args.append(item["value"])
    pre_build_attr = {"kwds_args": {},
                      "list_args": args
                      }
    return pre_build_attr


def create_compute_op(ori_json):
    """
    Create compute op's in and out desc
    :param ori_json: origin json file
    :return: dict
    """
    func_name = ori_json["name"]
    op_type = ori_json["Type"]
    full_name = ori_json["full_name"]
    pattern = ori_json["pattern"] if "pattern" in ori_json else ""
    op_common_info = {
        "func_name": func_name,
        "input_desc": create_inout_desc(ori_json["inputs"]) if "inputs" in ori_json else "null",
        "module_name": ori_json["module_name"],
        "name": full_name,
        "ori_name": [full_name],
        "output_desc": create_inout_desc(ori_json["outputs"]) if "outputs" in ori_json else "null",
        "output_data_desc": create_inout_desc(ori_json["outputs"]) if "outputs" in ori_json else "null",
        "pattern": pattern,
        "attr_desc": ori_json["attr_desc"] if "attr_desc" in ori_json else "null",
        "py_module_path": ori_json["py_module_path"],
        "type": op_type
    }
    return op_common_info


def single_to_fusion(json_file, tune_mode):
    """
    Change single op json to fusion op json for auto tune
    :param json_file: origin json file
    :param tune_mode: tune mode
    :return: a fusion op json, which contain one op
    """
    ori_file = json.loads(json_file)
    json_info = ori_file["op_info"]
    soc_info = ori_file["SocInfo"]
    soc_info["autoTilingMode"] = tune_mode
    kernel_name = json_info["kernel_name"]
    ops = create_input(json_info)
    ops2 = create_compute_op(json_info)
    ops.append(ops2)
    end_file = {
        "SocInfo": soc_info,
        "fusion_op_name": kernel_name,
        "l1_size": -1,
        "op_list": ops
    }
    res = json.dumps(end_file, ensure_ascii=False)
    return res


def add_ori_name_to_fusion(json_info):
    """Add ori_name to fusion json"""
    full_name = json_info["fusion_op"]["full_name"]
    ops = json_info["fusion_op"]["op_list"]
    for op in ops:
        if op["type"] != "Data":
            op["ori_name"] = [full_name]


def fusion_to_fusion(json_str, tune_mode):
    """
    Add l1_size for fusion json
    :param json_str: origin json file
    :param tune_mode: tune mode
    :return: fusion json info
    """

    json_info = json.loads(json_str)
    json_info["fusion_op"]["l1_size"] = -1
    json_info["SocInfo"]["autoTilingMode"] = tune_mode
    add_ori_name_to_fusion(json_info)
    end_file = json_info["fusion_op"]
    end_file["SocInfo"] = json_info["SocInfo"]
    res = json.dumps(end_file, ensure_ascii=False)
    return res
