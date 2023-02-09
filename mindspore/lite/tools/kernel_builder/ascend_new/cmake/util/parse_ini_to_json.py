# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

"""
parser ini to json
"""

import json
import os
import stat
import sys


ATTR_TYPE_LIST = ["int", "float", "bool", "str", "listInt", "listFloat", "listBool", "listStr", "listListInt",
                  "type", "listType", "tensor", "listTensor"]
ATTR_PARAMTYPE_LIST = ["optional", "required"]
BOOL_FLAG_KEY = ["dynamicFormat", "dynamicShapeSupport", "dynamicRankSupport", "precision_reduce", "heavyOp",
                 "needCheckSupport"]
BOOL_LIST = ["true", "false"]
DTYPE_LIST = ["float16", "float", "float32", "int8", "int16", "int32", "uint8", "uint16", "uint32", "bool",
              "int64", "uint64", "qint8", "qint16", "qint32", "quint8", "quint16", "double", "complex64",
              "complex128", "string", "resource"]
FORMAT_LIST = ["NCHW", "NHWC", "ND", "NC1HWC0", "FRACTAL_Z", "NC1C0HWPAD", "NHWC1C0", "FSR_NCHW", "FRACTAL_DECONV",
               "C1HWNC0", "FRACTAL_DECONV_TRANSPOSE", "FRACTAL_DECONV_SP_STRIDE_TRANS", "NC1HWC0_C04",
               "FRACTAL_Z_C04", "CHWN", "FRACTAL_DECONV_SP_STRIDE8_TRANS", "HWCN", "NC1KHKWHWC0", "BN_WEIGHT",
               "FILTER_HWCK", "HASHTABLE_LOOKUP_LOOKUPS", "HASHTABLE_LOOKUP_KEYS", "HASHTABLE_LOOKUP_VALUE",
               "HASHTABLE_LOOKUP_OUTPUT", "HASHTABLE_LOOKUP_HITS", "C1HWNCoC0", "MD", "NDHWC", "FRACTAL_ZZ",
               "FRACTAL_NZ", "NCDHW", "DHWCN", "NDC1HWC0", "FRACTAL_Z_3D", "CN", "NC", "DHWNC",
               "FRACTAL_Z_3D_TRANSPOSE", "FRACTAL_ZN_LSTM", "FRACTAL_ZN_RNN", "FRACTAL_Z_G", "NULL"]


def parse_ini_files(ini_files):
    """
    parse ini files to json
    Parameters:
    ----------------
    ini_files:input file list
    return:ops_info
    ----------------
    """
    tbe_ops_info = {}
    for ini_file in ini_files:
        check_file_size(ini_file)
        parse_ini_to_obj(ini_file, tbe_ops_info)
    return tbe_ops_info


def check_file_size(input_file):
    try:
        file_size = os.path.getsize(input_file)
    except OSError as os_error:
        print('[ERROR] Failed to open "%s". %s' % (input_file, str(os_error)))
        raise OSError from os_error
    if file_size > 10*1024*1024:
        print('[WARN] The size of %s exceeds 10MB, it may take more time to run, please wait.' % input_file)


def parse_ini_to_obj(ini_file_path, tbe_ops_info):
    """
    parse ini file to json obj
    Parameters:
    ----------------
    ini_file_path:ini file path
    tbe_ops_info:ops_info
    ----------------
    """
    with open(ini_file_path) as ini_file:
        lines = ini_file.readlines()
        op_dict = {}
        op_name = ""
        find_op_type = False
        for line in lines:
            line = line.rstrip()
            if line == "":
                continue
            if line.startswith("["):
                if line.endswith("]"):
                    op_name = line[1:-1]
                    op_dict = {}
                    tbe_ops_info[op_name] = op_dict
                    find_op_type = True
            elif "=" in line:
                key1 = line[:line.index("=")]
                key2 = line[line.index("=")+1:]
                key1_0, key1_1 = key1.split(".")
                if not key1_0 in op_dict:
                    op_dict[key1_0] = {}
                if key1_1 in op_dict.get(key1_0):
                    raise RuntimeError("Op:" + op_name + " " + key1_0 + " " +
                                       key1_1 + " is repeated!")
                dic_key = op_dict.get(key1_0)
                dic_key[key1_1] = key2
            else:
                continue
        if not find_op_type:
            raise RuntimeError("Not find OpType in .ini file.")


def check_output_exist(op_dict, is_valid):
    """
    Function Description:
        Check output is exist
    Parameter: op_dict
    Parameter: is_valid
    """
    if "output0" in op_dict:
        output0_dict = op_dict.get("output0")
        if output0_dict.get("name", None) is None:
            is_valid = False
            print("output0.name is required in .ini file!")
    else:
        is_valid = False
        print("output0 is required in .ini file!")
    return is_valid


def check_attr_dict(attr_dict, is_valid, attr):
    """
    Function Description:
        Check attr_dict
    Parameter: attr_dict
    Parameter: is_valid
    Parameter: attr
    """
    attr_type = attr_dict.get("type")
    value = attr_dict.get("value")
    param_type = attr_dict.get("paramType")
    if attr_type is None or value is None:
        is_valid = False
        print(
            "If attr.list is exist, {0}.type and {0}.value is required".format(attr))
    if param_type and param_type not in ATTR_PARAMTYPE_LIST:
        is_valid = False
        print("{0}.paramType only support {1}.".format(
            attr, ATTR_PARAMTYPE_LIST))
    if attr_type and attr_type not in ATTR_TYPE_LIST:
        is_valid = False
        print("{0}.type only support {1}.".format(attr, ATTR_TYPE_LIST))
    return is_valid


def check_attr(op_dict, is_valid):
    """
    Function Description:
        Check attr
    Parameter: op_dict
    Parameter: is_valid
    """
    if "attr" in op_dict:
        attr_dict = op_dict.get("attr")
        attr_list_str = attr_dict.get("list", None)
        if attr_list_str is None:
            is_valid = False
            print("attr.list is required in .ini file!")
        else:
            attr_list = attr_list_str.split(",")
            for attr_name in attr_list:
                attr = "attr_" + attr_name.strip()
                attr_dict = op_dict.get(attr)
                if attr_dict:
                    is_valid = check_attr_dict(attr_dict, is_valid, attr)
                else:
                    is_valid = False
                    print("%s is required in .ini file, when attr.list is %s!" % (
                        attr, attr_list_str))
    return is_valid


def check_bool_flag(op_dict, is_valid):
    """
    Function Description:
        check_bool_flag
    Parameter: op_dict
    Parameter: is_valid
    """
    for key in BOOL_FLAG_KEY:
        if key in op_dict:
            op_bool_key = op_dict.get(key)
            if op_bool_key.get("flag").strip() not in BOOL_LIST:
                is_valid = False
                print("{0}.flag only support {1}.".format(key, BOOL_LIST))
    return is_valid


def check_type_format(op_info, is_valid, op_info_key):
    """
    Function Description:
        Check type and format
    Parameter: op_info
    Parameter: is_valid
    Parameter: op_info_key
    """
    op_info_dtype_str = op_info.get("dtype")
    op_info_dtype_num = 0
    op_info_format_num = 0
    if op_info_dtype_str:
        op_info_dtype = op_info_dtype_str.split(",")
        op_info_dtype_num = len(op_info_dtype)
        for dtype in op_info_dtype:
            if dtype.strip() not in DTYPE_LIST:
                is_valid = False
                print("{0}.dtype not support {1}.".format(op_info_key, dtype))
    op_info_format_str = op_info.get("format")
    if op_info_format_str:
        op_info_format = op_info_format_str.split(",")
        op_info_format_num = len(op_info_format)
        for op_format in op_info_format:
            if op_format.strip() not in FORMAT_LIST:
                is_valid = False
                print("{0}.format not support {1}.".format(
                    op_info_key, op_format))
    if op_info_dtype_num > 0 and op_info_format_num > 0:
        if op_info_dtype_num != op_info_format_num:
            is_valid = False
            print("The number of {0}.dtype not match the number of {0}.format.".format(
                op_info_key))
    return is_valid


def check_op_info(tbe_ops):
    """
    Function Description:
        Check info.
    Parameter: tbe_ops
    Return Value: is_valid
    """
    print("\n\n==============check valid for ops info start==============")
    required_op_input_info_keys = ["paramType", "name"]
    required_op_output_info_keys = ["paramType", "name"]
    param_type_valid_value = ["dynamic", "optional", "required"]
    is_valid = True
    for op_key in tbe_ops:
        op_dict = tbe_ops[op_key]
        is_valid = check_output_exist(op_dict, is_valid)
        for op_info_key in op_dict:
            if op_info_key.startswith("input"):
                op_input_info = op_dict[op_info_key]
                missing_keys = []
                for required_op_input_info_key in required_op_input_info_keys:
                    if not required_op_input_info_key in op_input_info:
                        missing_keys.append(required_op_input_info_key)
                if missing_keys:
                    print("op: " + op_key + " " + op_info_key + " missing: " +
                          ",".join(missing_keys))
                    is_valid = False
                else:
                    if not op_input_info["paramType"] in param_type_valid_value:
                        print("op: " + op_key + " " + op_info_key +
                              " paramType not valid, valid key:[dynamic, "
                              "optional, required]")
                        is_valid = False
                is_valid = check_type_format(
                    op_input_info, is_valid, op_info_key)
            if op_info_key.startswith("output"):
                op_input_info = op_dict[op_info_key]
                missing_keys = []
                for required_op_input_info_key in required_op_output_info_keys:
                    if not required_op_input_info_key in op_input_info:
                        missing_keys.append(required_op_input_info_key)
                if missing_keys:
                    print("op: " + op_key + " " + op_info_key + " missing: " +
                          ",".join(missing_keys))
                    is_valid = False
                else:
                    if not op_input_info["paramType"] in param_type_valid_value:
                        print("op: " + op_key + " " + op_info_key +
                              " paramType not valid, valid key:[dynamic, "
                              "optional, required]")
                        is_valid = False
                is_valid = check_type_format(
                    op_input_info, is_valid, op_info_key)
        is_valid = check_attr(op_dict, is_valid)
        is_valid = check_bool_flag(op_dict, is_valid)
    print("==============check valid for ops info end================\n\n")
    return is_valid


def write_json_file(tbe_ops_info, json_file_path):
    """
    Save info to json file
    Parameters:
    ----------------
    tbe_ops_info: ops_info
    json_file_path: json file path
    ----------------
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    modes = stat.S_IWUSR | stat.S_IRUSR
    json_file_real_path = os.path.realpath(json_file_path)
    with os.fdopen(os.open(json_file_real_path, flags, modes), "w") as file_path:
        # Only the owner and group have rights
        os.chmod(json_file_real_path, stat.S_IWGRP + stat.S_IWUSR + stat.S_IRGRP
                 + stat.S_IRUSR)
        json.dump(tbe_ops_info, file_path, sort_keys=True, indent=4,
                  separators=(',', ':'))
    print("Compile op info cfg successfully.")


def parse_ini_to_json(ini_file_paths, outfile_path):
    """
    parse ini files to json file
    Parameters:
    ----------------
    ini_file_paths: list of ini file path
    outfile_path: output file path
    ----------------
    """
    tbe_ops_info = parse_ini_files(ini_file_paths)
    if not check_op_info(tbe_ops_info):
        print("Compile op info cfg failed.")
        return False
    write_json_file(tbe_ops_info, outfile_path)
    return True


if __name__ == '__main__':
    args = sys.argv

    output_file_path = "tbe_ops_info.json"
    ini_file_path_list = []

    for arg in args:
        if arg.endswith("ini"):
            ini_file_path_list.append(arg)
            output_file_path = arg.replace(".ini", ".json")
        if arg.endswith("json"):
            output_file_path = arg

    if not ini_file_path_list:
        ini_file_path_list.append("tbe_ops_info.ini")

    if not parse_ini_to_json(ini_file_path_list, output_file_path):
        sys.exit(1)
    sys.exit(0)
