# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
"""
Parse Update Weights Name.
"""
import re
import os


def _rename_variable_weight(name):
    """Rename variable weight"""
    if not name.endswith("weight"):
        raise RuntimeError("variable is not norm name, now only support **weight")
    if "up_blocks" not in name and "down_blocks" not in name and "mid_block" not in name:
        raise RuntimeError("variable is not norm name, must include one of up_blocks, up_blocks or mid_block")
    if "attentions" not in name or "transformer_blocks" not in name or "attn" not in name:
        raise RuntimeError("variable is not norm name, must include attentions, transformer_blocks or attn")
    name = name.replace("out_0", "out").replace("out.0", "out")
    nums = re.findall(r"\d+", name)
    if len(nums) < 3 or len(nums) > 4:
        raise RuntimeError("only support norm tensor name")
    new_name = ""
    if "down_blocks" in name:
        new_name = "/down_blocks." + nums[0]
    elif "mid_block" in name:
        new_name = "/mid_block"
    elif "up_blocks" in name:
        new_name = "/up_blocks." + nums[0]
    new_name += "/attentions." + nums[-3] + "/transformer_blocks." + nums[-2] + "/attn" + nums[-1]
    if "to_q" in name:
        new_name += "/to_q/MatMul"
    elif "to_v" in name:
        new_name += "/to_v/MatMul"
    elif "to_k" in name:
        new_name += "/to_k/MatMul"
    elif "to_out" in name:
        new_name += "/to_out.0/MatMul"
    return new_name


def _get_variable_weights_name(name_list_file):
    """Get variable weights name"""
    if not os.path.exists(name_list_file):
        raise RuntimeError("variable weight name list is not exists.")
    new_name_str = ""
    new_names = []
    with open(name_list_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line[-1] == "\n":
                line = line[:-1]
            new_name = _rename_variable_weight(line)
            if new_name not in new_names:
                new_names.append(new_name)
                new_name_str += ',' + new_name
    return new_name_str[1:]


def _parse_update_weight_config_name(name_list_file):
    """Parse update weight config name"""
    with open(name_list_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line[-1] == '\n':
                line = line[:-1]
            if "variable_weights_file" in line:
                name_list_file = line.split('=')[1]
                return _get_variable_weights_name(name_list_file)
    return None
