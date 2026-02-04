# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Test utils for testing mindformers mcore DeepSeekv3 pretrain"""
import os
import subprocess
from tests.st.networks.llm_parallel_feature.utils import update_parallel_speed_up_json, clear_directory


class DeepseekConfig:
    """add default config for DeepSeek model."""
    def __init__(self,
                 hidden_size=1024,
                 intermediate_size=1024,
                 moe_intermediate_size=512,
                 pp_interleave_num=2,
                 first_k_dense_replace=1,
                 num_layer=4,
                 parallel_speed_up_json=None,
                 npu_nums_per_device=2,
                 deterministic="ON"
                 ):
        # context
        self.parallel_speed_up_json = parallel_speed_up_json
        self.deterministic = deterministic

        # training parameters
        self.pp_interleave_num = pp_interleave_num

        # model parameters
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.first_k_dense_replace = first_k_dense_replace

        # moe
        self.npu_nums_per_device = npu_nums_per_device
        self.moe_intermediate_size = moe_intermediate_size


def prepare_deepseekv3_testcase_env(testcase_name, net_config):
    """prepare deepseekv3 testcase environment and files"""
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    # 1. create testcase folder
    os.makedirs(os.path.join(sh_path, testcase_name), exist_ok=True)
    # 2. clear folder (if exist)
    clear_directory(f"{sh_path}/{testcase_name}")
    # 3. copy yaml to testcase folder
    os.system(f"cp {sh_path}/pretrain_deepseek3.yaml ./{testcase_name}")
    # 4. replace config in yaml
    file_path = f'{sh_path}/{testcase_name}/pretrain_deepseek3.yaml'
    status = replace_deepseekv3_config(net_config, file_path)
    # 5. update parallel_speed_up.json if needed
    if net_config.parallel_speed_up_json is not None:
        if not update_parallel_speed_up_json(testcase_name, net_config, file_path, deepseekv3=True):
            raise ValueError("Failed to update parallel_speed_up.json")
    if not status:
        raise Exception("Failed to replace config in {}".format(file_path))

    return file_path


def replace_deepseekv3_config(net_config, file_path):
    """replace deepseekv3 yaml by config in testcases"""
    old_list = [
        "num_layers 4",
        "hidden_size: 1024",
        "intermediate_size: 1024",
        "moe_intermediate_size: 512",
        "first_k_dense_replace: 1",
        "npu_nums_per_device: 2",
        "pp_interleave_num: 2",
        "deterministic: \"ON\"",
    ]

    new_list = [
        f'num_layers {net_config.num_layer}',
        f'hidden_size: {net_config.hidden_size}',
        f'intermediate_size: {net_config.intermediate_size}',
        f'moe_intermediate_size: {net_config.moe_intermediate_size}',
        f'first_k_dense_replace: {net_config.first_k_dense_replace}',
        f'npu_nums_per_device: {net_config.npu_nums_per_device}',
        f'pp_interleave_num: {net_config.pp_interleave_num}',
        f'deterministic: \"{net_config.deterministic}\"'
    ]

    if len(old_list) != len(new_list):
        print(f"Old list and new list have different lengths: {len(old_list)} and {len(new_list)}")
        return False
    for old, new in zip(old_list, new_list):
        if "'" in old:
            sed_cmd = f'''sed -i "s#{old}#{new}#g" {file_path}'''
        else:
            sed_cmd = f"sed -i 's#{old}#{new}#g' {file_path}"

        status, _ = subprocess.getstatusoutput(sed_cmd)
        if status != 0:
            print(f"Failed to replace {old} with {new} in {file_path}")
            return False

    return True
