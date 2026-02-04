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
"""File Tools."""
import os
import shutil
from mindspore import log as logger


def clean_all_ckpt_files(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.ckpt') or file_name.endswith('.meta'):
                try:
                    os.remove(os.path.join(folder_path, file_name))
                except FileNotFoundError as e:
                    logger.warning("[{}] remove ckpt file error.".format(e))


def find_newest_ckpt_file_by_name(folder_path, format="ckpt"): # pylint: disable=redefined-builtin
    ckpt_files = map(lambda f: os.path.join(folder_path, f),
                     filter(lambda f: f.endswith(f'.{format}'),
                            os.listdir(folder_path)))
    return max(list(ckpt_files))


def find_newest_ckpt_file(folder_path):
    ckpt_files = map(lambda f: os.path.join(folder_path, f),
                     filter(lambda f: f.endswith('.ckpt'),
                            os.listdir(folder_path)))
    return max(ckpt_files, key=os.path.getctime)


def find_ckpt_file(file_path):
    checkpoint_file_list = []
    if os.path.exists(file_path):
        ls = os.listdir(file_path)
        for line in ls:
            if line.endswith(".ckpt"):
                file_name = line.split(".ckpt")[0]
                checkpoint_file_list.append(file_name)
    return checkpoint_file_list


def clean_directory(case_name):
    cur_dir = os.getcwd()
    dir_name = os.path.join(cur_dir, case_name)
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)


def find_file_list(base_path, prefix_name, subfix="ckpt"):
    checkpoint_file_list = []
    if os.path.exists(base_path):
        ls = os.listdir(base_path)
        logger.info("ls : {}".format(ls))
        for line in ls:
            if line.startswith(prefix_name) and line.endswith(subfix):
                file_path = os.path.join(base_path, line)
                checkpoint_file_list.append(file_path)
    return checkpoint_file_list
