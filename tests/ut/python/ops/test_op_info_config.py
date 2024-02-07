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
# ============================================================================
""" test_op_info_config """
import os
import hashlib
import mindspore as ms
from mindspore import log


def get_file_hash(file_path):
    with open(file_path, "rb") as f:
        content = f.read()
    f.close()
    hash_object = hashlib.md5(content)
    hash_value = hash_object.hexdigest()
    return hash_value


def test_op_info_config():
    """
    Feature: file of op_info.config
    Description: Ensure op_info.config file is consistent with the Registration file
    Expectation: Success.
    """
    file_path, _ = os.path.split(ms.__file__)
    aicpu_register_info_path = os.path.join(file_path, "ops/_op_impl/aicpu")
    cpu_register_info_path = os.path.join(file_path, "ops/_op_impl/cpu")
    hash_list = ""

    for file in sorted(os.listdir(cpu_register_info_path)):
        file_path = os.path.join(cpu_register_info_path, file)
        if os.path.isdir(file_path):
            continue
        hash_list = hash_list + get_file_hash(file_path)

    for file in sorted(os.listdir(aicpu_register_info_path)):
        file_path = os.path.join(aicpu_register_info_path, file)
        if os.path.isdir(file_path):
            continue
        hash_list = hash_list + get_file_hash(file_path)

    hash_object = hashlib.md5(hash_list.encode('utf-8'))
    hash_value = hash_object.hexdigest()

    expect_value = "ab2fab2eb6b4d168bf837bca5888d09f"
    if hash_value != expect_value:
        log.error(
            "Hash value check failed! You have modified the registration file of AICPU and CPU, please check whether "
            "the file [op_info.config] is modified accordingly. After this, you can modify the test case by replacing "
            "the current expected hash value [{}] with the new hash value [{}].".format(expect_value, hash_value))
    assert hash_value == expect_value
