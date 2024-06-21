# Copyright 2024 Huawei Technologies Co., Ltd
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
"""utils for custom op"""
import os
import mindspore as ms
import shutil
import glob
import subprocess


def compile_custom_run(workspace_dir):
    if "PATH" in os.environ:
        current_value = os.environ["PATH"]
        new_value = current_value + ':/usr/local/Ascend/latest/compiler/bin'

    else:
        new_value = '/usr/local/Ascend/latest/compiler/bin'
    os.environ["PATH"] = new_value
    ms_path = ms.__file__
    ms_dir_path, _ = os.path.split(ms_path)
    custom_compiler_path = os.path.join(ms_dir_path, "lib/plugin/ascend/custom_compiler")
    dst_compiler_path = os.path.join(workspace_dir, "custom_compiler")

    try:
        shutil.copytree(custom_compiler_path, dst_compiler_path)
    except FileExistsError:
        shutil.rmtree(dst_compiler_path)
        shutil.copytree(custom_compiler_path, dst_compiler_path)

    op_path, _ = os.path.split(__file__)
    op_host_path = os.path.join(op_path, 'op_host')
    op_kernel_path = os.path.join(op_path, 'op_kernel')
    command = [
        'cd ' + dst_compiler_path + '; python setup.py -o' + op_host_path + ' -k' + op_kernel_path]
    result = subprocess.run(command, shell=True, stderr=subprocess.STDOUT)
    run_files = glob.glob(workspace_dir + '/custom_compiler/build_out/*.run')
    assert result.returncode == 0
    assert len(run_files) == 1
