# Copyright 2019 Huawei Technologies Co., Ltd
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

"""save gpu param"""
import os
import hashlib
import akg.tvm
from akg.tvm import schedule
from akg.utils import validation_check as vc_util


def get_dim(dim, axis=True):
    """get dim info"""
    dims_str = {
        "grid_dim0": "// attr [iter_var(blockIdx.x, , blockIdx.x)] thread_extent = ",
        "grid_dim1": "// attr [iter_var(blockIdx.y, , blockIdx.y)] thread_extent = ",
        "grid_dim2": "// attr [iter_var(blockIdx.z, , blockIdx.z)] thread_extent = ",
        "block_dim0": "// attr [iter_var(threadIdx.x, , threadIdx.x)] thread_extent = ",
        "block_dim1": "// attr [iter_var(threadIdx.y, , threadIdx.y)] thread_extent = ",
        "block_dim2": "// attr [iter_var(threadIdx.z, , threadIdx.z)] thread_extent = "
    }
    dim_to_axis = {
        "grid_dim0": '"blockIdx.x" : ',
        "grid_dim1": '"blockIdx.y" : ',
        "grid_dim2": '"blockIdx.z" : ',
        "block_dim0": '"threadIdx.x" : ',
        "block_dim1": '"threadIdx.y" : ',
        "block_dim2": '"threadIdx.z" : '
    }
    if axis:
        return dim_to_axis.get(dim)
    return dims_str.get(dim)


def parse_params(file, dim, ir):
    """parse parameters"""
    dim_str = get_dim(dim, axis=False)
    pos = ir.find(dim_str)
    if pos != -1:
        index = pos + len(dim_str)
        param_temp = get_dim(dim)

        while ir[index].isdigit():
            param_temp += ir[index]
            index += 1
        file.write(param_temp + ",\n")
    else:
        param_temp = get_dim(dim) + '1'
        file.write(param_temp + ",\n")


@vc_util.check_input_type(schedule.Schedule, (list, tuple), tuple)
def save_gpu_params(s, args, kernel_info):
    """save gpu parameters"""
    ptx_code = kernel_info[0]
    file_name = kernel_info[1]
    kernel_name = kernel_info[2]
    ir = str(akg.tvm.lower(s, args, simple_mode=True))
    file_path = os.path.realpath(file_name)
    if os.path.exists(file_path):
        os.remove(file_path)

    sha256 = hashlib.sha256()
    sha256.update(ptx_code.encode("utf-8"))
    hash_str = sha256.hexdigest()
    with os.fdopen(os.open(file_path, os.O_WRONLY | os.O_CREAT, 0o400), 'w') as fo:
        fo.write("{\n")
        fo.write('"kernelName" : ' + '"' + kernel_name + "_kernel0" + '",\n')
        parse_params(fo, "grid_dim0", ir)
        parse_params(fo, "grid_dim1", ir)
        parse_params(fo, "grid_dim2", ir)
        parse_params(fo, "block_dim0", ir)
        parse_params(fo, "block_dim1", ir)
        parse_params(fo, "block_dim2", ir)
        fo.write('"sha256" : ' + '"' + hash_str + '"\n')
        fo.write("}\n")
