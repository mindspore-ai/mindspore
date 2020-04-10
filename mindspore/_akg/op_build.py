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

"""op_build"""
import os
import fcntl
import types
import typing
import logging
import traceback
import _akg.tvm
import _akg
from _akg import save_gpu_param as gpu_utils
from _akg.utils import validation_check as vc_util

MS_CUDA_KERNEL_PATH = "/tmp/cuda_meta/"

@vc_util.check_input_type(list, (list, tuple), (list, tuple), (types.FunctionType, type(None)), str, str, dict)
def op_build(opnames, computes, args, custom_schedule, device, kernel_name, attrs):
    """op_build"""
    if device == "cuda":
        cuda_path = os.path.realpath(MS_CUDA_KERNEL_PATH)
        if not os.path.isdir(cuda_path):
            os.makedirs(cuda_path)
        if not opnames:
            logging.error("no opname given.")
            return None

        schedule_name = 'gpu_schedule_' + opnames[0]
        schedule_func = getattr(_akg.gpu, schedule_name)
        if not isinstance(schedule_func, (types.FunctionType, typing.Callable)):
            logging.error("no schedule func found %s", str(schedule_name))
            return None

        ptx_file = os.path.realpath(MS_CUDA_KERNEL_PATH + kernel_name + ".ptx")
        if os.path.exists(ptx_file):
            os.chmod(ptx_file, 0o600)
        try:
            with open(ptx_file, 'at') as file:
                fcntl.flock(file.fileno(), fcntl.LOCK_EX)
                file.seek(0, 2)
                if file.tell() == 0:
                    s = schedule_func(computes)
                    foo = _akg.tvm.build(s, args, device, name=kernel_name)
                    ptx_code = foo.imported_modules[0].get_source("ptx")
                    file.write(ptx_code)
                    json_file = os.path.realpath(MS_CUDA_KERNEL_PATH + kernel_name + ".json")
                    kernel_info = (ptx_code, json_file, kernel_name)
                    gpu_utils.save_gpu_params(s, args, kernel_info)
            os.chmod(ptx_file, 0o400)
        except Exception:
            logging.error(traceback.format_exc())
            return None
        return True

    logging.error("Not support device %s.", device)
    return None
