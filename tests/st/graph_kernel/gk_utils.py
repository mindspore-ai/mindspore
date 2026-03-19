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

"""graph kernel test case utility functions"""

import shutil
import os
import numpy as np
import mindspore
from mindspore import context
from mindspore import Tensor


class AssertGKEnable:
    def __init__(self, enable_graph_kernel=True):
        self.enable_gk = enable_graph_kernel
        self.ir_path = ""

    @staticmethod
    def _rm_dir(dir_path):
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path, ignore_errors=True)

    def __enter__(self):
        if self.enable_gk:
            self.ir_path = os.path.join("./irs_{}".format(os.getpid()))
            context.set_context(save_graphs=True, save_graphs_path=self.ir_path)

    def __exit__(self, *args):
        if self.enable_gk:
            context.set_context(save_graphs=False)
            graph_kernel_ir_dir = os.path.join(self.ir_path, "verbose_ir_files/graph_kernel")
            if not os.path.isdir(graph_kernel_ir_dir) or not os.listdir(graph_kernel_ir_dir):
                self._rm_dir(self.ir_path)
                raise RuntimeError("Graph Kernel Fusion is not enabled")
            self._rm_dir(self.ir_path)


def gen_flag(*args):
    lst = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            s = "_".join(str(a) for a in arg)
        else:
            s = str(arg)
        lst.append(s)
    flag = "_".join(lst)
    return flag


def trans_data_type(data_type):
    type_map = {
        "float32": [np.float32, mindspore.float32],
        "float16": [np.float16, mindspore.float16],
        "bfloat16": [np.float32, mindspore.bfloat16],
        "int32": [np.int32, mindspore.int32],
        "int8": [np.int8, mindspore.int8],
        "bool": [np.bool_, mindspore.bool_],
    }
    return type_map[data_type]


def gen_input(shape, data_type, is_positive=False, data_range=1.0):
    np_type, ms_type = trans_data_type(data_type)
    if data_type.startswith("int"):
        x = np.random.randint(-100, 100, size=shape).astype(np_type)
        if is_positive:
            x = np.abs(x) + 1
    elif data_type == "bool":
        if is_positive:
            x = np.full(shape, True, dtype=np_type)
        else:
            x = np.random.randint(2, size=shape).astype(np_type)
    else:
        x = np.random.normal(0, data_range, shape).astype(np_type)
        if is_positive:
            x = np.abs(x) + 0.01
    x_ms = Tensor(x, ms_type)
    return x_ms


def get_func_name(func):
    func_name = func.__name__ if hasattr(func, "__name__") else type(func).__name__
    return func_name


def compare_outputs(flag, outputs, cmp_precision=0.0):
    enable_dvm = os.environ.get("MS_DEV_PYNATIVE_FUSION_FLAGS", "").find("--opt_level=1") != -1
    output_dir = os.environ.get("DVM_OUTPUT_DIR", "./")
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    else:
        outputs = list(outputs)
    for i, output in enumerate(outputs):
        if output is None:
            continue
        output = output.float().asnumpy() if output.dtype == mindspore.bfloat16 else output.asnumpy()
        output_name = "{}/{}_output_{}.npy".format(output_dir, flag, i)
        if enable_dvm:
            expect = np.load(output_name)
            if output.dtype != expect.dtype:
                raise RuntimeError("{} outputs[{}]: {} vs {}".format(flag, i, expect.dtype, output.dtype))
            if output.shape != expect.shape:
                raise RuntimeError("{} outputs[{}]: {} vs {}".format(flag, i, expect.shape, output.shape))
            if not np.allclose(expect, output, cmp_precision, cmp_precision, equal_nan=True):
                print("{} outputs[{}]: precision error! cmp_precision: {}".format(flag, i, cmp_precision))
                np.testing.assert_allclose(expect, output, cmp_precision, cmp_precision, equal_nan=True)
        else:
            np.save(output_name, output)
