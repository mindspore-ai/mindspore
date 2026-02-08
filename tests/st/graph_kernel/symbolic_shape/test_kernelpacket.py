# Copyright 2024-2025 Huawei Technologies Co., Ltd
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

import os
import copy
import subprocess
from tests.mark_utils import arg_mark


def run_with_log(test_case, log_file):
    env = copy.deepcopy(os.environ)
    env['ASCEND_GLOBAL_EVENT_ENABLE'] = '1'
    env['ASCEND_GLOBAL_LOG_LEVEL'] = '1'
    env['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
    env['GLOG_v'] = '1'
    env['MS_SUBMODULE_LOG_v'] = r'{RUNTIME_FRAMEWORK:0,SYMBOLIC_SHAPE:0,GRAPH_KERNEL:0}'
    command = "pytest --disable-warnings -s {} > {} 2>&1".format(test_case, log_file)
    try:
        subprocess.run(command, shell=True, env=env, timeout=600, check=True, text=True)
    except Exception as e:
        import datetime
        t = datetime.datetime.now()
        f = t.strftime('%m%d%H%M%S')
        os.system(f"mkdir ~/graph_kernel_log_{f}")
        os.system(f"cp -rf {log_file} ~/graph_kernel_log_{f}")
        if os.path.isfile(log_file):
            with open(log_file, 'r') as f:
                for line in f:
                    print(line)
        os.system(f"cp -rf ~/ascend/log ~/graph_kernel_log_{f}")
        raise RuntimeError("{}\n case {} run failed!".format(e, test_case))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_reshape():
    """
    Feature: KernelPacket
    Description: test kernelpacket with reshape
    Expectation: success
    """
    run_with_log("kernelpacket_cases.py::test_reshape", "test_kernelpacket_reshape.log")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_reducesum():
    """
    Feature: KernelPacket
    Description: test kernelpacket with ReduceSum
    Expectation: success
    """
    run_with_log("kernelpacket_cases.py::test_reducesum", "test_reducesum.log")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_fuse_host_ops():
    """
    Feature: KernelPacket
    Description: test kernelpacket with host-device ops
    Expectation: success
    """
    run_with_log("kernelpacket_cases.py::test_fuse_host_ops", "test_fuse_host_ops.log")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=0.01)
def test_stridedslice():
    """
    Feature: KernelPacket
    Description: test kernelpacket with stridedslice
    Expectation: success
    """
    run_with_log("kernelpacket_cases.py::test_stridedslice", "test_stridedslice.log")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_matmul_only_shape():
    """
    Feature: KernelPacket
    Description: test kernelpacket to fuse the only-shape-depended ops.
    Expectation: success
    """
    run_with_log("kernelpacket_cases.py::test_matmul_only_shape", "test_matmul_only_shape.log")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_concat_grad():
    """
    Feature: KernelPacket
    Description: test kernelpacket with slice
    Expectation: success
    """
    run_with_log("kernelpacket_cases.py::test_concat_grad", "test_concat_grad.log")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=0.01)
def test_stridedslice_grad():
    """
    Feature: KernelPacket
    Description: test kernelpacket with stridedslicegrad
    Expectation: success
    """
    run_with_log("kernelpacket_cases.py::test_stridedslice_grad", "test_stridedslice_grad.log")
