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
"""
Tests for exception dump.
"""

import sys
import os
import tempfile
import shutil
import time
import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore import Tensor
from tests.mark_utils import arg_mark
from dump_test_utils import generate_dump_json, check_dump_structure

ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend')


class Net(ms.nn.Cell):
    """Gather算子越界场景"""
    def construct(self, params, indices, axis):
        out = ops.gather(params, indices, axis)
        return out


class MoveToNet(ms.nn.Cell):
    """Gather算子异构越界场景"""
    def construct(self, params, indices, axis):
        params = ops.move_to(params, "CPU")
        out = ops.gather(params, indices, axis)
        return out


def run_exception_net():
    ms.set_context(jit_config={"jit_level": "O0"})
    ms.device_context.ascend.op_debug.aclinit_config({"dump": {"dump_scene": "lite_exception"}})

    input_params = Tensor(np.random.uniform(0, 1, size=(64,)).astype("float32"))
    input_indices = Tensor(np.array([100000, 101]), ms.int32)
    input_axis = 0
    net = Net()
    out = net(input_params, input_indices, input_axis)
    return out


def run_exception_heterogeneous_net():
    ms.set_context(jit_config={"jit_level": "O0"})
    input_params = Tensor(np.random.uniform(0, 1, size=(64,)).astype("float32"))
    input_indices = Tensor(np.array([100000, 101]), ms.int32)
    input_axis = 0
    net = MoveToNet()
    out = net(input_params, input_indices, input_axis)
    return out


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_exception_dump():
    """
    Feature: Test exception dump.
    Description: abnormal node should be dumped.
    Expectation: The AllReduce data is saved and the value is correct.
    """
    if sys.platform != 'linux':
        return
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'test_exception_dump')
        dump_config_path = os.path.join(tmp_dir, 'test_exception_dump.json')
        generate_dump_json(dump_path, dump_config_path, 'test_exception_dump', 'exception_data')
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        dump_file_path = os.path.join(dump_path, 'rank_0', 'exception_data', '0', '0')
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        exec_network_cmd = ('cd {0}; python -c "from test_exception_dump import run_exception_net;'
                            'run_exception_net()"').format(os.getcwd())
        _ = os.system(exec_network_cmd)
        for _ in range(3):
            if not os.path.exists(dump_file_path):
                time.sleep(2)
        check_dump_structure(dump_path, dump_config_path, 1, 1, 1, execution_history=False)
        del os.environ['MINDSPORE_DUMP_CONFIG']


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_exception_heterogeneous_dump():
    """
    Feature: Test exception dump in heterogeneous scenarios.
    Description: abnormal node should be dumped and the data on the CPU will be dumped.
    Expectation: The AllReduce data is saved and the value is correct.
    """
    if sys.platform != 'linux':
        return
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'test_exception_dump')
        dump_config_path = os.path.join(tmp_dir, 'test_exception_dump.json')
        generate_dump_json(dump_path, dump_config_path, 'test_exception_dump', 'exception_data')
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        dump_file_path = os.path.join(dump_path, 'rank_0', 'exception_data', '0', '0')
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        exec_network_cmd = ('cd {0}; python -c "from test_exception_dump import run_exception_heterogeneous_net;'
                        'run_exception_heterogeneous_net()"').format(os.getcwd())
        _ = os.system(exec_network_cmd)
        for _ in range(3):
            if not os.path.exists(dump_file_path):
                time.sleep(2)
        check_dump_structure(dump_path, dump_config_path, 1, 1, 1, execution_history=False)

        # 统计所有npy文件，以及其中名字包含'input.0'的文件
        npy_num = 0
        input0_num = 0
        for filename in os.listdir(dump_file_path):
            if filename.endswith('.npy'):
                npy_num += 1
                if 'input.0' in filename:
                    input0_num += 1
        assert npy_num == 4
        assert input0_num == 1

        del os.environ['MINDSPORE_DUMP_CONFIG']


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_lite_exception_dump():
    """
    Feature: Test lite exception dump.
    Description: abnormal node should be dumped.
    Expectation: The exception data is save.
    """
    exception_file_path = "./extra-info"
    if os.path.exists(exception_file_path):
        shutil.rmtree(exception_file_path)
    exec_network_cmd = ('cd {0}; python -c "from test_exception_dump import run_exception_net;'
                        'run_exception_net()"').format(os.getcwd())
    _ = os.system(exec_network_cmd)

    for _ in range(3):
        if not os.path.exists(exception_file_path):
            time.sleep(2)
    assert os.path.exists(exception_file_path)
    shutil.rmtree(exception_file_path)
