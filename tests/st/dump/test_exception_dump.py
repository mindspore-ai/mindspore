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
import sys
import os
import pytest
import tempfile
import shutil
import time
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor
from dump_test_utils import generate_dump_json, check_dump_structure

ms.set_context(mode=0, device_target='Ascend')


class Net(ms.nn.Cell):
    """Gather算子溢出场景"""
    def construct(self, params, indices, axis):
        """Construct."""
        out = ops.gather(params, indices, axis)
        return out


def run_exception_net():
    ms.set_context(jit_level='O0')
    input_params = Tensor(np.random.uniform(0, 1, size=(64,)).astype("float32"))
    input_indices = Tensor(np.array([100000, 101]), ms.int32)
    input_axis = 0
    net = Net()
    out = net(input_params, input_indices, input_axis)
    return out


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
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


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
def test_acl_dump_exception():
    """
    Feature: Test exception dump.
    Description: abnormal node should be dumped.
    Expectation: The exception data is save.
    """
    if sys.platform != 'linux':
        return
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'test_acl_dump_exception')
        dump_config_path = os.path.join(tmp_dir, 'test_acl_dump_exception.json')
        generate_dump_json(dump_path, dump_config_path, 'test_acl_dump_exception', 'exception_data')
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        exec_network_cmd = ('cd {0}; python -c "from test_exception_dump import run_exception_net;'
                            'run_exception_net()"').format(os.getcwd())
        _ = os.system(exec_network_cmd)
        exception_file_path = "./extra-info"
        for _ in range(3):
            if not os.path.exists(exception_file_path):
                time.sleep(2)
        assert os.path.exists(exception_file_path)
        del os.environ['MINDSPORE_DUMP_CONFIG']
