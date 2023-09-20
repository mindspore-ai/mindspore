# Copyright 2021 Huawei Technologies Co., Ltd
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
import sys
import tempfile
import time
import shutil
import glob
import json
import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from dump_test_utils import generate_dump_json, generate_dump_json_with_overflow, check_dump_structure
from tests.security_utils import security_off_wrap


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.Add()

    def construct(self, x_, y_):
        return self.add(x_, y_)


x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
y = np.array([[7, 8, 9], [10, 11, 12]]).astype(np.float32)


def run_async_dump(test_name):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'async_dump')
        dump_config_path = os.path.join(tmp_dir, 'async_dump.json')
        generate_dump_json(dump_path, dump_config_path, test_name)
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        dump_file_path = os.path.join(dump_path, 'rank_0', 'Net', '0', '0')
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        add = Net()
        add(Tensor(x), Tensor(y))
        for _ in range(3):
            if not os.path.exists(dump_file_path):
                time.sleep(2)
        check_dump_structure(dump_path, dump_config_path, 1, 1, 1)
        assert len(os.listdir(dump_file_path)) == 1
        # check content of the generated dump data
        if test_name == "test_async_dump_npy":
            output_name = "Add.Add-op*.*.*.*.output.0.ND.npy"
            output_path = glob.glob(os.path.join(dump_file_path, output_name))[0]
            real_path = os.path.realpath(output_path)
            output = np.load(real_path)
            expect = np.array([[8, 10, 12], [14, 16, 18]], np.float32)
            assert np.array_equal(output, expect)
        del os.environ['MINDSPORE_DUMP_CONFIG']


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_async_dump_npy():
    """
    Feature: async dump on Ascend
    Description: test async dump with file_format = "npy"
    Expectation: dump data are generated as npy file format
    """
    run_async_dump("test_async_dump_npy")


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_async_dump_bin():
    """
    Feature: async dump on Ascend in npy format
    Description: test async dump with file_format = "bin"
    Expectation: dump data are generated as protobuf file format (suffix with timestamp)
    """
    run_async_dump("test_async_dump_bin")


def run_overflow_dump(test_name):
    """Run async dump and generate overflow"""
    if sys.platform != 'linux':
        return
    overflow_x = np.array([60000, 60000]).astype(np.float16)
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'overflow_dump')
        dump_config_path = os.path.join(tmp_dir, 'overflow_dump.json')
        generate_dump_json_with_overflow(dump_path, dump_config_path, test_name, 3)
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        add = Net()
        add(Tensor(overflow_x), Tensor(overflow_x))
        exe_graph_path = os.path.join(dump_path, 'rank_0', 'Net', '0', '0')
        for _ in range(5):
            if not os.path.exists(exe_graph_path):
                time.sleep(2)
        check_dump_structure(dump_path, dump_config_path, 1, 1, 1)
        # check if overflow dump generate exact two files, and the naming format
        assert len(os.listdir(exe_graph_path)) == 2
        output_path = glob.glob(os.path.join(exe_graph_path, "Add.Add-op*.*.*.*.output.0.ND.npy"))[0]
        overflow_path = glob.glob(os.path.join(exe_graph_path, "Opdebug.Node_OpDebug.*.*.*.output.0.json"))[0]
        assert output_path
        assert overflow_path
        # check content of the output tensor
        real_path = os.path.realpath(output_path)
        output = np.load(real_path)
        expect = np.array([65504, 65504], np.float16)
        assert np.array_equal(output, expect)
        # check content of opdebug info json file
        with open(overflow_path, 'rb') as json_file:
            data = json.load(json_file)
            assert data
        del os.environ['MINDSPORE_DUMP_CONFIG']


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_ascend_overflow_dump():
    """
    Feature: Overflow Dump
    Description: Test overflow dump
    Expectation: Overflow is occurred, and overflow dump file is in correct format
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    run_overflow_dump("test_async_dump_npy")
