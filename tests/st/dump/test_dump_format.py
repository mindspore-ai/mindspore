# Copyright 2022 Huawei Technologies Co., Ltd
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
import glob
import shutil
import pytest
import numpy as np
import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from dump_test_utils import generate_dump_json, check_dump_structure
from tests.security_utils import security_off_wrap


class ConvNet(nn.Cell):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv2 = ops.Conv2D(out_channel=3, kernel_size=1)

    def construct(self, x, weight):
        return self.conv2(x, weight)


def run_trans_flag(test_name):
    if sys.platform != 'linux':
        return
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, test_name)
        dump_config_path = os.path.join(tmp_dir, '{}.json'.format(test_name))
        generate_dump_json(dump_path, dump_config_path, test_name)
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        net = ConvNet()
        tensor = Tensor(np.ones([1, 3, 3, 3]), ms.float32)
        weight = Tensor(np.ones([3, 3, 1, 1]), ms.float32)
        expect = net(tensor, weight)
        check_dump_structure(dump_path, dump_config_path, 1, 1, 1)
        dump_data_path = os.path.join(dump_path, 'rank_0', 'Net', '0', '0')
        assert os.path.exists(dump_data_path)
        if test_name == "test_e2e_dump_trans_true":
            # tensor data in host format.
            output_name = "Conv2D.Conv2D-op*.0.0.*.output.0.DefaultFormat.npy"
            output_path = glob.glob(os.path.join(dump_data_path, output_name))[0]
            real_path = os.path.realpath(output_path)
            output = np.load(real_path)
            assert output.shape == (1, 3, 3, 3)
            assert np.array_equal(output, expect)
        elif test_name == "test_e2e_dump_trans_false":
            # tensor data in device format.
            output_name = "Conv2D.Conv2D-op*.0.0.*.output.0.NC1HWC0.npy"
            output_path = glob.glob(os.path.join(dump_data_path, output_name))[0]
            real_path = os.path.realpath(output_path)
            output = np.load(real_path)
            assert output.shape == (1, 1, 3, 3, 16)
        else:
            # tensor data in host format.
            output_name = "Conv2D.Conv2D-op*.*.*.*.output.0.NCHW.npy"
            output_path = glob.glob(os.path.join(dump_data_path, output_name))[0]
            real_path = os.path.realpath(output_path)
            output = np.load(real_path)
            assert output.shape == (1, 3, 3, 3)
            assert np.array_equal(output, expect)
        del os.environ['MINDSPORE_DUMP_CONFIG']


@pytest.mark.level1
@pytest.mark.env_onecard
@security_off_wrap
def test_ascend_e2e_trans_true():
    """
    Feature: Ascend e2e dump.
    Description: Test e2e dump in Ascend with trans_flag is configured to true.
    Expectation: Dump files has tensor data in host format (4 dimensions).
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_trans_flag("test_e2e_dump_trans_true")


@pytest.mark.level1
@pytest.mark.env_onecard
@security_off_wrap
def test_ascend_e2e_trans_false():
    """
    Feature: Ascend e2e dump.
    Description: Test e2e dump in Ascend with trans_flag is configured to false.
    Expectation: Dump files has tensor data in device format.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_trans_flag("test_e2e_dump_trans_false")


@pytest.mark.level1
@pytest.mark.env_onecard
@security_off_wrap
def test_ascend_kernel_by_kernel_trans_true():
    """
    Feature: Ascend kernel by kernel dump.
    Description: Test kernel by kernel dump in Ascend with trans_flag is configured to true.
    Expectation: Dump files has tensor data in host format (4 dimensions).
    """
    context.set_context(jit_level='O0')
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_trans_flag("test_e2e_dump_trans_true")


@pytest.mark.level1
@pytest.mark.env_onecard
@security_off_wrap
def test_ascend_kernel_by_kernel_trans_false():
    """
    Feature: Ascend kernel by kernel dump.
    Description: Test kernel by kernel dump in Ascend with trans_flag is configured to false.
    Expectation: Dump files has tensor data in device format.
    """
    context.set_context(jit_level='O0')
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_trans_flag("test_e2e_dump_trans_false")


@pytest.mark.level0
@pytest.mark.env_onecard
@security_off_wrap
def test_ascend_a_plus_m_conversion():
    """
    Feature: Ascend A+M dump.
    Description: Test A+M dump in Ascend and check the format of the dump data.
    Expectation: Dump files has tensor data in host format (4 dimensions).
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_trans_flag("test_async_dump_npy")
