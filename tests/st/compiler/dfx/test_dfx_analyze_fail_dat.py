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
""" analyze fail test """
import os
import numpy as np
import pytest
from mindspore import jit, nn, Tensor
import mindspore.ops.operations as op
from mindspore.graph.api import _pynative_executor
from tests.st.pynative.utils import GradOfAllInputs
from tests.mark_utils import arg_mark


def find_keywords_in_logfile(file, keywords):
    find_result = False
    with open(file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if keywords in line:
                find_result = True
                break
    assert find_result


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.addn = op.AddN()
        self.relu = nn.ReLU()

    @jit
    def construct(self, input_x, input_y):
        input_z = self.addn((input_x, input_y))
        out = self.relu(input_z)
        return out


def get_folder_path(file_path):
    device_id = 0

    folder_path = os.path.join(file_path, "rank_" + str(device_id), 'om/')
    return folder_path


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_dfx_analyze_fail_dat_001():
    """
    Feature: Analyzefail Check.
    Description: Check if Analyzefail fail document exist.
    Expectation: Analyzefail fail document exist.
    """
    base_dir = os.path.split(os.path.realpath(__file__))[0]
    folder_path = get_folder_path(base_dir)
    log_file = os.path.join(base_dir, 'analyze_fail.log')
    analyze_fail_file = os.path.join(folder_path, 'analyze_fail.ir')

    cmd = "cd {0} && pytest -v -ra  analyze_fail_net.py > analyze_fail.log 2>&1".format(
        base_dir)
    out = os.popen(cmd)
    out.read()

    assert os.path.exists(analyze_fail_file)
    find_keywords_in_logfile(log_file, 'analyze_fail.ir')


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_dfx_analyze_fail_dat_002():
    """
    Feature: Analyzefail Check.
    Description: Check if Analyzefail fail document exist.
    Expectation: Analyzefail fail document exist.
    """
    base_dir = os.path.split(os.path.realpath(__file__))[0]
    ms_om_path = os.path.join(base_dir, 'analyze01哈哈')
    os.environ['MS_OM_PATH'] = ms_om_path
    folder_path = get_folder_path(ms_om_path)
    log_file = os.path.join(base_dir, 'analyze_fail.log')
    analyze_fail_file = os.path.join(folder_path, 'analyze_fail.ir')
    cmd = "cd {0} && pytest -v -ra  analyze_fail_net.py > analyze_fail.log 2>&1".format(
        base_dir)
    out = os.popen(cmd)
    out.read()
    mtime_01 = os.path.getmtime(analyze_fail_file)
    out = os.popen(cmd)
    out.read()
    mtime_02 = os.path.getmtime(analyze_fail_file)

    assert os.path.exists(analyze_fail_file)
    assert mtime_01 != mtime_02
    find_keywords_in_logfile(log_file, analyze_fail_file)
    del os.environ['MS_OM_PATH']


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_dfx_analyze_fail_dat_003():
    """
    Feature: Analyzefail Check.
    Description: Check if Analyzefail fail document exist.
    Expectation: Analyzefail fail document exist.
    """
    base_dir = os.path.split(os.path.realpath(__file__))[0]
    ms_om_path = os.path.join(base_dir, '../dfx/analyze02/01')
    os.environ['MS_OM_PATH'] = ms_om_path
    folder_path = get_folder_path(base_dir + '/analyze02/01')
    log_file = os.path.join(base_dir, 'analyze_fail.log')
    analyze_fail_file = os.path.join(folder_path, 'analyze_fail.ir')

    cmd = "cd {0} && pytest -v -ra  analyze_fail_net.py > analyze_fail.log 2>&1".format(
        base_dir)
    out = os.popen(cmd)
    out.read()
    assert os.path.exists(analyze_fail_file)
    os.remove(analyze_fail_file)
    out = os.popen(cmd)
    out.read()
    assert os.path.exists(analyze_fail_file)
    find_keywords_in_logfile(log_file, analyze_fail_file)
    del os.environ['MS_OM_PATH']


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_dfx_analyze_fail_dat_005():
    """
    Feature: Analyzefail Check.
    Description: Check if Analyzefail fail document exist.
    Expectation: Analyzefail fail document exist.
    """
    base_dir = os.path.split(os.path.realpath(__file__))[0]
    folder_path = get_folder_path(base_dir + '/analyze03')
    log_file = os.path.join(base_dir, 'analyze_fail.log')
    analyze_fail_file = os.path.join(folder_path, 'analyze_fail.ir')
    cmd = "cd {0} && pytest -v -ra  analyze_fail_net.py > analyze_fail.log 2>&1".format(
        base_dir)

    os.environ['MS_OM_PATH'] = os.path.join(base_dir, 'analyze03')
    out = os.popen(cmd)
    out.read()
    os.environ['MS_OM_PATH'] = analyze_fail_file
    out = os.popen(cmd)
    out.read()

    if 'DEVICE_ID' in os.environ:
        device_id = int(os.environ['DEVICE_ID'])
    else:
        device_id = 0
    keywords_to_find = os.path.join(
        analyze_fail_file, 'rank_', str(device_id), '/om')
    find_keywords_in_logfile(log_file, keywords_to_find)
    del os.environ['MS_OM_PATH']


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_dfx_analyze_fail_dat_004():
    """
    Feature: Analyzefail Check.
    Description: Check if Analyzefail fail document exist.
    Expectation: Analyzefail fail document exist.
    """
    base_dir = os.getcwd()
    folder_path = get_folder_path(base_dir)
    os.environ['MS_OM_PATH'] = ''
    analyze_fail_file = os.path.join(folder_path, 'analyze_fail.ir')
    input_1 = Tensor(np.random.randn(2, 3).astype(np.float32))
    input_2 = Tensor(np.random.randn(2, 3, 4).astype(np.float32))
    net = Net()
    net.set_grad()
    grad_net = GradOfAllInputs(net)
    grad_net.set_train()
    with pytest.raises(TypeError):
        grad_net(input_1, input_1, input_2, input_2)
        _pynative_executor.sync()

    assert os.path.exists(analyze_fail_file)
    del os.environ['MS_OM_PATH']
