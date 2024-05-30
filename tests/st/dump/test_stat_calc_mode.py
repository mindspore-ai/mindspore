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
import os
import glob
import csv
import pytest
import mindspore.context as context
import tempfile
import time
import math

from mindspore import JitConfig, Tensor, nn
from pathlib import Path
from dump_test_utils import generate_statistic_dump_json, generate_dump_json

def check_statistic_l2_value(tensor, l2_value):
    if "L2 Value" in tensor:
        assert math.isclose(float(tensor["L2 Value"]), l2_value, rel_tol=1e-4, abs_tol=1e-4)

def check_statistic_device_dump(dump_file_path):
    output_name = "statistic.csv"
    output_path = glob.glob(os.path.join(dump_file_path, output_name))[0]
    real_path = os.path.realpath(output_path)
    with open(real_path) as f:
        reader = csv.DictReader(f)
        stats = list(reader)

        def get_add_node(statistic):
            return statistic['Op Type'] == 'Add'

        add_statistics = list(filter(get_add_node, stats))
        num_tensors = len(add_statistics)
        assert num_tensors == 3
        for tensor in add_statistics:
            if tensor['IO'] == 'input' and tensor['Slot'] == '0':
                assert tensor['Min Value'] == '1'
                assert tensor['Max Value'] == '3'
                check_statistic_l2_value(tensor, 3.7416)
            elif tensor['IO'] == 'input' and tensor['Slot'] == '1':
                assert tensor['Min Value'] == '-10'
                assert tensor['Max Value'] == '2'
                check_statistic_l2_value(tensor, 10.3923)
            elif tensor['IO'] == 'output' and tensor['Slot'] == '0':
                assert tensor['Min Value'] == '-7'
                assert tensor['Max Value'] == '4'
                check_statistic_l2_value(tensor, 8.6023)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_kbk_stat_calc_mode_dump():
    """
    Feature: kbyk statistic dump support device calc.
    Description: Test kbyk statistic dump on device.
    Expectation: The statistics result does not meet the requirement.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_dir = tempfile.TemporaryDirectory(suffix="stat_calc_mode")

    path = Path(test_dir.name)
    dump_path = str(path / "dump_data")
    dump_config_path = str(path / "config.json")

    generate_statistic_dump_json(dump_path, dump_config_path, "stat_calc_mode", "statistic")
    os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
    try:
        class Net(nn.Cell):
            def construct(self, x, y):
                return x + y
        jit_config = JitConfig(jit_level="O0")
        net = Net()
        net.set_jit_config(jit_config)
        x = Tensor([1., 2., 3.])
        y = Tensor([2., 2., -10.])
        _ = net(x, y)
        time.sleep(2)
        check_statistic_device_dump(path / "dump_data" / "rank_0" / "Net" / "0" / "0")
    finally:
        del os.environ['MINDSPORE_DUMP_CONFIG']


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_kbk_stat_calc_mode_l2_dump():
    """
    Feature: kbyk statistic dump support host l2 value dump.
    Description: Test kbyk statistic l2 value dump on host.
    Expectation: The statistics result meet the requirement.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_dir = tempfile.TemporaryDirectory(suffix="stat_calc_mode")

    path = Path(test_dir.name)
    dump_path = str(path / "dump_data")
    dump_config_path = str(path / "config.json")

    def extra_json_settings(data):
        data["e2e_dump_settings"]["stat_calc_mode"] = "host"
        data["common_dump_settings"]["saved_data"] = "statistic"

    generate_dump_json(dump_path, dump_config_path, "e2e_dump_settings", extra_json_settings)
    os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
    try:
        class Net(nn.Cell):
            def construct(self, x, y):
                return x + y
        jit_config = JitConfig(jit_level="O0")
        net = Net()
        net.set_jit_config(jit_config)
        x = Tensor([1., 2., 3.])
        y = Tensor([2., 2., -10.])
        _ = net(x, y)
        time.sleep(2)
        check_statistic_device_dump(path / "dump_data" / "rank_0" / "Net" / "0" / "0")
    finally:
        del os.environ['MINDSPORE_DUMP_CONFIG']
