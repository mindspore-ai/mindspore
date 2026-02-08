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
import os
import glob
import csv
import numpy as np
from mindspore import context
import tempfile
import time
import json
import hashlib
import mindspore
import math

from mindspore import Tensor, nn, ops
from pathlib import Path
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    def construct(self, x, y):
        return x + y

class MoveToNet(nn.Cell):
    def construct(self, x):
        cpu_x = ops.move_to(x, "CPU")
        npu_x = ops.move_to(cpu_x, "Ascend")
        return npu_x


def generate_e2edump_json(dump_path, json_file_name, extra_settings_func=None, assign_dump_path=True):
    current_dir = Path(__file__).parent
    json_path = current_dir / "test_e2e_statistic_config.json"
    with open(json_path, 'r') as file:
        data = json.load(file)
        if assign_dump_path:
            data["common_dump_settings"]["path"] = dump_path
        if extra_settings_func is not None:
            extra_settings_func(data)
    with open(json_file_name, 'w') as f:
        json.dump(data, f)


def is_float_equal(value1, value2, rel_tol=1e-4, abs_tol=1e-4):
    try:
        value1 = float(value1)
        value2 = float(value2)
        return np.isclose(value1, value2, rtol=rel_tol, atol=abs_tol, equal_nan=True)
    except ValueError:
        return value1 == value2


def to_comparable_pairs(data):
    for key, value in data.items():
        if key in {'Max Value', 'Min Value', 'L2Norm Value', 'Avg Value'}:
            yield key, float(value)
        else:
            yield key, value


def match_dicts(target, data):
    for key, target_value in target.items():
        data_value = data.get(key)
        if isinstance(target_value, float):
            if not is_float_equal(target_value, data_value):
                return False
        else:
            if target_value != data_value:
                return False
    return True


def check_statistic_result(data_list, target_list):
    for target in target_list:
        target_pairs = dict(to_comparable_pairs(target))
        assert any(match_dicts(target_pairs, dict(to_comparable_pairs(data)))
                   for data in data_list)


def get_dumped_stat_list(dump_file_path):
    output_name = "statistic.csv"
    output_path = glob.glob(os.path.join(dump_file_path, output_name))[0]
    real_path = os.path.realpath(output_path)
    with open(real_path) as f:
        reader = csv.DictReader(f)
        stats_list = list(reader)
        for stat in stats_list:
            stat.pop(None, None)
        return stats_list


def check_moveto_dump(dump_path):
    stat_list = get_dumped_stat_list(Path(dump_path) / "rank_0" / "Net" / "0" / "0")
    for data in stat_list:
        if data['Op Name'] == 'Default/MoveTo-op0':
            assert data['IO'] == 'input'


def compare_single_data(x, y, data_len, net, dump_path, precision_mode="high"):
    t_x, t_y = x, y
    t_out = x + y
    if precision_mode == "high":
        t_x, t_y, t_out = t_x.astype(np.float32), t_y.astype(np.float32), t_out.astype(np.float32)

    common_res = {'Op Type': 'Add', 'Op Name': 'Default/Add-op0', 'Data Size': str(x.nbytes), 'Data Type': str(x.dtype),
                  'Shape': "[" + str(data_len) + "]"}
    target_list = []
    for idx, tensor in enumerate([t_x, t_y]):
        target = {**common_res, **{'IO': 'input', 'Slot': str(idx)}}
        target.update({
            'Max Value': format(tensor.max(), '.6g'), 'Min Value': format(tensor.min(), '.6g'),
            'Avg Value': format(tensor.mean(), '.6g'), 'L2Norm Value': format(np.linalg.norm(tensor), '.6g')
        })
        target_list.append(target)
    target_output = {**common_res, **{'IO': 'output', 'Slot': '0', 'Max Value': format(t_out.max(), '.6g'),
                                      'Min Value': format(t_out.min(), '.6g'),
                                      'Avg Value': format(t_out.mean(), '.6g'),
                                      'L2Norm Value': format(np.linalg.norm(t_out), '.6g')}}
    target_list.append(target_output)
    t = net(Tensor(x), Tensor(y))
    print(t)
    time.sleep(1)
    stat_list = get_dumped_stat_list(dump_path)
    assert len(stat_list) == 3
    check_statistic_result(stat_list, target_list)

def cal_sha1(value):
    sha1hash = hashlib.sha1(value)
    return sha1hash.hexdigest()

def compare_sha1_data(net, dump_path):
    for i, (x, y) in enumerate(TEST_CASES):
        out = x + y
        case = (cal_sha1(x), cal_sha1(y), cal_sha1(out))
        stat_list = get_dumped_stat_list(Path(dump_path) / "rank_0" / "Net" / "0" / str(i))
        for (data, case_data) in zip(stat_list, case):
            assert case_data == dict(to_comparable_pairs(data))['SHA1']

def cal_parallel_sha1(value):
    num_per_piece = 20480
    length = len(value)
    piece_num = math.floor((length - 1) / num_per_piece) + 1
    print("parallel_sha1_piece_num:", piece_num)
    if piece_num == 1:
        return cal_sha1(value)
    sha1_combine = ""
    for i in range(piece_num):
        one_piece = value[(num_per_piece * i) : num_per_piece * (i + 1)]
        sha1_combine += cal_sha1(one_piece)
    return cal_sha1(sha1_combine.encode('utf-8'))

def compare_parallel_hash(x, y, data_len, net, dump_path, precision_mode="high"):
    t_x, t_y = x, y
    t_out = x + y
    if precision_mode == "high":
        t_x, t_y, t_out = t_x.astype(np.float32), t_y.astype(np.float32), t_out.astype(np.float32)

    common_res = {'Op Type': 'Add', 'Op Name': 'Default/Add-op0', 'Data Size': str(x.nbytes), 'Data Type': str(x.dtype),
                  'Shape': "[" + str(data_len) + "]"}
    target_list = []
    for idx, tensor in enumerate([t_x, t_y]):
        target = {**common_res, **{'IO': 'input', 'Slot': str(idx)}}
        target.update({
            'Max Value': format(tensor.max(), '.6g'), 'Min Value': format(tensor.min(), '.6g'),
            'Avg Value': format(tensor.mean(), '.6g'), 'L2Norm Value': format(np.linalg.norm(tensor), '.6g'),
            'SHA1': cal_parallel_sha1(tensor)
        })
        target_list.append(target)

    target_output = {**common_res, **{'IO': 'output', 'Slot': '0', 'Max Value': format(t_out.max(), '.6g'),
                                      'Min Value': format(t_out.min(), '.6g'),
                                      'Avg Value': format(t_out.mean(), '.6g'),
                                      'L2Norm Value': format(np.linalg.norm(t_out), '.6g'),
                                      'SHA1': cal_parallel_sha1(t_out)}}
    target_list.append(target_output)
    net(Tensor(x), Tensor(y))
    time.sleep(1)
    stat_list = get_dumped_stat_list(dump_path)
    assert len(stat_list) == 3
    check_statistic_result(stat_list, target_list)

TEST_CASES = [
        (np.array([40000, 40000, 40000], np.float16),
         np.array([40000, 40000, 40000], np.float16)),
        (np.array([1., 2., float('inf')], np.float16),
         np.array([-float("inf"), 2., -10.], np.float16)),
        (np.array([1., 2., 3.], np.float16),
         np.array([2., 2., -10.], np.float16)),
        (np.array([float('inf'), float('inf'), float('inf')], np.float16),
         np.array([float('inf'), float('inf'), float('inf')], np.float16)),
        (np.array([float('-inf'), float('-inf'), float('-inf')], np.float16),
         np.array([float('-inf'), float('-inf'), float('-inf')], np.float16)),
    ]

def compare_multi_data(net, dump_path, precision_mode="high"):
    for i, (x, y) in enumerate(TEST_CASES):
        compare_single_data(x, y, len(x), net, Path(dump_path) / "rank_0" / "Net" / "0" / str(i), precision_mode)

def cal_md5(value):
    md5hash = hashlib.md5(value)
    return md5hash.hexdigest()

def compare_md5_data(net, dump_path):
    for i, (x, y) in enumerate(TEST_CASES):
        out = x + y
        case = (cal_md5(x), cal_md5(y), cal_md5(out))
        stat_list = get_dumped_stat_list(Path(dump_path) / "rank_0" / "Net" / "0" / str(i))
        for (data, case_data) in zip(stat_list, case):
            assert case_data == dict(to_comparable_pairs(data))['MD5']

def compare_massive_data(net, dump_path):
    data_len = 42000
    case_data = mindspore.ops.rand((2, data_len), dtype=mindspore.float32)
    case_data[0][0] = float('inf')
    case_data[1][0] = float('inf')
    compare_parallel_hash(case_data[0].asnumpy(), case_data[1].asnumpy(), data_len, net,
                          Path(dump_path) / "rank_0" / "Net" / "0" / "0")

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=4.0)
def test_e2e_statistic_async_device_high_precision():
    """
    Feature: kbyk statistic dump support device async high precision
    Description: Test kbyk statistic dump on device in high precision mode.
    Expectation: The statistics result meet the requirement.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", jit_config={"jit_level": "O0"})

    def extra_json_settings(data):
        data["e2e_dump_settings"]["stat_calc_mode"] = "device"
        data["e2e_dump_settings"]["enable"] = False

    with tempfile.TemporaryDirectory() as test_dir:
        path = Path(test_dir)
        dump_path = str(path / "dump_data")
        dump_config_path = str(path / "config.json")
        generate_e2edump_json(dump_path, dump_config_path, extra_json_settings)

        try:
            os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
            net = Net()
            compare_multi_data(net, dump_path)
        finally:
            del os.environ['MINDSPORE_DUMP_CONFIG']


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=4.0)
def test_e2e_statistic_async_device_low_precision():
    """
    Feature: kbyk statistic dump support device async low precision
    Description: Test kbyk statistic dump on device in low precision mode.
    Expectation: The statistics result meet the requirement.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", jit_config={"jit_level": "O0"})

    def extra_json_settings(data):
        data["e2e_dump_settings"]["stat_calc_mode"] = "device"
        data["e2e_dump_settings"]["device_stat_precision_mode"] = "low"
        data["e2e_dump_settings"]["enable"] = False

    with tempfile.TemporaryDirectory(suffix="e2e_statistic_host_with_nan_and_inf") as test_dir:
        path = Path(test_dir)
        dump_path = str(path / "dump_data")
        dump_config_path = str(path / "config.json")
        generate_e2edump_json(dump_path, dump_config_path, extra_json_settings)

        try:
            os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
            net = Net()
            compare_multi_data(net, dump_path, precision_mode="low")
        finally:
            del os.environ['MINDSPORE_DUMP_CONFIG']


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
def test_e2e_statistic_sync_device():
    """
    Feature: kbyk statistic dump support device sync with assign MS_DIAGNOSTIC_DATA_PATH
    Description: Test kbyk statistic dump on device with assign MS_DIAGNOSTIC_DATA_PATH.
    Expectation: The statistics result meet the requirement.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", jit_config={"jit_level": "O0"})

    def extra_json_settings(data):
        data["e2e_dump_settings"]["stat_calc_mode"] = "device"
        data["e2e_dump_settings"]["enable"] = True

    with tempfile.TemporaryDirectory() as test_dir:
        path = Path(test_dir)
        dump_path = str(path / "debug_dump")
        dump_config_path = str(path / "config.json")
        generate_e2edump_json(dump_path, dump_config_path, extra_json_settings, False)

        try:
            os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
            os.environ['MS_DIAGNOSTIC_DATA_PATH'] = str(path)
            net = Net()
            compare_multi_data(net, dump_path)
        finally:
            del os.environ['MS_DIAGNOSTIC_DATA_PATH']
            del os.environ['MINDSPORE_DUMP_CONFIG']


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=4.0)
def test_e2e_statistic_sync_host():
    """
    Feature: kbyk statistic dump support host sync, host also supports md5
    Description: Test kbyk statistic dump on host.
    Expectation: The statistics result meet the requirement.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", jit_config={"jit_level": "O0"})

    def extra_json_settings(data):
        data["e2e_dump_settings"]["stat_calc_mode"] = "host"
        data["e2e_dump_settings"]["enable"] = True
        data["e2e_dump_settings"]["save_kernel_args"] = True
        data["common_dump_settings"]["statistic_category"].append("md5")
        data["common_dump_settings"]["statistic_category"].append("hash:sha1")
        data["common_dump_settings"]["statistic_category"].append("hash")

    with tempfile.TemporaryDirectory() as test_dir:
        path = Path(test_dir)
        dump_path = str(path / "dump_data")
        dump_config_path = str(path / "config.json")
        generate_e2edump_json(dump_path, dump_config_path, extra_json_settings)

        try:
            os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
            net = Net()
            compare_multi_data(net, dump_path)
            compare_md5_data(net, dump_path)
            compare_sha1_data(net, dump_path)
        finally:
            del os.environ['MINDSPORE_DUMP_CONFIG']

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
def test_e2e_statistic_massive_data():
    """
    Feature: kbyk statistic dump with massive data
    Description: Test kbyk statistic dump with massive data.
    Expectation: The statistics result meet the requirement.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", jit_config={"jit_level": "O0"})

    def extra_json_settings(data):
        data["e2e_dump_settings"]["stat_calc_mode"] = "host"
        data["e2e_dump_settings"]["enable"] = False
        data["common_dump_settings"]["statistic_category"].append("hash:sha1")

    with tempfile.TemporaryDirectory() as test_dir:
        path = Path(test_dir)
        dump_path = str(path / "dump_data")
        dump_config_path = str(path / "config.json")
        generate_e2edump_json(dump_path, dump_config_path, extra_json_settings)

        try:
            os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
            net = Net()
            compare_massive_data(net, dump_path)
        finally:
            del os.environ['MINDSPORE_DUMP_CONFIG']

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_e2e_statistic_moveto():
    """
    Feature: test kernel mindspore.ops.move_to
    Description: If tensor's target device is different from running device, skip this tensor while dumping.
    Expectation: The statistics result meet the requirement.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", jit_config={"jit_level": "O0"})

    def extra_json_settings(data):
        data["e2e_dump_settings"]["stat_calc_mode"] = "device"
        data["e2e_dump_settings"]["enable"] = True

    with tempfile.TemporaryDirectory() as test_dir:
        path = Path(test_dir)
        dump_path = str(path / "debug_dump")
        dump_config_path = str(path / "config.json")
        generate_e2edump_json(dump_path, dump_config_path, extra_json_settings)

        try:
            os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
            net = MoveToNet()
            x = Tensor([1, 2, 3])
            y = net(x)
            assert (x.numpy() == y.numpy()).all()
            check_moveto_dump(dump_path)
        finally:
            del os.environ['MINDSPORE_DUMP_CONFIG']
