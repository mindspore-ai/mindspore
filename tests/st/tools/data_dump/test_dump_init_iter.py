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
import tempfile
import time
import json

from mindspore import Tensor, nn, _c_expression, context
from pathlib import Path
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    def construct(self, x, y):
        return x + y


def generate_e2edump_json(dump_path, json_file_name, extra_settings_func=None):
    current_dir = Path(__file__).parent
    json_path = current_dir / "test_e2e_statistic_config.json"
    with open(json_path, 'r') as file:
        data = json.load(file)
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


def compare_single_data(x, y, net, dump_path):
    t_x, t_y = x, y
    t_out = x + y
    t_x, t_y, t_out = t_x.astype(np.float32), t_y.astype(
        np.float32), t_out.astype(np.float32)

    common_res = {'Op Type': 'Add', 'Op Name': 'Default/Add-op0', 'Data Size': str(
        x.nbytes), 'Data Type': str(x.dtype), 'Shape': "[3]"}
    target_list = []
    for idx, tensor in enumerate([t_x, t_y]):
        target = {**common_res, **{'IO': 'input', 'Slot': str(idx)}}
        target.update({
            'Max Value': tensor.max(), 'Min Value': tensor.min(),
            'Avg Value': tensor.mean(), 'L2Norm Value': np.linalg.norm(tensor)
        })
        target_list.append(target)
    target_output = {**common_res, **{'IO': 'output', 'Slot': '0', 'Max Value': t_out.max(),
                                      'Min Value': t_out.min(), 'Avg Value': t_out.mean(),
                                      'L2Norm Value': np.linalg.norm(t_out)}}
    target_list.append(target_output)
    t = net(Tensor(x), Tensor(y))
    print(t)
    time.sleep(1)
    stat_list = get_dumped_stat_list(dump_path)
    assert len(stat_list) == 3
    check_statistic_result(stat_list, target_list)


def compare_multi_data(net, dtype, dump_path, init_iter=0):
    test_cases = [
        (np.array([1., 2., 3.], dtype), np.array([2., 2., -10.], dtype)),
        (np.array([3., 2., 3.], dtype), np.array([1., 2., -10.], dtype)),
    ]
    continue_iter = 100
    for i, (x, y) in enumerate(test_cases):
        if i == 0:
            compare_single_data(x, y, net, Path(dump_path) /
                                "rank_0" / "Net" / "0" / str(init_iter))
        else:
            # pylint: disable=W0212
            _c_expression._set_init_iter(continue_iter) # pylint: disable=W0212
            compare_single_data(x, y, net, Path(dump_path) /
                                "rank_0" / "Net" / "0" / str(continue_iter))


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=2048.01)
def test_dump_init_iter():
    """
    Feature: dump config initial_iteration
    Description: Test dump config initial_iteration.
    Expectation: The iteration result meet the requirement.
    """
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend", jit_config={"jit_level": "O0"})
    initial_iteration = 10

    def extra_json_settings(data):
        data["e2e_dump_settings"]["stat_calc_mode"] = "host"
        data["e2e_dump_settings"]["enable"] = True
        data["common_dump_settings"]["initial_iteration"] = initial_iteration

    with tempfile.TemporaryDirectory() as test_dir:
        path = Path(test_dir)
        dump_path = str(path / "dump_data")
        dump_config_path = str(path / "config.json")
        generate_e2edump_json(dump_path, dump_config_path, extra_json_settings)

        try:
            os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
            net = Net()
            compare_multi_data(net, np.float16, dump_path, initial_iteration)
        finally:
            del os.environ['MINDSPORE_DUMP_CONFIG']


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=2048.0)
def test_dump_user_step():
    """
    Feature: Update steps through Calling functions
    Description: Call enable func and update steps func
    Expectation: iteration_id updates according to update steps func
    """
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend", jit_config={"jit_level": "O0"})

    init_iter = 5
    user_step = 2

    def extra_json_settings(data):
        data["e2e_dump_settings"]["stat_calc_mode"] = "host"
        data["common_dump_settings"]["saved_data"] = "full"

    with tempfile.TemporaryDirectory() as test_dir:
        path = Path(test_dir)
        dump_path = str(path / "dump_data")
        dump_config_path = str(path / "config.json")
        generate_e2edump_json(dump_path, dump_config_path, extra_json_settings)
        try:
            os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
            net = Net()
            # pylint: disable=W0212
            _c_expression._set_init_iter(init_iter)
            test_cases = [
                (np.array([1., 2., 3.]), np.array([2., 2., -10.])),
                (np.array([3., 2., 3.]), np.array([1., 2., -10.])),
                (np.array([4., 2., 3.]), np.array([4., 2., -10.])),
            ]
            for i, (x, y) in enumerate(test_cases):
                net(Tensor(x), Tensor(y))
                # pylint: disable=W0212
                _c_expression._dump_step(user_step)
                real_dump_path = os.path.join(dump_path, 'rank_0', 'Net', '0', str(user_step * i + init_iter))
                assert os.path.exists(os.path.join(real_dump_path, 'statistic.csv'))
                files = os.listdir(real_dump_path)
                npy_cnt = 0
                for f in files:
                    if f.endswith('.npy'):
                        npy_cnt += 1
                assert npy_cnt == 3
        finally:
            del os.environ['MINDSPORE_DUMP_CONFIG']
