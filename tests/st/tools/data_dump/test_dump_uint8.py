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
import tempfile
import time
import numpy as np
from mindspore import Tensor
from mindspore.nn import Cell, ReLU
from mindspore.common import dtype as mstype
import mindspore.context as context
import csv
from dump_test_utils import generate_statistic_dump_json
from tests.mark_utils import arg_mark
from tests.security_utils import security_off_wrap

class MultiOpNetUint8Net(Cell):
    def __init__(self):
        super().__init__()
        self.relu = ReLU()
    def construct(self, x, y):
        a = x + y
        b = a * y
        c = b - x
        d = c / (y + 1)
        e = self.relu(d)
        return e

def run_dump_uint8(dump_scene):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'test_dump_uint8')
        dump_config_path = os.path.join(tmp_dir, 'test_dump_uint8.json')
        if dump_scene == 'sync_dump':
            generate_statistic_dump_json(dump_path, dump_config_path, 'test_sync_kbyk_dump', 'full')
        else:
            generate_statistic_dump_json(dump_path, dump_config_path, 'test_async_kbyk_dump', 'full')
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        x_np = np.random.randint(1, 100, (4, 4), dtype=np.uint8)
        y_np = np.random.randint(1, 100, (4, 4), dtype=np.uint8)
        x = Tensor(x_np, dtype=mstype.uint8)
        y = Tensor(y_np, dtype=mstype.uint8)
        net = MultiOpNetUint8Net()
        _ = net(x, y)
        for _ in range(3):
            if not os.path.exists(dump_path):
                time.sleep(2)
        find_input_cmd = f'find {dump_path} -name "Add*.*.input.*.npy"'
        input_file_list = os.popen(find_input_cmd).read().strip().split('\n')
        input_file_list = [f for f in input_file_list if f]
        assert any(os.path.exists(f) for f in input_file_list)
        dump_input = np.load(input_file_list[1])
        assert dump_input.shape == (4, 4)
        assert dump_input.dtype == np.uint8
        assert np.allclose(dump_input, y_np)
        find_statistic_cmd = f'find {dump_path} -name "statistic.csv"'
        statistic_file = os.popen(find_statistic_cmd).read().strip()
        assert os.path.exists(statistic_file)
        with open(statistic_file) as f:
            reader = csv.DictReader(f)
            stats = list(reader)
            uint8_stats = [s for s in stats if s['Data Type'] == 'uint8']
            assert len(uint8_stats) >= 1
            stat = uint8_stats[0]
            np_mean = np.mean(x_np)
            np_norm = np.linalg.norm(x_np)
            np_min = np.min(x_np)
            np_max = np.max(x_np)
            assert np.isclose(float(stat['Avg Value']), np_mean, rtol=1e-3, atol=1e-5)
            assert np.isclose(float(stat['L2Norm Value']), np_norm, rtol=1e-3, atol=1e-5)
            assert np.isclose(float(stat['Min Value']), np_min, rtol=1e-3, atol=1e-5)
            assert np.isclose(float(stat['Max Value']), np_max, rtol=1e-3, atol=1e-5)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
@security_off_wrap
def test_e2e_kbyk_dump_uint8():
    """
    Feature: e2e kbyk dump for uint8 data type.
    Description: Test e2e kbyk dump when the input data is with dtype uint8.
    Expectation: Data is expected to be dumped correctly, and the statistic file is correctly record the data type.
    """
    run_dump_uint8("sync_dump")

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=4.0)
@security_off_wrap
def test_async_kbyk_dump_uint8():
    """
    Feature: async kbyk dump for uint8 data type.
    Description: Test async kbyk dump when the input data is with dtype uint8.
    Expectation: Data is expected to be dumped correctly, and the statistic file is correctly record the data type.
    """
    run_dump_uint8("async_dump")
