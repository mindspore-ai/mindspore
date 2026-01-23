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
Tests the data dump functionality for bfloat16 data type
"""
import os
import tempfile
import time
import numpy as np
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.common import dtype as mstype
from mindspore import context
import csv
from dump_test_utils import generate_statistic_dump_json
from tests.mark_utils import arg_mark
from tests.security_utils import security_off_wrap
from dump_check import SyncDumpCheck
import json

class MultiOpNetBfloat16Net(Cell):
    def construct(self, x, y):
        a = x + y
        b = a - y
        c = b * 2.0
        d = c / (x + 1.0)
        e = d.astype(mstype.float32)
        return e.astype(mstype.bfloat16)

def run_dump_bfloat16(dump_scene):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'test_dump_bfloat16')
        dump_config_path = os.path.join(tmp_dir, 'test_dump_bfloat16.json')
        if dump_scene == 'sync_dump':
            generate_statistic_dump_json(dump_path, dump_config_path, 'test_sync_kbyk_dump', 'full')
        else:
            generate_statistic_dump_json(dump_path, dump_config_path, 'test_async_kbyk_dump', 'full')
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        x_np = np.random.randn(4, 4).astype(np.float32)
        y_np = np.random.randn(4, 4).astype(np.float32)
        x = Tensor(x_np, dtype=mstype.bfloat16)
        y = Tensor(y_np, dtype=mstype.bfloat16)
        net = MultiOpNetBfloat16Net()
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
        assert dump_input.dtype == np.dtype('float32')
        assert np.allclose(dump_input, x.astype(np.float32))
        find_statistic_cmd = f'find {dump_path} -name "statistic.csv"'
        statistic_file = os.popen(find_statistic_cmd).read().strip()
        assert os.path.exists(statistic_file)
        with open(dump_config_path, 'r', encoding="utf-8") as f:
            dump_json = json.load(f)
        dump_check = SyncDumpCheck(dump_json, iteration_id_list=1)
        dump_check.dump_result_check()
        with open(statistic_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            stats = list(reader)
            bfloat16_stats = [s for s in stats if s['Data Type'] == 'bfloat16']
            assert len(bfloat16_stats) >= 1
            stat = bfloat16_stats[0]
            np_mean = np.mean(x.astype("float32").asnumpy())
            np_norm = np.linalg.norm(x.astype("float32").asnumpy())
            np_min = np.min(x.astype("float32").asnumpy())
            np_max = np.max(x.astype("float32").asnumpy())
            assert np.isclose(float(stat['Avg Value']), np_mean, rtol=1e-3, atol=1e-5)
            assert np.isclose(float(stat['L2Norm Value']), np_norm, rtol=1e-3, atol=1e-5)
            assert np.isclose(float(stat['Min Value']), np_min, rtol=1e-3, atol=1e-5)
            assert np.isclose(float(stat['Max Value']), np_max, rtol=1e-3, atol=1e-5)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
@security_off_wrap
def test_e2e_kbyk_dump_bfloat16():
    """
    Feature: e2e kbyk dump for bfloat16 data type.
    Description: Test e2e kbyk dump when the input data is with dtype bfloat16.
    Expectation: Data is expected to be dumped correctly, and the statistic file is correctly record the data type.
    """
    run_dump_bfloat16("sync_dump")

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=4.0)
@security_off_wrap
def test_async_kbyk_dump_bfloat16():
    """
    Feature: async kbyk dump for bfloat16 data type.
    Description: Test async kbyk dump when the input data is with dtype bfloat16.
    Expectation: Data is expected to be dumped correctly, and the statistic file is correctly record the data type.
    """
    run_dump_bfloat16("async_dump")
