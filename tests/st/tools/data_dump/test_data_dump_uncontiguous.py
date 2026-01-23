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
Tests the data dump functionality for non-contiguous data
"""

import os
import tempfile
import time
import numpy as np
import mindspore
from mindspore.nn import Cell
from mindspore import context
from mindspore import ops
import csv
from dump_test_utils import generate_statistic_dump_json
from tests.mark_utils import arg_mark
from tests.security_utils import security_off_wrap

class ViewNet(Cell):
    def __init__(self):
        super().__init__()
        self.transpose = ops.TransposeView()

    def construct(self, x, perm):
        out = self.transpose(x, perm)
        return out

def run_dump_bfloat16_uncontiguous(dump_scene):
    """Run e2e dump on non-contiguous tensor with type bfloat16"""
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'test_dump_bfloat16')
        dump_config_path = os.path.join(tmp_dir, 'test_dump_bfloat16.json')
        if dump_scene == 'sync_dump':
            generate_statistic_dump_json(dump_path, dump_config_path, 'test_sync_kbyk_dump', 'full')
        else:
            generate_statistic_dump_json(dump_path, dump_config_path, 'test_async_kbyk_dump', 'full')
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        origin_x = mindspore.Tensor(np.arange(5*10*8).reshape(5, 10, 8), dtype=mindspore.bfloat16)
        begin = (1, 3, 2)
        end = (3, 5, 6)
        strides = (1, 1, 2)
        strided_slice = ops.StridedSlice()
        input_x = strided_slice(origin_x, begin, end, strides)
        perm = (1, 2, 0)
        net = ViewNet()
        expect = net(input_x, perm)
        for _ in range(3):
            if not os.path.exists(dump_path):
                time.sleep(2)
        find_output_cmd = f'find {dump_path} -name "TransposeView*.*.output.*.npy"'
        output_file_list = os.popen(find_output_cmd).read().strip().split('\n')
        output_file_list = [f for f in output_file_list if f]
        assert any(os.path.exists(f) for f in output_file_list)
        dump_output = np.load(output_file_list[0])
        assert dump_output.shape == (2, 2, 2)
        assert dump_output.dtype == np.dtype('float32')
        assert np.allclose(dump_output, expect.astype(np.float32))
        find_statistic_cmd = f'find {dump_path} -name "statistic.csv"'
        statistic_file = os.popen(find_statistic_cmd).read().strip()
        assert os.path.exists(statistic_file)
        with open(statistic_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            stats = list(reader)
            bfloat16_stats = [s for s in stats if s['Data Type'] == 'bfloat16']
            assert len(bfloat16_stats) >= 1
            stat = bfloat16_stats[0]
            np_mean = np.mean(input_x.astype("float32").asnumpy())
            np_norm = np.linalg.norm(input_x.astype("float32").asnumpy())
            np_min = np.min(input_x.astype("float32").asnumpy())
            np_max = np.max(input_x.astype("float32").asnumpy())
            assert np.isclose(float(stat['Avg Value']), np_mean, rtol=1e-3, atol=1e-5)
            assert np.isclose(float(stat['L2Norm Value']), np_norm, rtol=1e-3, atol=1e-5)
            assert np.isclose(float(stat['Min Value']), np_min, rtol=1e-3, atol=1e-5)
            assert np.isclose(float(stat['Max Value']), np_max, rtol=1e-3, atol=1e-5)

def run_dump_int16_uncontiguous(dump_scene):
    """Run e2e dump on non-contiguous tensor with type int16"""
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'test_dump_int16')
        dump_config_path = os.path.join(tmp_dir, 'test_dump_int16.json')
        if dump_scene == 'sync_dump':
            generate_statistic_dump_json(dump_path, dump_config_path, 'test_sync_kbyk_dump', 'full')
        else:
            generate_statistic_dump_json(dump_path, dump_config_path, 'test_async_kbyk_dump', 'full')
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        origin_x = mindspore.Tensor(np.arange(5*10*8).reshape(5, 10, 8), dtype=mindspore.int16)
        begin = (1, 3, 2)
        end = (3, 5, 6)
        strides = (1, 1, 2)
        strided_slice = ops.StridedSlice()
        input_x = strided_slice(origin_x, begin, end, strides)
        perm = (1, 2, 0)
        net = ViewNet()
        expect = net(input_x, perm)
        for _ in range(3):
            if not os.path.exists(dump_path):
                time.sleep(2)
        find_output_cmd = f'find {dump_path} -name "TransposeView*.*.output.*.npy"'
        output_file_list = os.popen(find_output_cmd).read().strip().split('\n')
        output_file_list = [f for f in output_file_list if f]
        assert any(os.path.exists(f) for f in output_file_list)
        dump_output = np.load(output_file_list[0])
        assert dump_output.shape == (2, 2, 2)
        assert dump_output.dtype == np.dtype('int16')
        assert np.allclose(dump_output, expect)
        find_statistic_cmd = f'find {dump_path} -name "statistic.csv"'
        statistic_file = os.popen(find_statistic_cmd).read().strip()
        assert os.path.exists(statistic_file)
        with open(statistic_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            stats = list(reader)
            int16_stats = [s for s in stats if s['Data Type'] == 'int16']
            assert len(int16_stats) >= 1
            stat = int16_stats[0]
            np_mean = np.mean(input_x.asnumpy())
            np_norm = np.linalg.norm(input_x.asnumpy())
            np_min = np.min(input_x.asnumpy())
            np_max = np.max(input_x.asnumpy())
            assert np.isclose(float(stat['Avg Value']), np_mean, rtol=1e-3, atol=1e-5)
            assert np.isclose(float(stat['L2Norm Value']), np_norm, rtol=1e-3, atol=1e-5)
            assert np.isclose(float(stat['Min Value']), np_min, rtol=1e-3, atol=1e-5)
            assert np.isclose(float(stat['Max Value']), np_max, rtol=1e-3, atol=1e-5)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
@security_off_wrap
def test_e2e_kbyk_dump_bfloat16_uncontiguous():
    """
    Feature: e2e kbyk dump for bfloat16 data type.
    Description: Test e2e kbyk dump when the input data is with dtype bfloat16.
    Expectation: Data is expected to be dumped correctly, and the statistic file is correctly record the data type.
    """
    run_dump_bfloat16_uncontiguous("sync_dump")

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
@security_off_wrap
def test_async_kbyk_dump_bfloat16_uncontiguous():
    """
    Feature: async kbyk dump for bfloat16 data type.
    Description: Test async kbyk dump when the input data is with dtype bfloat16.
    Expectation: Data is expected to be dumped correctly, and the statistic file is correctly record the data type.
    """
    run_dump_bfloat16_uncontiguous("async_dump")

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
@security_off_wrap
def test_e2e_kbyk_dump_int16_uncontiguous():
    """
    Feature: e2e kbyk dump for int16 data type.
    Description: Test e2e kbyk dump when the input data is with dtype int16.
    Expectation: Data is expected to be dumped correctly, and the statistic file is correctly record the data type.
    """
    run_dump_int16_uncontiguous("sync_dump")

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=4.0)
@security_off_wrap
def test_async_kbyk_dump_int16_uncontiguous():
    """
    Feature: async kbyk dump for int16 data type.
    Description: Test async kbyk dump when the input data is with dtype int16.
    Expectation: Data is expected to be dumped correctly, and the statistic file is correctly record the data type.
    """
    run_dump_int16_uncontiguous("async_dump")
