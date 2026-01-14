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
Tests the resnet sync statistic dump with default statistic_category settings.
"""
import os
import tempfile
import time
import numpy as np
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore import context
import csv
from dump_test_utils import generate_statistic_dump_json
from tests.mark_utils import arg_mark
from tests.security_utils import security_off_wrap
from tests.st.networks.models.resnet50.src.resnet import resnet50

def run_statistic_dump_resnet():
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'run_statistic_dump_resnet')
        dump_config_path = os.path.join(tmp_dir, 'run_statistic_dump_resnet.json')
        generate_statistic_dump_json(dump_path, dump_config_path, 'test_sync_kbyk_dump_resnet', 'statistic')
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        x_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
        x = Tensor(x_np, dtype=mstype.float32)
        net = resnet50()
        net(x)
        for _ in range(3):
            if not os.path.exists(dump_path):
                time.sleep(2)
        find_statistic_cmd = f'find {dump_path} -name "statistic.csv"'
        statistic_file = os.popen(find_statistic_cmd).read().strip()
        assert os.path.exists(statistic_file)
        with open(statistic_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            stats = list(reader)
            float32_stats = [s for s in stats if s['Data Type'] == 'float32']
            assert len(float32_stats) >= 1
            stat = float32_stats[0]
            np_norm = np.linalg.norm(x.astype("float32").asnumpy())
            np_min = np.min(x.astype("float32").asnumpy())
            np_max = np.max(x.astype("float32").asnumpy())
            assert np.isclose(float(stat['L2Norm Value']), np_norm, rtol=1e-3, atol=1e-5)
            assert np.isclose(float(stat['Min Value']), np_min, rtol=1e-3, atol=1e-5)
            assert np.isclose(float(stat['Max Value']), np_max, rtol=1e-3, atol=1e-5)
        del os.environ['MINDSPORE_DUMP_CONFIG']

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=106.0)
@security_off_wrap
def test_e2e_sync_kbyk_dump_resnet():
    """
    Feature: e2e kbyk statistic dump with default statistic_category settings for float32 data type.
    Description: Test e2e kbyk dump when the statistic_category is set to default.
    Expectation: Data is expected to be dumped correctly, and the statistic file is correctly record the data type.
    """
    context.set_context(jit_level='O0')
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_statistic_dump_resnet()

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=106.0)
@security_off_wrap
def test_e2e_sync_dvm_dump_resnet():
    """
    Feature: e2e dvm statistic dump with default statistic_category settings for float32 data type.
    Description: Test e2e dvm dump when the statistic_category is set to default.
    Expectation: Data is expected to be dumped correctly, and the statistic file is correctly record the data type.
    """
    context.set_context(jit_level='O1')
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_statistic_dump_resnet()
