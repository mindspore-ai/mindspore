# Copyright 2023 Huawei Technologies Co., Ltd
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
import time
import shutil
import pytest
import numpy as np
from dump_test_utils import generate_dump_json, check_dump_structure


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
def test_dump_hccl():
    """
    Feature: Test async dump for hccl operator.
    Description: Set dump envs for async dump, run AllReduce script on 8 cards ascend computor.
    Expectation: The AllReduce data is saved and the value is correct.
    """
    if sys.platform != 'linux':
        return
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'test_dump_hccl')
        dump_config_path = os.path.join(tmp_dir, 'test_dump_hccl.json')
        generate_dump_json(dump_path, dump_config_path, 'test_async_dump_npy')
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        dump_file_path = os.path.join(dump_path, 'rank_0', 'Net', '0', '0')
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        exec_network_cmd = 'cd {0}; bash run_allreduce.sh'.format(os.path.split(os.path.realpath(__file__))[0])
        ret = os.system(exec_network_cmd)
        print("ret of exec_network_cmd: ", ret)
        for _ in range(3):
            if not os.path.exists(dump_file_path):
                time.sleep(2)
        check_dump_structure(dump_path, dump_config_path, 8, 1, 1)
        output_name = "AllReduce.AllReduce-op*.*.*.*.output.0.*.npy"
        # Check data in 8 cards.
        for i in range(8):
            dump_file_path = os.path.join(dump_path, 'rank_{}'.format(i), 'Net', '0', '0')
            output_path = glob.glob(os.path.join(dump_file_path, output_name))[0]
            real_path = os.path.realpath(output_path)
            output = np.load(real_path)
            expect = [[28]]
            assert output.shape == (1, 1)
            assert np.array_equal(output, expect)
        del os.environ['MINDSPORE_DUMP_CONFIG']
