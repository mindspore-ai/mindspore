# Copyright 2021-2024 Huawei Technologies Co., Ltd
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
import mindspore
import mindspore.context as context
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor
from tests.security_utils import security_off_wrap
from dump_test_utils import generate_dump_json, check_dump_structure

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
        if test_name == "test_e2e_dump_trans_true_op_debug_mode":
            tensor = Tensor(np.full((1, 3, 3, 3), 65504, dtype=np.float16), mindspore.float16)
            weight = Tensor(np.full((3, 3, 1, 1), 65504, dtype=np.float16), mindspore.float16)
        net = ConvNet()
        net(tensor, weight)
        if test_name == "test_e2e_dump_trans_true_op_debug_mode":
            check_dump_structure(dump_path, dump_config_path, 1, 0, 1)
        dump_data_path = os.path.join(dump_path, 'rank_0', 'Net', '0', '0')
        assert os.path.exists(dump_data_path)
        if test_name == "test_e2e_dump_trans_true_op_debug_mode":
            # tensor data in host format.
            output_name = "Conv2D.Default_Conv2D-op*.output.0.DefaultFormat.npy"
            output_path = glob.glob(os.path.join(dump_data_path, output_name))[0]
            real_path = os.path.realpath(output_path)
            output = np.load(real_path)
            assert output.shape == (20,)
        del os.environ['MINDSPORE_DUMP_CONFIG']


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_ascend_kernel_by_kernel_dump_sample():
    """
    Feature: Ascend kernel by kernel dump.
    Description: Test kernel by kernel dump in Ascend with trans_flag is configured to true.
    Expectation: Dump files has tensor data in host format (4 dimensions).
    """
    context.set_context(jit_level='O0')
    os.environ['INF_NAN_MODE_ENABLE'] = "1"
    os.environ['MS_ASCEND_CHECK_OVERFLOW_MODE'] = "INFNAN_MODE"
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_trans_flag("test_e2e_dump_sample_debug_mode")
    del os.environ['INF_NAN_MODE_ENABLE']
    del os.environ['MS_ASCEND_CHECK_OVERFLOW_MODE']
