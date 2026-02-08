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
"""resnet e2e async dump test case."""
import os
import sys
import tempfile
import glob
import shutil
import numpy as np
import json

from mindspore import context
from mindspore.common.tensor import Tensor
from tests.mark_utils import arg_mark
from tests.security_utils import security_off_wrap
from tests.st.networks.models.resnet50.src.resnet import resnet50
from dump_test_utils import check_dump_structure

e2e_async_dump_json = {
    "common_dump_settings": {
        "dump_mode": 0,
        "path": "",
        "net_name": "Net",
        "iteration": "0",
        "input_output": 0,
        "kernels": ["Default/Conv-op12"],
        "support_device": [0, 1, 2, 3, 4, 5, 6, 7],
        "op_debug_mode": 0
    },
    "e2e_dump_settings": {
        "enable": False,
        "trans_flag": True
    }
}

def generate_dump_json(dump_path, json_file_name):
    json_data = e2e_async_dump_json
    json_data["common_dump_settings"]["path"] = dump_path
    with open(json_file_name, 'w', encoding="utf-8") as f:
        json.dump(json_data, f)

def run_trans_flag(test_name):
    if sys.platform != 'linux':
        return
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, test_name)
        dump_config_path = os.path.join(tmp_dir, '{}.json'.format(test_name))
        generate_dump_json(dump_path, dump_config_path)
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        net = resnet50()
        predict = Tensor(np.ones([1, 3, 224, 224]).astype(np.float32) * 0.01)
        expect = net(predict)
        check_dump_structure(dump_path, dump_config_path, 1, 0, 1)
        dump_data_path = os.path.join(dump_path, 'rank_0', 'Net', '0', '0')
        assert os.path.exists(dump_data_path)
        if test_name == "test_e2e_dump_kbk_resnet":
            output_name = "BiasAdd.Default_end_point-Dense_BiasAdd-op0.0.0.*.output.0.DefaultFormat.*.npy"
            output_path = glob.glob(os.path.join(dump_data_path, output_name))[0]
            real_path = os.path.realpath(output_path)
            output = np.load(real_path)
            assert output.shape == (1, 10)
            assert np.array_equal(output, expect)
        elif test_name == "test_e2e_dump_dvm_resnet":
            output_name = "GraphKernel_BiasAdd.Default_GraphKernel_BiasAdd-op0.0.0.*.output.0.DefaultFormat.*.npy"
            output_path = glob.glob(os.path.join(dump_data_path, output_name))[0]
            real_path = os.path.realpath(output_path)
            output = np.load(real_path)
            assert output.shape == (1, 10)
            assert np.array_equal(output, expect)
        del os.environ['MINDSPORE_DUMP_CONFIG']


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=106.0)
@security_off_wrap
def test_ascend_kernel_by_kernel_resnet():
    """
    Feature: Ascend kernel by kernel dump with resnet50.
    Description: Test kernel by kernel dump in Ascend with trans_flag is configured to true.
    Expectation: Dump files has tensor data in host format (4 dimensions).
    """
    context.set_context(jit_level='O0')
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_trans_flag("test_e2e_dump_kbk_resnet")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=106.0)
@security_off_wrap
def test_ascend_dvm_resnet():
    """
    Feature: Ascend dvm dump with resnet50.
    Description: Test kernel by kernel dump in Ascend with trans_flag is configured to true.
    Expectation: Dump files has tensor data in host format (4 dimensions).
    """
    context.set_context(jit_level='O1')
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_trans_flag("test_e2e_dump_dvm_resnet")
