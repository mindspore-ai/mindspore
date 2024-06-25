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
import json
import mindspore.context as context
import tempfile
import time

import mindspore
from mindspore import JitConfig, Tensor, nn
from mindspore.ops import operations as P
from pathlib import Path
import numpy as np
from tests.mark_utils import arg_mark
from dump_test_utils import generate_dump_json


def check_kernel_args_dump(dump_file_path):
    output_name = "MatMul.*.json"
    output_path = glob.glob(os.path.join(dump_file_path, output_name))[0]
    real_path = os.path.realpath(output_path)
    with open(real_path, 'r') as f:
        net_args = json.load(f)
    assert net_args.get("transpose_a") == "True"
    assert net_args.get("transpose_b") == "False"


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_e2e_dump_save_kernel_args_true():
    """
    Feature: kbyk dump support kernel args.
    Description: Test kbyk dump kernel args on device.
    Expectation: dump real kernel args.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_dir = tempfile.TemporaryDirectory(suffix="save_kernel_args")

    path = Path(test_dir.name)
    dump_path = str(path / "dump_data")
    dump_config_path = str(path / "config.json")

    generate_dump_json(dump_path, dump_config_path, "test_e2e_dump_save_kernel_args_true", "Net")
    os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
    try:
        class Net(nn.Cell):
            def __init__(self, transpose_a=False, transpose_b=False):
                super(Net, self).__init__()
                self.matmul = P.MatMul(transpose_a, transpose_b)

            def construct(self, x, y):
                return self.matmul(x, y)

        jit_config = JitConfig(jit_level="O0")
        net = Net(transpose_a=True)
        net.set_jit_config(jit_config)
        x = Tensor(np.ones(shape=[3, 3]), mindspore.float32)
        y = Tensor(np.ones(shape=[3, 4]), mindspore.float32)
        _ = net(x, y)
        time.sleep(2)
        check_kernel_args_dump(path / "dump_data" / "rank_0" / "Net" / "0" / "0")
    finally:
        del os.environ['MINDSPORE_DUMP_CONFIG']
