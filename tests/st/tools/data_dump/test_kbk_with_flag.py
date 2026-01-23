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
import shutil
import numpy as np
import json
import pandas as pd

import mindspore.nn as nn
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore import log as logger
from tests.mark_utils import arg_mark
from tests.security_utils import security_off_wrap
from dump_test_utils import check_dump_structure

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.dump = P.TensorDump()
        self.dump.add_prim_attr("td_flag", True)

    def construct(self, x):
        x += 1.
        self.dump('add', x)
        x *= 5.
        return x

kbk_dump_config = {
    "common_dump_settings": {
        "dump_mode": 1,
        "path": "",
        "net_name": "Net",
        "iteration": "0",
        "saved_data": "statistic",
        "input_output": 0,
        "kernels": ["TensorDump"],
        "support_device": [0, 1, 2, 3, 4, 5, 6, 7],
        "op_debug_mode": 0
    },
    "e2e_dump_settings": {
        "enable": False,
        "trans_flag": True
    }
}


def generate_dump_json(dump_path, json_file_name):
    json_data = kbk_dump_config
    json_data["common_dump_settings"]["path"] = dump_path
    with open(json_file_name, 'w') as f:
        json.dump(json_data, f)


def remove_trailing_commas(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        new_lines = []
        for line in lines:
            line = line.rstrip()
            line = line.rstrip(',')
            new_lines.append(line + '\n')

        new_filename = "statistic_new.csv"
        new_filename = os.path.join(os.path.dirname(filename), new_filename)
        with open(new_filename, 'w', encoding='utf-8') as file:
            file.writelines(new_lines)

        df = pd.read_csv(new_filename)
        return df
    except FileNotFoundError:
        logger.error(f"File {filename} not found.")
        return None
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file {filename}: {e}")
        return None
    except pd.errors.EmptyDataError as e:
        logger.error(f"CSV file {filename} is empty: {e}")
        return None
    except PermissionError:
        logger.error(f"Permission denied for file {filename}.")
        return None


def run_trans_flag(test_name):
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, test_name)
        dump_config_path = os.path.join(tmp_dir, '{}.json'.format(test_name))
        generate_dump_json(dump_path, dump_config_path)
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        os.environ["MS_KERNEL_LAUNCH_SKIP"] = "TensorDump"
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).astype(np.float32)
        input_x = Tensor(x)
        net = Net()
        net(input_x)
        check_dump_structure(dump_path, dump_config_path, 1, 0, 1)
        dump_data_path = os.path.join(dump_path, 'rank_0', 'Net', '0', '0')
        if test_name == "test_ascend_kbk_with_flag":
            target_file = 'statistic.csv'
            csv_path = os.path.join(dump_data_path, target_file)
            assert os.path.exists(csv_path)
            df = remove_trailing_commas(csv_path)
            op_name = df['Op Name'].iloc[0]
            expect_op_name = "add|Default/TensorDump-op0"
            assert op_name == expect_op_name
        del os.environ['MINDSPORE_DUMP_CONFIG']
        del os.environ['MS_KERNEL_LAUNCH_SKIP']


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=2048.0)
@security_off_wrap
def test_ascend_kbk_with_flag():
    """
    Feature: Ascend kernel by kernel dump with td_flag.
    Description: Ascend kernel by kernel dump with td_flag.
    Expectation: The 'Op Name' field of the TensorDump operator contains its first parameter.
    """
    context.set_context(jit_level='O0')
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_trans_flag("test_ascend_kbk_with_flag")
