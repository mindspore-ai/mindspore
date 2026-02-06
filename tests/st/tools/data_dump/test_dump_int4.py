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
from mindspore import Tensor, Parameter
from mindspore.ops.auto_generate import WeightQuantBatchMatmul
from mindspore.common import dtype as mstype
import mindspore.context as context
from mindspore.nn import Cell
import csv
from dump_test_utils import generate_statistic_dump_json
from tests.mark_utils import arg_mark
from tests.security_utils import security_off_wrap


class WeightQuantBatchMatmulNet(Cell):
    """
    WeightQuantBatchMatmulNet.
    """

    def __init__(self, weight=None, transpose_x=False, transpose_weight=False, antiquant_group_size=0):
        super().__init__()
        self.wqbmm = WeightQuantBatchMatmul(transpose_x, transpose_weight, antiquant_group_size)
        self.weight = weight

    def construct(self, x, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias):
        out = self.wqbmm(x, self.weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias)
        return out


def np_antiquant(np_data, scale=1.0, offset=0.):
    """mindspore implemented antiquant"""
    np_antiquant_data = np_data.astype(np.float16)
    if offset is None:
        offset = 0
    np_antiquant_data = scale * (np_antiquant_data - offset)
    np_antiquant_data = np_antiquant_data.astype(np.float16)
    return np_antiquant_data


def np_int4data_pack_to_int8(np_data):
    """pack int4(represented in int8) data to int8(int4*2)"""
    np_data = np_data.astype(np.int8)
    np_data &= 0x000F
    np_data[::, 0::2] <<= 0
    np_data[::, 1::2] <<= 4
    np_int4_data = np_data[::, 0::2] | np_data[::, 1::2]
    return np_int4_data


def np_quant_int4(np_data, scale=1.0, offset=0.0):
    """quant data to int4 data"""
    np_quant_int8_data = np.round(np_data / scale + offset).astype(np.int8)
    np_quant_int8_data = np.clip(np_quant_int8_data, -8, 7).astype(np.int8)
    np_quant_int4_data = np_int4data_pack_to_int8(np_quant_int8_data)
    return np_quant_int8_data, np_quant_int4_data


def np_gen_int4_data(scale, offset=0.):
    """
    gen fp16_activation and int4_weight for test
    :param scale: scale for quant
    :return: activation with dtype fp16, weight width dtype int4
    """
    np_x = np.random.rand(8, 8).astype(np.float16)
    np_weight = np.linspace(-0.64, 0.64, 64).astype(np.float16).reshape((8, 8))
    np_quant_int8_data, np_quant_int4_data = np_quant_int4(np_weight, scale, offset)
    return np_x, np_quant_int8_data, np_quant_int4_data


def run_dump_int4(dump_scene):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_level="O1")
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'test_dump_int4')
        dump_config_path = os.path.join(tmp_dir, 'test_dump_int4.json')
        if dump_scene == 'e2e_dump':
            generate_statistic_dump_json(dump_path, dump_config_path, 'test_e2e_dump', 'full')
        else:
            generate_statistic_dump_json(dump_path, dump_config_path, 'test_e2e_async_dump', 'full')
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        scale = 0.1
        offset = 4
        np_x, np_int8_data, np_int4_weight = np_gen_int4_data(scale, offset)
        ms_int4_weight = Parameter(Tensor(np_int4_weight, dtype=mstype.qint4x2))
        antiquant_scale = Tensor([scale], dtype=mstype.float16)
        antiquant_offset = Tensor([-offset], dtype=mstype.float16)
        quant_scale = None
        quant_offset = None
        bias = None
        wqbm_net = WeightQuantBatchMatmulNet(ms_int4_weight)
        x = Tensor(np_x, dtype=mstype.float16)
        _ = wqbm_net(x, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias)
        for _ in range(3):
            if not os.path.exists(dump_path):
                time.sleep(2)
        find_int4_input_cmd = 'find {0} -name "WeightQuantBatchMatmul*.input.1.*.npy"'.format(dump_path)
        int4_input_file_path = os.popen(find_int4_input_cmd).read()
        int4_input_file_path = int4_input_file_path.replace('\n', '')
        dump_int4 = np.load(int4_input_file_path)
        assert dump_int4.shape == (8, 8)
        expect_data = np_int8_data.flatten()[:dump_int4.size].reshape(dump_int4.shape)
        np.testing.assert_allclose(dump_int4, expect_data, rtol=1e-3)
        find_statistic_cmd = 'find {0} -name "statistic.csv"'.format(dump_path)
        statistic_file = os.popen(find_statistic_cmd).read()
        statistic_file = statistic_file.replace('\n', '')
        with open(statistic_file) as f:
            reader = csv.DictReader(f)
            stats = list(reader)

            def get_int4_data(statistic):
                return statistic['Data Type'] == 'int4'

            int4_statistics = list(filter(get_int4_data, stats))
            int4_num = len(int4_statistics)
            assert int4_num == 1
            for statistic_item in int4_statistics:
                assert statistic_item['Max Value'] == str(expect_data.max())
                assert statistic_item['Min Value'] == str(expect_data.min())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=20.0)
@security_off_wrap
def test_e2e_dump_int4():
    """
    Feature: e2e dump for int4x2 data type.
    Description: Test e2e dump when the input data is with dtype int4.
    Expectation: Data is expected to be dumped correctly, and the statistic file is correctly record
     the data type and  the statistic items.
    """
    run_dump_int4("e2e_dump")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=20.0)
@security_off_wrap
def test_e2e_async_dump_int4():
    """
    Feature: e2e async dump for int4x2 data type.
    Description: Test e2e async dump when the input data is with dtype int4.
    Expectation: Data is expected to be dumped correctly, and the statistic file is correctly record
     the data type and  the statistic items.
    """
    run_dump_int4("e2e_async_dump")
