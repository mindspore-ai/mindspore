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
import shutil
import glob
import numpy as np
import csv
import pytest
import mindspore.context as context

import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore import dataset as ds
from mindspore.train import Model
from dump_test_utils import generate_dump_json, check_dump_structure, generate_statistic_dump_json
from tests.security_utils import security_off_wrap


def dataset_generator():
    for i in range(1, 10):
        yield np.ones((32, 2 * i), dtype=np.float32), np.ones((32, 2 * i), dtype=np.float32)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.Add()
        self.shape = P.Shape()
        self.reshape = P.Reshape()

    def construct(self, x_, y_):
        val = self.add(x_, y_)
        size = self.shape(val)
        res = self.reshape(val, size)
        return res


def run_trans_flag(test_name):
    if sys.platform != 'linux':
        return
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, test_name)
        dump_config_path = os.path.join(tmp_dir, '{}.json'.format(test_name))
        if test_name == "test_e2e_dump_dynamic_shape_custom_statistic":
            generate_statistic_dump_json(dump_path, dump_config_path, test_name, saved_data="statistic",
                                         statistic_category=["l2norm", "max"])
        else:
            generate_dump_json(dump_path, dump_config_path, test_name)
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        network = Net()
        dataset = ds.GeneratorDataset(dataset_generator, ['data1', 'data2'])
        t0 = Tensor(dtype=mindspore.float32, shape=[32, None])
        t1 = Tensor(dtype=mindspore.float32, shape=[32, None])
        network.set_inputs(t0, t1)
        model = Model(network)
        model.train(10, dataset, dataset_sink_mode=True)
        check_dump_structure(dump_path, dump_config_path, 1, 0, 1)
        dump_data_path = os.path.join(dump_path, 'rank_0', 'Net', '1', '0')
        assert os.path.exists(dump_data_path)
        if test_name == "test_e2e_dump_dynamic_shape":
            output_name = "Add.Default_network-Net_Add-op0.0.0.*.output.0.DefaultFormat.npy"
            output_path = glob.glob(os.path.join(dump_data_path, output_name))[0]
            real_path = os.path.realpath(output_path)
            output = np.load(real_path)
            assert output.shape == (32, 2)
        if test_name == "test_e2e_dump_dynamic_shape_custom_statistic":
            statistic_file_name = "statistic.csv"
            output_path = glob.glob(os.path.join(dump_data_path, statistic_file_name))[0]
            real_path = os.path.realpath(output_path)
            with open(real_path) as f:
                reader = csv.DictReader(f)
                stats = list(reader)
                def get_add_node(statistic):
                    return statistic['Op Type'] == 'Add'

                add_statistics = list(filter(get_add_node, stats))
                num_tensors = len(add_statistics)
                assert num_tensors == 3
                for tensor in add_statistics:
                    if tensor['IO'] == 'input':
                        assert tensor['Max Value'] == '1'
                        assert tensor['L2Norm Value'] == '8'
                    elif tensor['IO'] == 'output' and tensor['Slot'] == '0':
                        assert tensor['Max Value'] == '2'
                        assert tensor['L2Norm Value'] == '16'
        del os.environ['MINDSPORE_DUMP_CONFIG']


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_ascend_kernel_by_kernel_dynamic_shape():
    """
    Feature: Ascend kernel by kernel dump with dynamic shape model.
    Description: Test kernel by kernel dump in Ascend with trans_flag is configured to true.
    Expectation: Dump files has tensor data in host format (4 dimensions).
    """
    context.set_context(jit_level='O0')
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_trans_flag("test_e2e_dump_dynamic_shape")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_ascend_kernel_by_kernel_dynamic_shape_custom_statistic():
    """
    Feature: Ascend kernel by kernel dump with dynamic shape model.
    Description: Test kernel by kernel dump in Ascend with trans_flag is configured to true.
    Expectation: Dump user configured statistic_category, the config is ["l2norm", "max"].
    """
    context.set_context(jit_level='O0')
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_trans_flag("test_e2e_dump_dynamic_shape_custom_statistic")
