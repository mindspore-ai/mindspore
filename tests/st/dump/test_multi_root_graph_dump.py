# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
from os import path
import tempfile
import time
import shutil
import csv
import numpy as np
import pytest
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.nn import Cell
from dump_test_utils import generate_dump_json
from tests.security_utils import security_off_wrap


class AddNet(Cell):
    def __init__(self):
        super(AddNet, self).__init__()
        self.add = P.TensorAdd()

    def construct(self, input_x, input_y):
        output_z = self.add(input_x, input_y)
        return output_z


class NewAddNet(Cell):
    def __init__(self):
        super(NewAddNet, self).__init__()
        self.add = P.AddN()

    def construct(self, b1, b2, x, y):
        z = self.add([x, y, y])
        if b1 < b2:
            z = self.add([x, y, y])
        else:
            z = self.add([x, x, y])
        return z


def train_addnet(epoch):
    net = AddNet()
    net2 = NewAddNet()
    output_list = []
    b1 = Tensor(np.array(1).astype(np.float32))
    b2 = Tensor(np.array(3).astype(np.float32))
    input_x = Tensor(np.ones([2, 1, 2, 1]).astype(np.float32))
    input_y = Tensor(np.ones([2, 1, 2, 1]).astype(np.float32))
    for _ in range(epoch):
        out_put = net(input_x, input_y)
        out2 = net2(b1, b2, out_put, input_x)
        output_list.append(out2.asnumpy())
        input_x = input_x + input_y
        b1 = b1+1
    return output_list


def check_graph_structure(dump_file_path, execution_order_path, graph_id, expect_steps):
    dump_data_path = os.path.join(dump_file_path, graph_id)
    assert sorted(os.listdir(dump_data_path)) == expect_steps
    graph_history_file_path = os.path.join(
        execution_order_path, 'ms_global_execution_order_graph_{}.csv'.format(graph_id))
    assert path.exists(graph_history_file_path)
    with open(graph_history_file_path) as csvfile:
        history_graph = csv.reader(csvfile)
        iter_list_graph = [row[0] for row in history_graph]
    assert iter_list_graph == expect_steps


def run_multi_root_graph_dump(device, dump_mode, test_name):
    """Run dump for multi root graph script."""

    context.set_context(mode=context.GRAPH_MODE, device_target=device)

    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, dump_mode)
        dump_config_path = os.path.join(tmp_dir, dump_mode + ".json")
        generate_dump_json(dump_path, dump_config_path, test_name)
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        dump_file_path = os.path.join(dump_path, 'rank_0', 'Net')
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        epoch = 3
        train_addnet(epoch)
        for _ in range(3):
            if not os.path.exists(dump_file_path):
                time.sleep(2)
        execution_order_path = os.path.join(dump_path, 'rank_0', 'execution_order')
        # Multi root graph script: check dump data dir and graph history files and see if iteration number is matched.
        if device == "GPU":
            # In GPU, we have 4 kernel graphs folders under rank_0 dir.
            # In graph history dir, there are 2 files for each graph (ms_execution_order and ms_global_execution_order).
            assert len(os.listdir(dump_file_path)) == 4
            assert len(os.listdir(execution_order_path)) == 8
            check_graph_structure(dump_file_path, execution_order_path, '0', ['0', '2', '4'])
            check_graph_structure(dump_file_path, execution_order_path, '1', ['1', '3', '5'])
            check_graph_structure(dump_file_path, execution_order_path, '2', ['1', '3'])
            check_graph_structure(dump_file_path, execution_order_path, '3', ['5'])
        else:
            # In Ascend, we have 2 root graphs folders under rank_0 dir.
            # In graph history dir, there are 4 ms_execution_order files and 2 ms_global_execution_order files.
            # Each graph should have 3 iterations. Each graph was executed once per epoch.
            # Graph 0 was executed in even iterations, graph 1 was executed in odd iterations.
            assert len(os.listdir(dump_file_path)) == 2
            assert len(os.listdir(execution_order_path)) == 6
            check_graph_structure(dump_file_path, execution_order_path, '0', ['0', '2', '4'])
            check_graph_structure(dump_file_path, execution_order_path, '1', ['1', '3', '5'])


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@security_off_wrap
def test_GPU_e2e_multi_root_graph_dump():
    """
    Feature:
        Multi root graph e2e dump for GPU.
    Description:
        Test multi root graph e2e dump GPU.
    Expectation:
        Dump for two different graphs, graph 0 even iterations and graph 1 odd iterations.
    """
    run_multi_root_graph_dump("GPU", "e2e_dump", "test_GPU_e2e_multi_root_graph_dump")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_Ascend_e2e_multi_root_graph_dump():
    """
    Feature:
        Multi root graph e2e dump for Ascend.
    Description:
        Test multi root graph e2e dump Ascend.
    Expectation:
        Dump for two different graphs, graph 0 even iterations and graph 1 odd iterations.
    """

    run_multi_root_graph_dump("Ascend", "e2e_dump", "test_Ascend_e2e_multi_root_graph_dump")


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_Ascend_async_multi_root_graph_dump():
    """
    Feature:
        Multi root graph async dump for Ascend.
    Description:
        Test multi root graph async dump Ascend.
    Expectation:
        Dump for two different graphs, graph 0 even iterations and graph 1 odd iterations.
    """
    run_multi_root_graph_dump("Ascend", "async_dump", "test_Ascend_async_multi_root_graph_dump")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_ascend_multi_root_graph_dump_kernel_by_kernel():
    """
    Feature:
        Multi root graph dump for Ascend kernel by kernel.
    Description:
        Test multi root graph dump in Ascend kernel by kernel.
    Expectation:
        Dump for two different graphs, graph 0 even iterations and graph 1 odd iterations.
    """
    os.environ['GRAPH_OP_RUN'] = "1"
    run_multi_root_graph_dump("Ascend", "e2e_dump", "test_Ascend_e2e_multi_root_graph_dump")
    del os.environ['GRAPH_OP_RUN']
