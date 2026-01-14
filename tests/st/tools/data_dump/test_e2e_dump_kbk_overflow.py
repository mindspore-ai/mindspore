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
Tests sync data dump overflow
"""

import os
import json
import sys
import tempfile
import glob
import shutil
import numpy as np
import mindspore
from mindspore import ops
from mindspore import nn
from mindspore import context
from mindspore import Tensor
from mindspore.common import dtype as msdtype
from mindspore.common.parameter import Parameter
from mindspore import dataset
from mindspore.nn import Momentum
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.train import Model
from tests.mark_utils import arg_mark
from tests.security_utils import security_off_wrap
from dump_test_utils import generate_dump_json, check_dump_structure
from dump_test_utils import migrate_resnet50
from dump_check import SyncDumpCheck

class ConvNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv2 = ops.Conv2D(out_channel=3, kernel_size=1)

    @mindspore.jit(backend="ms_backend", jit_level="O0")
    def construct(self, x, weight):
        return self.conv2(x, weight)


class NetMulAdd(nn.Cell):
    """A simple net with mul and add ops."""
    def __init__(self):
        super().__init__()
        self.add = ops.Add()
        self.mul = ops.Mul()

    @mindspore.jit(backend="ms_backend", jit_level="O0")
    def construct(self, x_, y_):
        x_ = self.mul(x_, 2)
        y_ = self.mul(y_, 2)
        x_ = self.add(x_, y_)
        y_ = self.add(x_, y_)
        return self.add(x_, y_)


class ViewNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.transpose = ops.TransposeView()

    @mindspore.jit(backend="ms_backend", jit_level="O0")
    def construct(self, x, perm):
        out = self.transpose(x, perm)
        return out

class AddMulNet(nn.Cell):
    """A simple net with mul and add ops."""
    def __init__(self, fill_value, strategy=None, dtype=np.float32):
        super().__init__()
        add_np = np.full((1, 3), fill_value=fill_value, dtype=dtype)
        self.add_weight = Parameter(Tensor(add_np), name="add_weight")
        mul_np = np.full((1, 3), fill_value=1, dtype=dtype)
        self.mul_weight = Parameter(Tensor(mul_np), name="mul_weight")
        self.add = ops.Add()
        self.mul = ops.Mul()
        if strategy is not None:
            self.add.shard(strategy[0])
            self.mul.shard(strategy[1])

    @mindspore.jit(backend="ms_backend", jit_level="O0")
    def construct(self, x):
        x = self.add(x, self.add_weight)
        out = self.mul(x, self.mul_weight)
        return out

def two_dataset_generator_fp32():
    for num in range(0, 3):
        yield (np.full((8 + 8 * num, 3), fill_value=num, dtype=np.float32),
               np.full((8 + 8 * num, 3), fill_value=num, dtype=np.float32))

def check_dump_dynamic_net_sync_overflow_dump(dump_path, dump_config_path, test_name):
    """check dynamic shape overflow dump"""
    generate_dump_json(dump_path, dump_config_path, test_name)
    net_parallel = AddMulNet(fill_value=3.4e38, strategy=None, dtype=np.float32)
    input_x = Tensor(shape=[None, 3], dtype=msdtype.float32)
    input_y = Tensor(shape=[None, 3], dtype=msdtype.float32)
    parallel_dataset = dataset.GeneratorDataset(two_dataset_generator_fp32, ["data", "label"])
    loss = SoftmaxCrossEntropyWithLogits(reduction='mean')
    opt_fn = Momentum(learning_rate=0.01, momentum=0.9, params=net_parallel.get_parameters())
    model = Model(network=net_parallel, loss_fn=loss, optimizer=opt_fn, amp_level="O0")
    model.train_network.set_inputs(input_x, input_y)
    model.train(epoch=3, train_dataset=parallel_dataset, dataset_sink_mode=False, sink_size=-1)
    check_dump_structure(dump_path, dump_config_path, 1, 0, 1)
    dump_data_path = os.path.join(dump_path, 'rank_0', 'Net', '0', '0')
    assert os.path.exists(dump_data_path)
    # tensor data in host format.
    output_name1 = "TupleToTensor.Gradients_Default_network-WithLossCell__loss_fn-SoftmaxCross"
    output_name2 = "EntropyWithLogits_Grad_ReduceMean_TupleToTensor-op*.input.0.DefaultFormat.*.npy"
    output_name = output_name1 + output_name2
    output_path = glob.glob(os.path.join(dump_data_path, output_name))[0]
    real_path = os.path.realpath(output_path)
    output = np.load(real_path)
    assert output.shape == (1,)
    assert oct(os.stat(real_path).st_mode)[-3:] == str(400)

def run_trans_flag_dvm(test_name):
    """Run e2e dump on scenario, testing trans_flag functionality"""
    if sys.platform != 'linux':
        return
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, test_name)
        dump_config_path = os.path.join(tmp_dir, '{}.json'.format(test_name))
        generate_dump_json(dump_path, dump_config_path, test_name)
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)

        migrate_resnet50(tmp_dir)
        src_dir = os.path.join(tmp_dir, "src")
        sys.path.append(os.path.dirname(src_dir))
        from src.resnet import resnet50
        generate_dump_json(dump_path, dump_config_path, test_name)
        net = resnet50()
        predict = Tensor(np.ones([32, 3, 32, 32]).astype(np.float32) * 65534)
        net(predict)
        check_dump_structure(dump_path, dump_config_path, 1, 0, 1)
        dump_data_path = os.path.join(dump_path, 'rank_0', 'Net', '0', '0')
        assert os.path.exists(dump_data_path)
        with open(dump_config_path, 'r', encoding="utf-8") as f:
            dump_json = json.load(f)
        dump_check = SyncDumpCheck(dump_json, iteration_id_list=1)
        dump_check.dump_result_check()
        del os.environ['MINDSPORE_DUMP_CONFIG']
        sys.path.remove(os.path.dirname(src_dir))


def run_trans_flag(test_name):
    """Run e2e dump on scenario, testing trans_flag functionality"""
    if sys.platform != 'linux':
        return
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, test_name)
        dump_config_path = os.path.join(tmp_dir, '{}.json'.format(test_name))
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)

        if test_name == "test_e2e_dump_trans_true_op_debug_mode":
            generate_dump_json(dump_path, dump_config_path, test_name)
            tensor = Tensor(np.full((1, 3, 3, 3), 65504, dtype=np.float16), mindspore.float16)
            weight = Tensor(np.full((3, 3, 1, 1), 65504, dtype=np.float16), mindspore.float16)
            net = ConvNet()
            expect = net(tensor, weight)
            check_dump_structure(dump_path, dump_config_path, 1, 0, 1)
            dump_data_path = os.path.join(dump_path, 'rank_0', 'Net', '0', '0')
            assert os.path.exists(dump_data_path)
            # tensor data in host format.
            output_name = "Conv2D.Default_Conv2D-op*.output.0.DefaultFormat.*.npy"
            output_path = glob.glob(os.path.join(dump_data_path, output_name))[0]
            real_path = os.path.realpath(output_path)
            output = np.load(real_path)
            assert output.shape == (1, 3, 3, 3)
            assert np.array_equal(output, expect)

        if test_name == "test_e2e_dump_set_overflow_number":
            set_overflow_num = 2
            generate_dump_json(dump_path, dump_config_path, test_name, overflow_number=set_overflow_num)
            data = np.array([60000, 60000]).astype(np.float16)
            net = NetMulAdd()
            net(Tensor(data), Tensor(data))
            check_dump_structure(dump_path, dump_config_path, 1, 0, 1)
            dump_data_path = os.path.join(dump_path, 'rank_0', 'Net', '0', '0')
            assert os.path.exists(dump_data_path)
            overflow_files = glob.glob(os.path.join(dump_data_path, "*.npy"))
            overflow_files_num = len(overflow_files)
            assert overflow_files_num == set_overflow_num * 3

        if test_name == "test_dump_dynamic_net_sync_overflow_dump":
            check_dump_dynamic_net_sync_overflow_dump(dump_path, dump_config_path, test_name)

        del os.environ['MINDSPORE_DUMP_CONFIG']


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
@security_off_wrap
def test_ascend_kernel_by_kernel_trans_true_op_debug_mode():
    """
    Feature: Ascend kernel by kernel dump with overflow.
    Description: Test kernel by kernel dump in Ascend with trans_flag is configured to true.
    Expectation: Dump files has tensor data in host format (4 dimensions).
    """
    context.set_context(mode=context.GRAPH_MODE)
    run_trans_flag("test_e2e_dump_trans_true_op_debug_mode")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=104.0)
@security_off_wrap
def test_ascend_sync_overflow_dvm():
    """
    Feature: Ascend kernel by kernel dump with overflow support for uncontiguous tensor.
    Description: Test kernel by kernel dump in Ascend with uncontiguous tensor.
    Expectation: Dump files has tensor data in host format (3 dimensions).
    """
    context.set_context(jit_level='O1')
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_trans_flag_dvm("test_e2e_dump_trans_true_op_debug_mode")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
@security_off_wrap
def test_e2e_dump_set_overflow_number():
    """
    Feature: The number of overflow dump during the training process can be configured.
    Description: Test kernel by kernel dump in Ascend with overflow_number is configured.
    Expectation: The number of dump files matches the value of the overflow_number parameter that has been set.
    """
    context.set_context(mode=context.GRAPH_MODE)
    run_trans_flag("test_e2e_dump_set_overflow_number")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=4.01)
@security_off_wrap
def test_dump_dynamic_net_sync_overflow_dump():
    """
    Feature: The number of overflow dump during the training process can be configured.
    Description: Test kernel by kernel dump in Ascend with overflow_number is configured.
    Expectation: The number of dump files matches the value of the overflow_number parameter that has been set.
    """
    context.set_context(mode=context.GRAPH_MODE)
    run_trans_flag("test_dump_dynamic_net_sync_overflow_dump")
