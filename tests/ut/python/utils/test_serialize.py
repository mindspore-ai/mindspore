# Copyright 2020 Huawei Technologies Co., Ltd
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
"""ut for model serialize(save/load)"""
import os
import stat
import time

import numpy as np
import pytest

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import context
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.nn import WithLossCell, TrainOneStepCell
from mindspore.nn.optim.momentum import Momentum
from mindspore.ops import operations as P
from mindspore.train.callback import _CheckpointManager
from mindspore.train.serialization import save_checkpoint, load_checkpoint, load_param_into_net, \
     export, _save_graph
from ..ut_filter import non_graph_engine

context.set_context(mode=context.GRAPH_MODE, print_file_path="print/print.pb")


class Net(nn.Cell):
    """Net definition."""

    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0, weight_init="zeros")
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(int(224 * 224 * 64 / 16), num_classes)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


_input_x = Tensor(np.random.randint(0, 255, [1, 3, 224, 224]).astype(np.float32))
_cur_dir = os.path.dirname(os.path.realpath(__file__))


def setup_module():
    import shutil
    if os.path.exists('./test_files'):
        shutil.rmtree('./test_files')


def test_save_graph():
    """ test_exec_save_graph """

    class Net1(nn.Cell):
        def __init__(self):
            super(Net1, self).__init__()
            self.add = P.Add()

        def construct(self, x, y):
            z = self.add(x, y)
            return z

    net = Net1()
    net.set_train()
    out_me_list = []
    x = Tensor(np.random.rand(2, 1, 2, 3).astype(np.float32))
    y = Tensor(np.array([1.2]).astype(np.float32))
    out_put = net(x, y)
    output_file = "net-graph.meta"
    _save_graph(network=net, file_name=output_file)
    out_me_list.append(out_put)
    assert os.path.exists(output_file)
    os.chmod(output_file, stat.S_IWRITE)
    os.remove(output_file)


def test_save_checkpoint_for_list():
    """ test save_checkpoint for list"""
    parameter_list = []
    one_param = {}
    param1 = {}
    param2 = {}
    one_param['name'] = "param_test"
    one_param['data'] = Tensor(np.random.randint(0, 255, [1, 3, 224, 224]), dtype=mstype.float32)
    param1['name'] = "param"
    param1['data'] = Tensor(np.random.randint(0, 255, [12, 1024]), dtype=mstype.float32)
    param2['name'] = "new_param"
    param2['data'] = Tensor(np.random.randint(0, 255, [12, 1024, 1]), dtype=mstype.float32)
    parameter_list.append(one_param)
    parameter_list.append(param1)
    parameter_list.append(param2)

    if os.path.exists('./parameters.ckpt'):
        os.chmod('./parameters.ckpt', stat.S_IWRITE)
        os.remove('./parameters.ckpt')

    ckpt_file_name = os.path.join(_cur_dir, './parameters.ckpt')
    save_checkpoint(parameter_list, ckpt_file_name)


def test_load_checkpoint_error_filename():
    ckpt_file_name = 1
    with pytest.raises(ValueError):
        load_checkpoint(ckpt_file_name)


def test_load_checkpoint():
    ckpt_file_name = os.path.join(_cur_dir, './parameters.ckpt')
    par_dict = load_checkpoint(ckpt_file_name)

    assert len(par_dict) == 3
    assert par_dict['param_test'].name == 'param_test'
    assert par_dict['param_test'].data.dtype == mstype.float32
    assert par_dict['param_test'].data.shape == (1, 3, 224, 224)
    assert isinstance(par_dict, dict)


def test_checkpoint_manager():
    """ test_checkpoint_manager """
    ckp_mgr = _CheckpointManager()

    ckpt_file_name = os.path.join(_cur_dir, './test1.ckpt')
    with open(ckpt_file_name, 'w'):
        os.chmod(ckpt_file_name, stat.S_IWUSR | stat.S_IRUSR)

    ckp_mgr.update_ckpoint_filelist(_cur_dir, "test")
    assert ckp_mgr.ckpoint_num == 1

    ckp_mgr.remove_ckpoint_file(ckpt_file_name)
    ckp_mgr.update_ckpoint_filelist(_cur_dir, "test")
    assert ckp_mgr.ckpoint_num == 0
    assert not os.path.exists(ckpt_file_name)

    another_file_name = os.path.join(_cur_dir, './test2.ckpt')
    another_file_name = os.path.realpath(another_file_name)
    with open(another_file_name, 'w'):
        os.chmod(another_file_name, stat.S_IWUSR | stat.S_IRUSR)

    ckp_mgr.update_ckpoint_filelist(_cur_dir, "test")
    assert ckp_mgr.ckpoint_num == 1
    ckp_mgr.remove_oldest_ckpoint_file()
    ckp_mgr.update_ckpoint_filelist(_cur_dir, "test")
    assert ckp_mgr.ckpoint_num == 0
    assert not os.path.exists(another_file_name)

    # test keep_one_ckpoint_per_minutes
    file1 = os.path.realpath(os.path.join(_cur_dir, './time_file1.ckpt'))
    file2 = os.path.realpath(os.path.join(_cur_dir, './time_file2.ckpt'))
    file3 = os.path.realpath(os.path.join(_cur_dir, './time_file3.ckpt'))
    with open(file1, 'w'):
        os.chmod(file1, stat.S_IWUSR | stat.S_IRUSR)
    with open(file2, 'w'):
        os.chmod(file2, stat.S_IWUSR | stat.S_IRUSR)
    with open(file3, 'w'):
        os.chmod(file3, stat.S_IWUSR | stat.S_IRUSR)
    time1 = time.time()
    ckp_mgr.update_ckpoint_filelist(_cur_dir, "time_file")
    assert ckp_mgr.ckpoint_num == 3
    ckp_mgr.keep_one_ckpoint_per_minutes(1, time1)
    ckp_mgr.update_ckpoint_filelist(_cur_dir, "time_file")
    assert ckp_mgr.ckpoint_num == 1
    if os.path.exists(_cur_dir + '/time_file1.ckpt'):
        os.chmod(_cur_dir + '/time_file1.ckpt', stat.S_IWRITE)
        os.remove(_cur_dir + '/time_file1.ckpt')


def test_load_param_into_net_error_net():
    parameter_dict = {}
    one_param = Parameter(Tensor(np.ones(shape=(64, 3, 7, 7)), dtype=mstype.float32),
                          name="conv1.weight")
    parameter_dict["conv1.weight"] = one_param
    with pytest.raises(TypeError):
        load_param_into_net('', parameter_dict)


def test_load_param_into_net_error_dict():
    net = Net(10)
    with pytest.raises(TypeError):
        load_param_into_net(net, '')


def test_load_param_into_net_erro_dict_param():
    net = Net(10)
    net.init_parameters_data()
    assert net.conv1.weight.data.asnumpy()[0][0][0][0] == 0

    parameter_dict = {}
    one_param = ''
    parameter_dict["conv1.weight"] = one_param
    with pytest.raises(TypeError):
        load_param_into_net(net, parameter_dict)


def test_load_param_into_net_has_more_param():
    """ test_load_param_into_net_has_more_param """
    net = Net(10)
    net.init_parameters_data()
    assert net.conv1.weight.data.asnumpy()[0][0][0][0] == 0

    parameter_dict = {}
    one_param = Parameter(Tensor(np.ones(shape=(64, 3, 7, 7)), dtype=mstype.float32),
                          name="conv1.weight")
    parameter_dict["conv1.weight"] = one_param
    two_param = Parameter(Tensor(np.ones(shape=(64, 3, 7, 7)), dtype=mstype.float32),
                          name="conv1.weight")
    parameter_dict["conv1.w"] = two_param
    load_param_into_net(net, parameter_dict)
    assert net.conv1.weight.data.asnumpy()[0][0][0][0] == 1


def test_load_param_into_net_param_type_and_shape_error():
    net = Net(10)
    net.init_parameters_data()
    assert net.conv1.weight.data.asnumpy()[0][0][0][0] == 0

    parameter_dict = {}
    one_param = Parameter(Tensor(np.ones(shape=(64, 3, 7, 7))), name="conv1.weight")
    parameter_dict["conv1.weight"] = one_param
    with pytest.raises(RuntimeError):
        load_param_into_net(net, parameter_dict)


def test_load_param_into_net_param_type_error():
    net = Net(10)
    net.init_parameters_data()
    assert net.conv1.weight.data.asnumpy()[0][0][0][0] == 0

    parameter_dict = {}
    one_param = Parameter(Tensor(np.ones(shape=(64, 3, 7, 7)), dtype=mstype.int32),
                          name="conv1.weight")
    parameter_dict["conv1.weight"] = one_param
    with pytest.raises(RuntimeError):
        load_param_into_net(net, parameter_dict)


def test_load_param_into_net_param_shape_error():
    net = Net(10)
    net.init_parameters_data()
    assert net.conv1.weight.data.asnumpy()[0][0][0][0] == 0

    parameter_dict = {}
    one_param = Parameter(Tensor(np.ones(shape=(64, 3, 7,)), dtype=mstype.int32),
                          name="conv1.weight")
    parameter_dict["conv1.weight"] = one_param
    with pytest.raises(RuntimeError):
        load_param_into_net(net, parameter_dict)


def test_load_param_into_net():
    net = Net(10)
    net.init_parameters_data()
    assert net.conv1.weight.data.asnumpy()[0][0][0][0] == 0

    parameter_dict = {}
    one_param = Parameter(Tensor(np.ones(shape=(64, 3, 7, 7)), dtype=mstype.float32),
                          name="conv1.weight")
    parameter_dict["conv1.weight"] = one_param
    load_param_into_net(net, parameter_dict)
    assert net.conv1.weight.data.asnumpy()[0][0][0][0] == 1


def test_save_checkpoint_for_network():
    """ test save_checkpoint for network"""
    net = Net()
    loss = SoftmaxCrossEntropyWithLogits(sparse=True)
    opt = Momentum(net.trainable_params(), 0.0, 0.9, 0.0001, 1024)

    loss_net = WithLossCell(net, loss)
    train_network = TrainOneStepCell(loss_net, opt)
    save_checkpoint(train_network, ckpt_file_name="./new_ckpt.ckpt")

    load_checkpoint("new_ckpt.ckpt")


def test_load_checkpoint_empty_file():
    os.mknod("empty.ckpt")
    with pytest.raises(ValueError):
        load_checkpoint("empty.ckpt")


class MYNET(nn.Cell):
    """ NET definition """

    def __init__(self):
        super(MYNET, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, has_bias=False, weight_init='normal', pad_mode='valid')
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(64 * 222 * 222, 3)  # padding=0

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.flatten(x)
        out = self.fc(x)
        return out


@non_graph_engine
def test_export():
    net = MYNET()
    input_data = Tensor(np.random.randint(0, 255, [1, 3, 224, 224]).astype(np.float32))
    with pytest.raises(ValueError):
        export(net, input_data, file_name="./me_export.pb", file_format="AIR")


@non_graph_engine
def test_mindir_export():
    net = MYNET()
    input_data = Tensor(np.random.randint(0, 255, [1, 3, 224, 224]).astype(np.float32))
    export(net, input_data, file_name="./me_binary_export", file_format="MINDIR")


class PrintNet(nn.Cell):
    def __init__(self):
        super(PrintNet, self).__init__()
        self.print = P.Print()

    def construct(self, int8, uint8, int16, uint16, int32, uint32, int64, uint64, flt16, flt32, flt64, bool_,
                  scale1, scale2):
        self.print('============tensor int8:==============', int8)
        self.print('============tensor int8:==============', int8)
        self.print('============tensor uint8:==============', uint8)
        self.print('============tensor int16:==============', int16)
        self.print('============tensor uint16:==============', uint16)
        self.print('============tensor int32:==============', int32)
        self.print('============tensor uint32:==============', uint32)
        self.print('============tensor int64:==============', int64)
        self.print('============tensor uint64:==============', uint64)
        self.print('============tensor float16:==============', flt16)
        self.print('============tensor float32:==============', flt32)
        self.print('============tensor float64:==============', flt64)
        self.print('============tensor bool:==============', bool_)
        self.print('============tensor scale1:==============', scale1)
        self.print('============tensor scale2:==============', scale2)
        return int8, uint8, int16, uint16, int32, uint32, int64, uint64, flt16, flt32, flt64, bool_, scale1, scale2


def test_print():
    print_net = PrintNet()
    int8 = Tensor(np.random.randint(100, size=(10, 10), dtype="int8"))
    uint8 = Tensor(np.random.randint(100, size=(10, 10), dtype="uint8"))
    int16 = Tensor(np.random.randint(100, size=(10, 10), dtype="int16"))
    uint16 = Tensor(np.random.randint(100, size=(10, 10), dtype="uint16"))
    int32 = Tensor(np.random.randint(100, size=(10, 10), dtype="int32"))
    uint32 = Tensor(np.random.randint(100, size=(10, 10), dtype="uint32"))
    int64 = Tensor(np.random.randint(100, size=(10, 10), dtype="int64"))
    uint64 = Tensor(np.random.randint(100, size=(10, 10), dtype="uint64"))
    float16 = Tensor(np.random.rand(224, 224).astype(np.float16))
    float32 = Tensor(np.random.rand(224, 224).astype(np.float32))
    float64 = Tensor(np.random.rand(224, 224).astype(np.float64))
    bool_ = Tensor(np.arange(-10, 10, 2).astype(np.bool_))
    scale1 = Tensor(np.array(1))
    scale2 = Tensor(np.array(0.1))
    print_net(int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bool_, scale1,
              scale2)


def teardown_module():
    files = ['parameters.ckpt', 'new_ckpt.ckpt', 'empty.ckpt']
    for item in files:
        file_name = './' + item
        if not os.path.exists(file_name):
            continue
        os.chmod(file_name, stat.S_IWRITE)
        os.remove(file_name)
    import shutil
    if os.path.exists('./print'):
        shutil.rmtree('./print')
