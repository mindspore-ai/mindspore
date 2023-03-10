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
import platform
import stat
import time
import secrets

import numpy as np
import pytest

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import context
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.nn import SoftmaxCrossEntropyWithLogits, WithLossCell, TrainOneStepCell
from mindspore.nn.optim.momentum import Momentum
from mindspore.ops import operations as P
from mindspore.train.callback import _CheckpointManager
from mindspore.train.serialization import save_checkpoint, load_checkpoint, load_param_into_net, \
    export, _save_graph, load
from tests.security_utils import security_off_wrap
from ..ut_filter import non_graph_engine


class Net(nn.Cell):
    """
    Net definition.
    parameter name :
        conv1.weight
        bn1.moving_mean
        bn1.moving_variance
        bn1.gamma
        bn1.beta
        fc.weight
        fc.bias
    """

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
    context.set_context(mode=context.GRAPH_MODE)

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
    context.set_context(mode=context.GRAPH_MODE)
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
    """
    Feature: Load checkpoint.
    Description: Load checkpoint with error filename.
    Expectation: Raise value error for error filename.
    """
    context.set_context(mode=context.GRAPH_MODE)
    ckpt_file_name = 1
    with pytest.raises(TypeError):
        load_checkpoint(ckpt_file_name)


def test_save_checkpoint_for_list_append_info_and_load_checkpoint():
    """
    Feature: Save checkpoint for list append info and load checkpoint.
    Description: Save checkpoint for list append info and load checkpoint with list append info.
    Expectation: Checkpoint for list append info can be saved and reloaded.
    """
    context.set_context(mode=context.GRAPH_MODE)
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
    append_dict = {"lr": 0.01, "epoch": 20, "train": True, "par_string": "string_test",
                   "par_param": Parameter(Tensor(1.0)), "par_tensor": Tensor([[1, 2], [2, 3]])}
    if os.path.exists('./parameters.ckpt'):
        os.chmod('./parameters.ckpt', stat.S_IWRITE)
        os.remove('./parameters.ckpt')

    ckpt_file_name = os.path.join(_cur_dir, './parameters.ckpt')
    save_checkpoint(parameter_list, ckpt_file_name, append_dict=append_dict)
    par_dict = load_checkpoint(ckpt_file_name)

    assert len(par_dict) == 9
    assert par_dict.get('param_test').name == 'param_test'
    assert par_dict.get('param_test').data.dtype == mstype.float32
    assert par_dict.get('param_test').data.shape == (1, 3, 224, 224)

    assert par_dict.get('par_string') == "string_test"
    assert par_dict.get('par_param').name == 'par_param'
    assert par_dict.get('par_param').data.dtype == mstype.float32
    assert par_dict.get('par_param').data.shape == ()
    assert par_dict.get('par_tensor').name == 'par_tensor'
    assert par_dict.get('par_tensor').data.dtype == mstype.int64
    assert par_dict.get('par_tensor').data.shape == (2, 2)


def test_checkpoint_manager():
    """ test_checkpoint_manager """
    context.set_context(mode=context.GRAPH_MODE)
    ckp_mgr = _CheckpointManager()

    ckpt_file_name = os.path.join(_cur_dir, './test-1_1.ckpt')
    with open(ckpt_file_name, 'w'):
        os.chmod(ckpt_file_name, stat.S_IWUSR | stat.S_IRUSR)

    ckp_mgr.update_ckpoint_filelist(_cur_dir, "test")
    assert ckp_mgr.ckpoint_num == 1

    ckp_mgr.remove_ckpoint_file(ckpt_file_name)
    ckp_mgr.update_ckpoint_filelist(_cur_dir, "test")
    assert ckp_mgr.ckpoint_num == 0
    assert not os.path.exists(ckpt_file_name)

    another_file_name = os.path.join(_cur_dir, './test-2_1.ckpt')
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
    file1 = os.path.realpath(os.path.join(_cur_dir, './time_file-1_1.ckpt'))
    file2 = os.path.realpath(os.path.join(_cur_dir, './time_file-2_1.ckpt'))
    file3 = os.path.realpath(os.path.join(_cur_dir, './time_file-3_1.ckpt'))
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
    if os.path.exists(_cur_dir + '/time_file-1_1.ckpt'):
        os.chmod(_cur_dir + '/time_file-1_1.ckpt', stat.S_IWRITE)
        os.remove(_cur_dir + '/time_file-1_1.ckpt')


def test_load_param_into_net_error_net():
    context.set_context(mode=context.GRAPH_MODE)
    parameter_dict = {}
    one_param = Parameter(Tensor(np.ones(shape=(64, 3, 7, 7)), dtype=mstype.float32),
                          name="conv1.weight")
    parameter_dict["conv1.weight"] = one_param
    with pytest.raises(TypeError):
        load_param_into_net('', parameter_dict)


def test_load_param_into_net_error_dict():
    context.set_context(mode=context.GRAPH_MODE)
    net = Net(10)
    with pytest.raises(TypeError):
        load_param_into_net(net, '')


def test_load_param_into_net_erro_dict_param():
    context.set_context(mode=context.GRAPH_MODE)
    net = Net(10)
    net.init_parameters_data()
    assert net.conv1.weight.data.asnumpy()[0][0][0][0] == 0

    parameter_dict = {}
    one_param = 1
    parameter_dict["conv1.weight"] = one_param
    with pytest.raises(TypeError):
        load_param_into_net(net, parameter_dict)


def test_load_param_into_net_has_more_param():
    """ test_load_param_into_net_has_more_param """
    context.set_context(mode=context.GRAPH_MODE)
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
    context.set_context(mode=context.GRAPH_MODE)
    net = Net(10)
    net.init_parameters_data()
    assert net.conv1.weight.data.asnumpy()[0][0][0][0] == 0

    parameter_dict = {}
    one_param = Parameter(Tensor(np.ones(shape=(64, 3, 7, 7)), dtype=mstype.int32), name="conv1.weight")
    parameter_dict["conv1.weight"] = one_param
    with pytest.raises(RuntimeError):
        load_param_into_net(net, parameter_dict)


def test_load_param_into_net_param_type_error():
    context.set_context(mode=context.GRAPH_MODE)
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
    context.set_context(mode=context.GRAPH_MODE)
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
    context.set_context(mode=context.GRAPH_MODE)
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
    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    loss = SoftmaxCrossEntropyWithLogits(sparse=True)
    opt = Momentum(net.trainable_params(), 0.0, 0.9, 0.0001, 1024)

    loss_net = WithLossCell(net, loss)
    train_network = TrainOneStepCell(loss_net, opt)
    save_checkpoint(train_network, ckpt_file_name="./new_ckpt.ckpt")

    load_checkpoint("new_ckpt.ckpt")


def test_load_checkpoint_empty_file():
    context.set_context(mode=context.GRAPH_MODE)
    os.mknod("empty.ckpt")
    with pytest.raises(ValueError):
        load_checkpoint("empty.ckpt")


def test_load_checkpoint_error_param():
    """
    Feature: Load checkpoint.
    Description: Load checkpoint with error param.
    Expectation: Raise value error for error param.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net(10)
    ckpt_file = "check_name.ckpt"
    save_checkpoint(net, ckpt_file)
    with pytest.raises(ValueError):
        load_checkpoint(ckpt_file, choice_func=lambda x: x.startswith(123))
    with pytest.raises(ValueError):
        load_checkpoint(ckpt_file, choice_func=lambda x: not x.startswith(""))
    if os.path.exists(ckpt_file):
        os.remove(ckpt_file)


def test_load_checkpoint_error_load():
    """
    Feature: Load checkpoint.
    Description: Load checkpoint with empty parameter dict.
    Expectation: Raise value error for error load.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net(10)
    ckpt_file = "check_name.ckpt"
    save_checkpoint(net, ckpt_file)
    with pytest.raises(ValueError):
        load_checkpoint(ckpt_file, choice_func=lambda x: x.startswith("123"))
    if os.path.exists(ckpt_file):
        os.remove(ckpt_file)


def test_load_checkpoint_specify_prefix():
    """
    Feature: Load checkpoint.
    Description: Load checkpoint with param `specify_prefix`.
    Expectation: Correct loaded checkpoint file.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net(10)
    ckpt_file = "specify_prefix.ckpt"
    save_checkpoint(net, ckpt_file)
    param_dict = load_checkpoint(ckpt_file, choice_func=lambda x: x.startswith("bn"))
    assert len(param_dict) == 4
    param_dict = load_checkpoint(ckpt_file, choice_func=lambda x: x.startswith("fc"))
    assert len(param_dict) == 2
    param_dict = load_checkpoint(ckpt_file, choice_func=lambda x: x.startswith(("fc", "bn")))
    assert len(param_dict) == 6
    if os.path.exists(ckpt_file):
        os.remove(ckpt_file)


def test_load_checkpoint_filter_prefix():
    """
    Feature: Load checkpoint.
    Description: Load checkpoint with param `filter_prefix`.
    Expectation: Correct loaded checkpoint file.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net(10)
    ckpt_file = "filter_prefix.ckpt"
    save_checkpoint(net, ckpt_file)
    param_dict = load_checkpoint(ckpt_file, choice_func=lambda x: not x.startswith("fc"))
    assert len(param_dict) == 5
    param_dict = load_checkpoint(ckpt_file, choice_func=lambda x: not x.startswith("bn"))
    assert len(param_dict) == 3
    param_dict = load_checkpoint(ckpt_file, choice_func=lambda x: not x.startswith(("bn", "fc")))
    assert len(param_dict) == 1
    if os.path.exists(ckpt_file):
        os.remove(ckpt_file)


def test_load_checkpoint_specify_filter_prefix():
    """
    Feature: Load checkpoint.
    Description: Load checkpoint with param `filter_prefix` and `specify_prefix`.
    Expectation: Correct loaded checkpoint file.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net(10)
    ckpt_file = "specify_filter_prefix.ckpt"
    save_checkpoint(net, ckpt_file)
    param_dict = load_checkpoint(ckpt_file, choice_func=lambda x: x.startswith("bn") and not x.startswith("bn1.moving"))
    assert len(param_dict) == 2
    param_dict = load_checkpoint(ckpt_file,
                                 choice_func=lambda x: x.startswith(("bn", "fc")) and not x.startswith("fc.weight"))
    assert len(param_dict) == 5
    if os.path.exists(ckpt_file):
        os.remove(ckpt_file)


def test_save_and_load_checkpoint_for_network_with_encryption():
    """ test save and checkpoint for network with encryption"""
    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    loss = SoftmaxCrossEntropyWithLogits(sparse=True)
    opt = Momentum(net.trainable_params(), 0.0, 0.9, 0.0001, 1024)

    loss_net = WithLossCell(net, loss)
    train_network = TrainOneStepCell(loss_net, opt)
    key = secrets.token_bytes(16)
    mode = "AES-GCM"
    ckpt_path = "./encrypt_ckpt.ckpt"
    if platform.system().lower() == "windows":
        with pytest.raises(NotImplementedError):
            save_checkpoint(train_network, ckpt_file_name=ckpt_path, enc_key=key, enc_mode=mode)
            param_dict = load_checkpoint(ckpt_path, dec_key=key, dec_mode="AES-GCM")
            load_param_into_net(net, param_dict)
    else:
        save_checkpoint(train_network, ckpt_file_name=ckpt_path, enc_key=key, enc_mode=mode)
        param_dict = load_checkpoint(ckpt_path, dec_key=key, dec_mode="AES-GCM")
        load_param_into_net(net, param_dict)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


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
    context.set_context(mode=context.GRAPH_MODE)
    net = MYNET()
    input_data = Tensor(np.random.randint(0, 255, [1, 3, 224, 224]).astype(np.float32))
    with pytest.raises(ValueError):
        export(net, input_data, file_name="./me_export.pb", file_format="AIR")


@non_graph_engine
def test_mindir_export():
    context.set_context(mode=context.GRAPH_MODE)
    net = MYNET()
    input_data = Tensor(np.random.randint(0, 255, [1, 3, 224, 224]).astype(np.float32))
    export(net, input_data, file_name="./me_binary_export", file_format="MINDIR")


@non_graph_engine
def test_mindir_export_and_load_with_encryption():
    context.set_context(mode=context.GRAPH_MODE)
    net = MYNET()
    input_data = Tensor(np.random.randint(0, 255, [1, 3, 224, 224]).astype(np.float32))
    key = secrets.token_bytes(16)
    export(net, input_data, file_name="./me_cipher_binary_export.mindir", file_format="MINDIR", enc_key=key)
    load("./me_cipher_binary_export.mindir", dec_key=key)


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


@security_off_wrap
def test_print():
    context.set_context(mode=context.GRAPH_MODE, print_file_path="print/print.pb")
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
