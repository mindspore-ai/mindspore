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
"""test callback function."""
import os
import stat

import numpy as np
import pytest

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.common.api import ms_function
from mindspore.common.tensor import Tensor
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import Momentum
from mindspore.train.callback import ModelCheckpoint, _check_file_name_prefix, RunContext, _checkpoint_cb_for_save_op, \
    LossMonitor, _InternalCallbackParam, _chg_ckpt_file_name_if_same_exist, \
    _build_callbacks, CheckpointConfig, _set_cur_net


class Net(nn.Cell):
    """Net definition."""

    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, has_bias=False, weight_init='normal')
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(64 * 222 * 222, 3)

    @ms_function
    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.flatten(x)
        out = self.fc(x)
        return out


class LossNet(nn.Cell):
    """ LossNet definition """

    def __init__(self):
        super(LossNet, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, has_bias=False, weight_init='normal', pad_mode='valid')
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(64 * 222 * 222, 3)  # padding=0
        self.loss = nn.SoftmaxCrossEntropyWithLogits()

    @ms_function
    def construct(self, x, y):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        out = self.loss(x, y)
        return out


def test_Model_Checkpoint_prefix_invalid():
    """Test ModelCheckpoint prefix invalid."""
    with pytest.raises(ValueError):
        ModelCheckpoint(123)
    ModelCheckpoint(directory="./")
    with pytest.raises(TypeError):
        ModelCheckpoint(config='type_error')
    ModelCheckpoint(config=CheckpointConfig())
    ModelCheckpoint(prefix="ckpt_2", directory="./test_files")


def test_save_checkpoint():
    """Test save checkpoint."""
    train_config = CheckpointConfig(
        save_checkpoint_steps=16,
        save_checkpoint_seconds=0,
        keep_checkpoint_max=5,
        keep_checkpoint_per_n_minutes=0)
    cb_params = _InternalCallbackParam()
    net = Net()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    optim = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    network_ = WithLossCell(net, loss)
    _train_network = TrainOneStepCell(network_, optim)
    cb_params.train_network = _train_network
    cb_params.epoch_num = 10
    cb_params.cur_epoch_num = 5
    cb_params.cur_step_num = 0
    cb_params.batch_num = 32
    ckpoint_cb = ModelCheckpoint(prefix="test_ckpt", directory='./test_files', config=train_config)
    run_context = RunContext(cb_params)
    ckpoint_cb.begin(run_context)
    ckpoint_cb.step_end(run_context)
    if os.path.exists('./test_files/test_ckpt-model.pkl'):
        os.chmod('./test_files/test_ckpt-model.pkl', stat.S_IWRITE)
        os.remove('./test_files/test_ckpt-model.pkl')


def test_loss_monitor_sink_mode():
    """Test loss monitor sink mode."""
    cb_params = _InternalCallbackParam()
    cb_params.cur_epoch_num = 4
    cb_params.cur_step_num = 2
    cb_params.batch_num = 2
    cb_params.net_outputs = Tensor(2.0)
    run_context = RunContext(cb_params)
    loss_cb = LossMonitor(1)
    callbacks = [loss_cb]
    callbacklist = _build_callbacks(callbacks)
    callbacklist.begin(run_context)
    callbacklist.epoch_begin(run_context)
    callbacklist.step_begin(run_context)
    callbacklist.step_end(run_context)
    callbacklist.epoch_end(run_context)
    callbacklist.end(run_context)


def test_loss_monitor_normal_mode():
    """Test loss monitor normal(non-sink) mode."""
    cb_params = _InternalCallbackParam()
    run_context = RunContext(cb_params)
    loss_cb = LossMonitor(1)
    cb_params.cur_epoch_num = 4
    cb_params.cur_step_num = 1
    cb_params.batch_num = 1
    cb_params.net_outputs = Tensor(2.0)
    loss_cb.begin(run_context)
    loss_cb.epoch_begin(run_context)
    loss_cb.step_begin(run_context)
    loss_cb.step_end(run_context)
    loss_cb.epoch_end(run_context)
    loss_cb.end(run_context)


def test_check_file_name_not_str():
    """Test check file name not str."""
    ret = _check_file_name_prefix(1)
    assert not ret


def test_check_file_name_back_err():
    """Test check file name back err."""
    ret = _check_file_name_prefix('abc.')
    assert ret


def test_check_file_name_one_alpha():
    """Test check file name one alpha."""
    ret = _check_file_name_prefix('a')
    assert ret
    ret = _check_file_name_prefix('_')
    assert ret


def test_check_file_name_err():
    """Test check file name err."""
    ret = _check_file_name_prefix('_123')
    assert ret


def test_chg_ckpt_file_name_if_same_exist():
    """Test chg ckpt file name if same exist."""
    _chg_ckpt_file_name_if_same_exist(directory="./test_files", prefix="ckpt")


def test_checkpoint_cb_for_save_op():
    """Test checkpoint cb for save op."""
    parameter_list = []
    one_param = {}
    one_param['name'] = "conv1.weight"
    one_param['data'] = Tensor(np.random.randint(0, 255, [1, 3, 224, 224]), dtype=mstype.float32)
    parameter_list.append(one_param)
    _checkpoint_cb_for_save_op(parameter_list)


def test_checkpoint_cb_for_save_op_update_net():
    """Test checkpoint cb for save op."""
    parameter_list = []
    one_param = {}
    one_param['name'] = "conv.weight"
    one_param['data'] = Tensor(np.ones(shape=(64, 3, 3, 3)), dtype=mstype.float32)
    parameter_list.append(one_param)
    net = Net()
    _set_cur_net(net)
    _checkpoint_cb_for_save_op(parameter_list)
    assert net.conv.weight.default_input.asnumpy()[0][0][0][0] == 1


def test_internal_callback_param():
    """Test Internal CallbackParam."""
    cb_params = _InternalCallbackParam()
    cb_params.member1 = 1
    cb_params.member2 = "abc"
    assert cb_params.member1 == 1
    assert cb_params.member2 == "abc"


def test_checkpoint_save_ckpt_steps():
    """Test checkpoint save ckpt steps."""
    train_config = CheckpointConfig(
        save_checkpoint_steps=16,
        save_checkpoint_seconds=0,
        keep_checkpoint_max=5,
        keep_checkpoint_per_n_minutes=0)
    ckpt_cb = ModelCheckpoint(config=train_config)
    cb_params = _InternalCallbackParam()
    net = Net()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    optim = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    network_ = WithLossCell(net, loss)
    _train_network = TrainOneStepCell(network_, optim)
    cb_params.train_network = _train_network
    cb_params.epoch_num = 10
    cb_params.cur_epoch_num = 5
    cb_params.cur_step_num = 160
    cb_params.batch_num = 32
    run_context = RunContext(cb_params)
    ckpt_cb.begin(run_context)
    ckpt_cb.step_end(run_context)
    ckpt_cb2 = ModelCheckpoint(config=train_config)
    cb_params.cur_epoch_num = 1
    cb_params.cur_step_num = 15
    ckpt_cb2.begin(run_context)
    ckpt_cb2.step_end(run_context)


def test_checkpoint_save_ckpt_seconds():
    """Test checkpoint save ckpt seconds."""
    train_config = CheckpointConfig(
        save_checkpoint_steps=16,
        save_checkpoint_seconds=100,
        keep_checkpoint_max=0,
        keep_checkpoint_per_n_minutes=1)
    ckpt_cb = ModelCheckpoint(config=train_config)
    cb_params = _InternalCallbackParam()
    net = Net()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    optim = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    network_ = WithLossCell(net, loss)
    _train_network = TrainOneStepCell(network_, optim)
    cb_params.train_network = _train_network
    cb_params.epoch_num = 10
    cb_params.cur_epoch_num = 4
    cb_params.cur_step_num = 128
    cb_params.batch_num = 32
    run_context = RunContext(cb_params)
    ckpt_cb.begin(run_context)
    ckpt_cb.step_end(run_context)
    ckpt_cb2 = ModelCheckpoint(config=train_config)
    cb_params.cur_epoch_num = 1
    cb_params.cur_step_num = 16
    ckpt_cb2.begin(run_context)
    ckpt_cb2.step_end(run_context)


def test_build_callbacks():
    """Test_build_callbacks."""
    ck_obj = ModelCheckpoint()
    loss_cb_1 = LossMonitor(1)

    callbacks = [None]
    with pytest.raises(TypeError):
        callbacks = _build_callbacks(callbacks)

    callbacks = ['Error']
    with pytest.raises(TypeError):
        callbacks = _build_callbacks(callbacks)

    callbacks = [ck_obj, loss_cb_1, 'Error', None]
    with pytest.raises(TypeError):
        _ = _build_callbacks(callbacks)


def test_RunContext():
    """Test RunContext."""
    context_err = 666
    with pytest.raises(TypeError):
        _ = RunContext(context_err)

    cb_params = _InternalCallbackParam()
    cb_params.member1 = 1
    cb_params.member2 = "abc"

    run_context = RunContext(cb_params)
    run_context.original_args()
    assert cb_params.member1 == 1
    assert cb_params.member2 == "abc"

    run_context.request_stop()
    should_stop = run_context.get_stop_requested()
    assert should_stop


def test_Checkpoint_Config():
    """Test CheckpointConfig all None or 0."""
    with pytest.raises(ValueError):
        CheckpointConfig(0, 0, 0, 0, True)

    with pytest.raises(ValueError):
        CheckpointConfig(0, None, 0, 0, True)


def test_step_end_save_graph():
    """Test save checkpoint."""
    train_config = CheckpointConfig(
        save_checkpoint_steps=16,
        save_checkpoint_seconds=0,
        keep_checkpoint_max=5,
        keep_checkpoint_per_n_minutes=0)
    cb_params = _InternalCallbackParam()
    net = LossNet()
    input_data = Tensor(np.random.randint(0, 255, [1, 3, 224, 224]).astype(np.float32))
    input_label = Tensor(np.random.randint(0, 3, [1, 3]).astype(np.float32))
    net(input_data, input_label)
    cb_params.train_network = net
    cb_params.epoch_num = 10
    cb_params.cur_epoch_num = 5
    cb_params.cur_step_num = 0
    cb_params.batch_num = 32
    ckpoint_cb = ModelCheckpoint(prefix="test", directory='./test_files', config=train_config)
    run_context = RunContext(cb_params)
    ckpoint_cb.begin(run_context)
    # import pdb;pdb.set_trace()
    ckpoint_cb.step_end(run_context)
    assert os.path.exists('./test_files/test-graph.meta') == True
    if os.path.exists('./test_files/test-graph.meta'):
        os.chmod('./test_files/test-graph.meta', stat.S_IWRITE)
        os.remove('./test_files/test-graph.meta')
    ckpoint_cb.step_end(run_context)
    assert os.path.exists('./test_files/test-graph.meta') == False
