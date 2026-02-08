# Copyright 2020-2025 Huawei Technologies Co., Ltd
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
import platform
import stat
import secrets
from unittest import mock

import numpy as np
import pytest

from mindspore import context
import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore.graph.api import jit
from mindspore.common.tensor import Tensor
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import Momentum
from mindspore.train import ModelCheckpoint, RunContext, LossMonitor, Callback, CheckpointConfig, \
    LambdaCallback, History, TrainFaultTolerance, OnRequestExit
from mindspore.train.callback import _InternalCallbackParam, _CallbackManager, _checkpoint_cb_for_save_op, _set_cur_net
from mindspore.train.callback._checkpoint import _chg_ckpt_file_name_if_same_exist


class Net(nn.Cell):
    """Net definition."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, has_bias=False, weight_init='normal')
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(64 * 222 * 222, 3)

    @jit
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
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, has_bias=False, weight_init='normal', pad_mode='valid')
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(64 * 222 * 222, 3)  # padding=0
        self.loss = nn.SoftmaxCrossEntropyWithLogits()

    @jit
    def construct(self, x, y):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        out = self.loss(x, y)
        return out


def test_model_checkpoint_prefix_invalid():
    """
    Feature: callback
    Description: Test ModelCheckpoint prefix invalid
    Expectation: run success
    """
    with pytest.raises(ValueError):
        ModelCheckpoint(123)
    ModelCheckpoint(directory="./")
    with pytest.raises(TypeError):
        ModelCheckpoint(config='type_error')
    ModelCheckpoint(config=CheckpointConfig())
    ModelCheckpoint(prefix="ckpt_2", directory="./test_files")


def test_loss_monitor_sink_mode():
    """
    Feature: callback
    Description: Test loss monitor sink mode
    Expectation: run success
    """
    cb_params = _InternalCallbackParam()
    cb_params.cur_epoch_num = 4
    cb_params.epoch_num = 4
    cb_params.cur_step_num = 2
    cb_params.batch_num = 2
    cb_params.net_outputs = Tensor(2.0)
    run_context = RunContext(cb_params)
    loss_cb = LossMonitor(1)
    callbacks = [loss_cb]
    with _CallbackManager(callbacks) as callbacklist:
        callbacklist.begin(run_context)
        callbacklist.epoch_begin(run_context)
        callbacklist.step_begin(run_context)
        callbacklist.step_end(run_context)
        callbacklist.epoch_end(run_context)
        callbacklist.end(run_context)


def test_loss_monitor_normal_mode():
    """
    Feature: callback
    Description: Test loss monitor normal(non-sink) mode
    Expectation: run success
    """
    cb_params = _InternalCallbackParam()
    run_context = RunContext(cb_params)
    loss_cb = LossMonitor(1)
    cb_params.cur_epoch_num = 4
    cb_params.epoch_num = 4
    cb_params.cur_step_num = 1
    cb_params.batch_num = 1
    cb_params.net_outputs = Tensor(2.0)
    loss_cb.begin(run_context)
    loss_cb.epoch_begin(run_context)
    loss_cb.step_begin(run_context)
    loss_cb.step_end(run_context)
    loss_cb.epoch_end(run_context)
    loss_cb.end(run_context)


def test_loss_monitor_args():
    """
    Feature: callback
    Description: Test loss monitor illegal args
    Expectation: run success
    """
    with pytest.raises(ValueError):
        LossMonitor(per_print_times=-1)


def test_save_ckpt_and_test_chg_ckpt_file_name_if_same_exist():
    """
    Feature: Save checkpoint and check if there is a file with the same name.
    Description: Save checkpoint and check if there is a file with the same name.
    Expectation: Checkpoint is saved and checking is successful.
    """
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
    _chg_ckpt_file_name_if_same_exist(directory="./test_files", prefix="ckpt")


def test_checkpoint_cb_for_save_op():
    """
    Feature: callback
    Description: Test checkpoint cb for save op
    Expectation: run success
    """
    parameter_list = []
    one_param = {}
    one_param['name'] = "conv1.weight"
    one_param['data'] = Tensor(np.random.randint(0, 255, [1, 3, 224, 224]), dtype=mstype.float32)
    parameter_list.append(one_param)
    _checkpoint_cb_for_save_op(parameter_list)


def test_checkpoint_cb_for_save_op_update_net():
    """
    Feature: callback
    Description: Test checkpoint cb for save op
    Expectation: run success
    """
    parameter_list = []
    one_param = {}
    one_param['name'] = "conv.weight"
    one_param['data'] = Tensor(np.ones(shape=(64, 3, 3, 3)), dtype=mstype.float32)
    parameter_list.append(one_param)
    net = Net()
    _set_cur_net(net)
    _checkpoint_cb_for_save_op(parameter_list)
    assert net.conv.weight.data.asnumpy()[0][0][0][0] == 1


def test_internal_callback_param():
    """
    Feature: callback
    Description: Test Internal CallbackParam
    Expectation: run success
    """
    cb_params = _InternalCallbackParam()
    cb_params.member1 = 1
    cb_params.member2 = "abc"
    assert cb_params.member1 == 1
    assert cb_params.member2 == "abc"


def test_checkpoint_save_ckpt_steps():
    """
    Feature: callback
    Description: Test checkpoint save ckpt steps
    Expectation: run success
    """
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
    """
    Feature: callback
    Description: Test checkpoint save ckpt seconds
    Expectation: run success
    """
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


def test_checkpoint_save_ckpt_with_encryption():
    """
    Feature: callback
    Description: Test checkpoint save ckpt with encryption
    Expectation: run success
    """
    train_config = CheckpointConfig(
        save_checkpoint_steps=16,
        save_checkpoint_seconds=0,
        keep_checkpoint_max=5,
        keep_checkpoint_per_n_minutes=0,
        enc_key=secrets.token_bytes(16),
        enc_mode="AES-GCM")
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

    if platform.system().lower() == "windows":
        with pytest.raises(NotImplementedError):
            ckpt_cb2.begin(run_context)
            ckpt_cb2.step_end(run_context)
    else:
        ckpt_cb2.begin(run_context)
        ckpt_cb2.step_end(run_context)


def test_callbackmanager():
    """
    Feature: callback
    Description: Test CallbackManager
    Expectation: run success
    """
    ck_obj = ModelCheckpoint()
    loss_cb_1 = LossMonitor(1)

    callbacks = [None]
    with pytest.raises(TypeError):
        _CallbackManager(callbacks)

    callbacks = ['Error']
    with pytest.raises(TypeError):
        _CallbackManager(callbacks)

    callbacks = [ck_obj, loss_cb_1, 'Error', None]
    with pytest.raises(TypeError):
        _CallbackManager(callbacks)


def test_callbackmanager_exit_called():
    """
    Feature: callback
    Description: Test CallbackManager exit called
    Expectation: run success
    """
    with mock.patch.object(Callback, '__exit__', return_value=None) as mock_exit:
        cb1, cb2 = Callback(), Callback()
        with _CallbackManager([cb1, cb2]):
            pass
    for call_args in mock_exit.call_args_list:
        assert call_args == mock.call(mock.ANY, None, None, None)
    assert mock_exit.call_count == 2


def prefix_func(cb_params):
    return str(cb_params.cur_step_num) + "_custom_file"


def directory_func(cb_params):
    return "./custom_ckpt"


def custom_checkpoint_dir_and_prefix(prefix, directory):
    train_config = CheckpointConfig(
        save_checkpoint_steps=16,
        save_checkpoint_seconds=100,
        keep_checkpoint_max=0,
        keep_checkpoint_per_n_minutes=1)
    ckpt_cb = ModelCheckpoint(prefix=prefix, directory=directory, config=train_config)
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


def test_checkpoint_custom_dir_and_prefix():
    """
    Feature: callback
    Description: Test checkpoint save ckpt with custom dir and custom prefix
    Expectation: run success
    """
    custom_checkpoint_dir_and_prefix(prefix_func, directory_func)
    ckpt_name = './custom_ckpt/128_custom_file.ckpt'
    assert os.path.exists(ckpt_name)
    os.chmod(ckpt_name, stat.S_IWRITE)
    os.remove(ckpt_name)


def test_checkpoint_custom_dir():
    """
    Feature: callback
    Description: Test checkpoint save ckpt with custom dir
    Expectation: run success
    """
    custom_checkpoint_dir_and_prefix("100", directory_func)
    ckpt_name = './custom_ckpt/100-4_32.ckpt'
    assert os.path.exists(ckpt_name)
    os.chmod(ckpt_name, stat.S_IWRITE)
    os.remove(ckpt_name)


def test_checkpoint_custom_prefix():
    """
    Feature: callback
    Description: Test checkpoint save ckpt with custom prefix
    Expectation: run success
    """
    custom_checkpoint_dir_and_prefix(prefix_func, "./ckpt_path")
    ckpt_name = './ckpt_path/128_custom_file.ckpt'
    assert os.path.exists(ckpt_name)
    os.chmod(ckpt_name, stat.S_IWRITE)
    os.remove(ckpt_name)


def test_callbackmanager_exit_called_when_raises():
    """
    Feature: callback
    Description: Test when CallbackManager exit called
    Expectation: run success
    """
    with mock.patch.object(Callback, '__exit__', return_value=None) as mock_exit:
        cb1, cb2 = Callback(), Callback()
        with pytest.raises(ValueError):
            with _CallbackManager([cb1, cb2]):
                raise ValueError()
    for call_args in mock_exit.call_args_list:
        assert call_args == mock.call(*[mock.ANY] * 4)
    assert mock_exit.call_count == 2


def test_callbackmanager_begin_called():
    """
    Feature: callback
    Description: Test CallbackManager called begin
    Expectation: run success
    """
    run_context = dict()  # pylint: disable=R1735
    with mock.patch.object(Callback, 'begin', return_value=None) as mock_begin:
        cb1, cb2 = Callback(), Callback()
        with _CallbackManager([cb1, cb2]) as cm:
            cm.begin(run_context)
    for call_args in mock_begin.call_args_list:
        assert call_args == mock.call(run_context)
    assert mock_begin.call_count == 2


def test_runcontext():
    """
    Feature: callback
    Description: Test RunContext init
    Expectation: run success
    """
    context_err = 666
    with pytest.raises(TypeError):
        RunContext(context_err)

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


def test_checkpoint_config():
    """
    Feature: callback
    Description: Test checkpoint config error args
    Expectation: run success
    """
    with pytest.raises(ValueError):
        CheckpointConfig(0, 0, 0, 0, True)

    with pytest.raises(ValueError):
        CheckpointConfig(0, None, 0, 0, True)


def test_step_end_save_graph():
    """
    Feature: callback
    Description: Test save graph at step end
    Expectation: run success
    """
    context.set_context(mode=context.GRAPH_MODE)
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
    ckpoint_cb.step_end(run_context)
    assert os.path.exists('./test_files/test-graph.meta')
    if os.path.exists('./test_files/test-graph.meta'):
        os.chmod('./test_files/test-graph.meta', stat.S_IWRITE)
        os.remove('./test_files/test-graph.meta')
    ckpoint_cb.step_end(run_context)
    assert not os.path.exists('./test_files/test-graph.meta')


def test_history():
    """
    Feature: callback.
    Description: Test history object saves epoch and history properties.
    Expectation: run success.
    """
    cb_params = _InternalCallbackParam()
    cb_params.cur_epoch_num = 4
    cb_params.epoch_num = 4
    cb_params.cur_step_num = 2
    cb_params.batch_num = 2
    cb_params.net_outputs = Tensor(2.0)
    cb_params.metrics = {'mae': 6.343789100646973, 'mse': 59.03999710083008}

    run_context = RunContext(cb_params)
    history_cb = History()
    callbacks = [history_cb]
    with _CallbackManager(callbacks) as callbacklist:
        callbacklist.begin(run_context)
        callbacklist.epoch_begin(run_context)
        callbacklist.step_begin(run_context)
        callbacklist.step_end(run_context)
        callbacklist.epoch_end(run_context)
        callbacklist.end(run_context)
    print(history_cb.epoch)
    print(history_cb.history)


def test_lambda():
    """
    Feature: callback.
    Description: Test lambda callback.
    Expectation: run success.
    """
    cb_params = _InternalCallbackParam()
    cb_params.cur_epoch_num = 4
    cb_params.epoch_num = 4
    cb_params.cur_step_num = 2
    cb_params.batch_num = 2
    cb_params.net_outputs = Tensor(2.0)

    run_context = RunContext(cb_params)
    lambda_cb = LambdaCallback(
        on_train_epoch_end=lambda run_context: print("loss result: ", run_context.original_args().net_outputs))

    callbacks = [lambda_cb]
    with _CallbackManager(callbacks) as callbacklist:
        callbacklist.on_train_begin(run_context)
        callbacklist.on_train_epoch_begin(run_context)
        callbacklist.on_train_step_begin(run_context)
        callbacklist.on_train_step_end(run_context)
        callbacklist.on_train_epoch_end(run_context)
        callbacklist.on_train_end(run_context)


def test_graceful_exit():
    """
    Feature: test graceful exit.
    Description: test graceful exit callback.
    Expectation: run success
    """
    os.environ["MS_ENABLE_GRACEFUL_EXIT"] = '1'
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    net = LossNet()
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    net = TrainOneStepCell(net, optimizer)
    input_data = Tensor(np.random.randint(0, 255, [1, 3, 224, 224]).astype(np.float32))
    input_label = Tensor(np.random.randint(0, 3, [1, 3]).astype(np.float32))
    net(input_data, input_label)

    cb_params = _InternalCallbackParam()
    cb_params.cur_epoch_num = 4
    cb_params.epoch_num = 4
    cb_params.cur_step_num = 2
    cb_params.batch_num = 2
    cb_params.train_network = net
    run_context = RunContext(cb_params)

    # cb function
    graceful_cb = OnRequestExit(save_ckpt=False, save_mindir=False, config_file="./reset.json")
    assert graceful_cb.save_path == "./"
    assert graceful_cb.use_graceful is True
    ret = graceful_cb.remote_config_file == "./reset.json"
    assert ret is True
    assert graceful_cb.is_distributed is False
    # 1. single device
    graceful_cb.on_train_step_begin(run_context)
    graceful_cb.on_train_step_end(run_context)
    del os.environ["MS_ENABLE_GRACEFUL_EXIT"]


def test_mindio_ttp_adapter():
    """
    Feature: callback.
    Description: Test tft register callback.
    Expectation: run success.
    """
    import sys
    from unittest.mock import MagicMock
    # create a temp module for 'mindio_ttp'
    mindio_ttp = type('module', (), {})()
    mindio_ttp.framework_ttp = type('module', (), {'tft_init_processor': MagicMock(return_value=None),
                                                   'tft_start_processor': MagicMock(return_value=None),
                                                   "tft_is_reboot_node": MagicMock(return_value=False),
                                                   "tft_register_save_ckpt_handler": MagicMock(return_value=None),
                                                   "tft_register_rename_handler": MagicMock(return_value=None),
                                                   "tft_register_exit_handler": MagicMock(return_value=None),
                                                   "tft_register_stop_handler": MagicMock(return_value=None),
                                                   "tft_register_clean_handler": MagicMock(return_value=None),
                                                   "tft_register_repair_handler": MagicMock(return_value=None),
                                                   "tft_register_rebuild_group_handler": MagicMock(return_value=None),
                                                   "tft_register_stream_sync_handler": MagicMock(return_value=None),
                                                   "tft_set_step_args": MagicMock(return_value=None),
                                                   "tft_init_controller": MagicMock(return_value=None),
                                                   "tft_start_controller": MagicMock(return_value=None)
                                                   }
                                    )()

    mindio_ttp.controller_ttp = type('module', (), {'tft_register_mindx_callback': MagicMock(return_value=None),
                                                    'tft_notify_controller_stop_train': MagicMock(return_value=None),
                                                    'tft_notify_controller_on_global_rank': MagicMock(
                                                        return_value=None),
                                                    'tft_notify_controller_change_strategy': MagicMock(
                                                        return_value=None),
                                                    'tft_notify_controller_prepare_action': MagicMock(
                                                        return_value=None)
                                                    }
                                     )()

    sys.modules['mindio_ttp'] = mindio_ttp
    sys.modules['mindio_ttp.framework_ttp'] = mindio_ttp.framework_ttp
    sys.modules['mindio_ttp.controller_ttp'] = mindio_ttp.controller_ttp

    with mock.patch("mindspore.hal.Stream", return_value={"test": 1}):
        # case1: not set MS_ENABLE_TFT, raise ERROR
        with pytest.raises(ValueError):
            mindio_cb = TrainFaultTolerance(ckpt_save_path='./ckpt')

        # case 2: must init advance when using ARF
        context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
        os.environ["MS_ENABLE_TFT"] = '{TFT:1,UCE:1,ARF:1,TSP:1}'
        with pytest.raises(ValueError):
            mindio_cb = TrainFaultTolerance(ckpt_save_path='./ckpt')

        # case 3: disable hccl watchdog when using ARF
        from mindspore.utils import _tft_handler
        os.environ["MS_TFT_PORT"] = str(8000)
        os.environ["MS_TFT_IP"] = "127.0.0.1"
        os.environ["MS_WORKER_NUM"] = "8"
        os.environ["MS_NODE_ID"] = "0"
        key_value = {"any": 0}
        _tft_handler.init(config=key_value)

        # case 4: if sink_size > 1, raise error
        os.environ["MS_ENABLE_TFT"] = '{TFT:1,UCE:1,TSP:1}'
        cb_params = _InternalCallbackParam()
        cb_params.cur_epoch_num = 4
        cb_params.epoch_num = 4
        cb_params.cur_step_num = 2
        cb_params.dataset_sink_mode = True
        cb_params.sink_size = 2
        run_context = RunContext(cb_params)
        mindio_cb = TrainFaultTolerance(ckpt_save_path='./ckpt')
        with pytest.raises(ValueError):
            mindio_cb.on_train_begin(run_context)


def test_tft_adapter_interface():
    """
    Feature: tft adapter interface.
    Description: Test tft adapter interface.
    Expectation: run success.
    """
    from mindspore._c_expression import set_is_reboot_node, is_reboot_node, check_is_arf, set_is_arf

    reboot_flag = is_reboot_node()
    assert reboot_flag is False
    arf_flag = check_is_arf()
    assert arf_flag is False

    set_is_reboot_node(True)
    set_is_arf(True)

    reboot_flag = is_reboot_node()
    assert reboot_flag is True
    arf_flag = check_is_arf()
    assert arf_flag is True


def test_taskd_plugin():
    """
    Feature: callback.
    Description: Test taskd register callback.
    Expectation: run success.
    """
    import sys
    from unittest.mock import MagicMock
    from mindspore.utils import RSCPluginHandle

    mock_plugin = MagicMock()
    sys.modules['taskd'] = MagicMock()
    sys.modules['taskd.python'] = MagicMock()
    sys.modules['taskd.python.framework'] = MagicMock()
    sys.modules['taskd.python.framework.agent'] = MagicMock()
    sys.modules['taskd.python.framework.agent.ms_mgr'] = MagicMock()
    sys.modules['taskd.python.framework.agent.ms_mgr.msrun_plugin'] = mock_plugin

    handle = RSCPluginHandle()
    enable = handle.check_enable()
    assert enable is False
    handle.tft_env = "{RSC:1}"
    handle._check_env()  # pylint: disable=W0212
    enable = handle.check_enable()
    assert enable is True

    def case1():
        """exception :No module named MSRunPlugin"""
        mock_plugin.MSRunPlugin.side_effect = Exception("No module named MSRunPlugin")
        assert handle.register_callback({"Start": None, "Stop": None}) is False

    def case2():
        """Module exist"""
        mock_plugin.MSRunPlugin.side_effect = None
        assert handle.register_callback({"Start": None, "Stop": None}) is True

    def case3():
        """No method: register_callbacks"""
        mock_plugin.MSRunPlugin.return_value.register_callbacks.side_effect = Exception("register_callbacks not exist")
        assert handle.register_callback({"Start": None, "Stop": None}) is False

    def case4():
        """register_callbacks exist"""
        mock_plugin.MSRunPlugin.return_value.register_callbacks.side_effect = None
        assert handle.register_callback({"Start": None, "Stop": None}) is True

    def case5():
        """handle is None"""
        with pytest.raises(RuntimeError):
            handle.register_callback({"Start": None, "Stop": None})
            handle.msmgr = None
            handle.start()

    def case6():
        """Normal"""
        _ = handle.register_callback({"Start": None, "Stop": None})
        try:
            handle.start()
        except Exception as e:  # pylint: disable=broad-except
            print(f"exception:{str(e)}")
            return False
        return True

    case1()
    case2()
    case3()
    case4()
    case5()
    ret = case6()
    assert ret is True
