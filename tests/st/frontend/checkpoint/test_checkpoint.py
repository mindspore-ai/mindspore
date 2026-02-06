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
"""Test checkpoint."""
import os
import shutil
import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore import log as logger
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.nn import Momentum
from mindspore.nn import TrainOneStepCell
from mindspore.nn import WithLossCell
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import _InternalCallbackParam
from mindspore.train.callback import RunContext
from mindspore.train.callback import _CallbackManager as _build_callbacks
from mindspore.train.serialization import load_checkpoint
from mindspore.train.serialization import load_param_into_net
from mindspore.train.serialization import save_checkpoint


def clean_directory(case_name):
    cur_dir = os.getcwd()
    dir_name = os.path.join(cur_dir, case_name)
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)


def find_file_list(base_path, prefix_name, subfix="ckpt"):
    checkpoint_file_list = []
    if os.path.exists(base_path):
        ls = os.listdir(base_path)
        logger.info("ls : {}".format(ls))
        for line in ls:
            if line.startswith(prefix_name) and line.endswith(subfix):
                file_path = os.path.join(base_path, line)
                checkpoint_file_list.append(file_path)
    return checkpoint_file_list


class ReLUReduceMeanDense(nn.Cell):
    def __init__(self, kernel, bias, in_channel, num_class):
        super().__init__()
        self.relu = ops.ReLU()
        self.mean = ops.ReduceMean(keep_dims=False)
        self.dense = nn.Dense(in_channel, num_class, kernel, bias)

    def construct(self, x):
        x = self.relu(x)
        x = self.mean(x, (2, 3))
        x = self.dense(x)
        return x


class CheckpointFactory:
    def __init__(self, input_shape, dense_input_c, output_c, has_bias=True, case_name=""):
        self.input_np = np.random.randn(*input_shape).astype(np.float32)
        self.input_np *= 0.01
        self.input_n = input_shape[0]
        self.input_channels = dense_input_c
        self.output_channels = output_c

        self.weight_np = np.ones((output_c, dense_input_c)).astype(np.float32) * 0.01
        self.has_bias = has_bias
        if self.has_bias is True:
            self.bias_np = np.ones((output_c,)).astype(np.float32) * 0.01
        else:
            self.bias = None
        self.label_np = np.random.randint(output_c, size=self.input_n)
        self.label_np_onehot = np.zeros(shape=(self.input_n, output_c)).astype(np.float32)
        self.label_np_onehot[np.arange(self.input_n), self.label_np] = 1

        self.input = Tensor(self.input_np.copy())
        self.weight = Tensor(self.weight_np.copy())
        self.bias = Tensor(self.bias_np.copy())
        self.label = Tensor(self.label_np_onehot.copy())
        self.cur_dir = os.path.join(os.getcwd(), case_name)
        self.case_dir = os.path.split(__file__)[0]
        self.case_data = os.path.join(self.case_dir, "data")

    def net_train(self, train_network, epoch, cb_list):
        loss_list = []
        _should_stop = False
        cb_params = _InternalCallbackParam()
        cb_params.train_network = train_network
        cb_params.epoch_num = epoch
        cb_params.batch_num = 1
        cb_params.cur_step_num = 0
        cb_params.train_mode = "graph"
        run_context = RunContext(cb_params)
        callbacks = _build_callbacks(cb_list)
        callbacks.begin(run_context)
        for i in range(epoch):
            callbacks.epoch_begin(run_context)
            callbacks.step_begin(run_context)
            loss = train_network(self.input, self.label)
            cb_params.net_outputs = loss
            cb_params.train_network = train_network
            cb_params.cur_epoch_num = i + 1
            cb_params.cur_step_num = (i + 1) * cb_params.batch_num
            callbacks.step_end(run_context)
            callbacks.epoch_end(run_context)
            loss_list.append(loss.asnumpy())
            _should_stop = _should_stop or run_context.get_stop_requested()
            if _should_stop:
                break
        callbacks.end(run_context)
        return loss_list

    def train_with_checkpoint_callback_and_infer(self, net, epoch, save_file_steps=1,
                                                 keep_file_max=5,
                                                 prefix_name="CKPT", save_dir="./", config_tag=True,
                                                 async_save=False):
        criterion = SoftmaxCrossEntropyWithLogits(sparse=False)
        optimizer = Momentum(learning_rate=0.1, momentum=0.1, params=net.trainable_params())
        net_with_criterion = WithLossCell(net, criterion)
        train_network = TrainOneStepCell(net_with_criterion, optimizer)
        train_network.set_train()

        if config_tag is True:
            config = CheckpointConfig(save_checkpoint_steps=save_file_steps,
                                      keep_checkpoint_max=keep_file_max, async_save=async_save)
            ckpoint_cb = ModelCheckpoint(prefix=prefix_name, directory=save_dir, config=config)
        elif config_tag is None:
            ckpoint_cb = ModelCheckpoint(prefix=prefix_name, directory=save_dir, config=None)
        else:
            ckpoint_cb = ModelCheckpoint()
        self.net_train(train_network, epoch, ckpoint_cb)
        output = net(self.input)
        return output.asnumpy()

    def load_checkpoint_into_net_and_infer(self, net, file_name):
        data_dict = load_checkpoint(ckpt_file_name=file_name)
        load_param_into_net(net, data_dict)
        output = net(self.input)
        return output.asnumpy()

    def checkpoint_save_load_check(self, net1, net2, epoch, steps, max_num, pre_name, path,
                                   file_name, tag, assert_flag=True, async_save=False):
        infer1 = self.train_with_checkpoint_callback_and_infer(net1, epoch, steps, max_num,
                                                               pre_name, path, tag,
                                                               async_save=async_save)
        infer2 = self.load_checkpoint_into_net_and_infer(net2, file_name)
        logger.info("infer1: {},\n infer2:{}".format(infer1, infer2))
        if assert_flag is True:
            assert np.allclose(infer1, infer2, 0.0, 0.0)
        return infer1

    @staticmethod
    def checkpoint_files_num_check(prefix_name, file_path, expr_num):
        ckpt_file_list = find_file_list(file_path, prefix_name)
        logger.info("find checkpoint file list. {}".format(ckpt_file_list))
        assert len(ckpt_file_list) == expr_num


@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("async_save", [True, False])
def test_checkpoint_config_epoch2_steps1_max2(context_mode, async_save):
    """
    Feature: test checkpoint.
    Description: test checkpoint.
    Expectation: the result match with expected result.
    """
    ms.context.set_context(mode=context_mode)
    case_name = "test_checkpoint_config_epoch2_steps2_max2_{}".format(async_save)
    clean_directory(case_name)
    fact = CheckpointFactory(input_shape=(32, 2048, 7, 7), dense_input_c=2048, output_c=1000,
                             has_bias=True, case_name=case_name)
    net1 = ReLUReduceMeanDense(fact.weight, fact.bias, fact.input_channels, fact.output_channels)
    net2 = ReLUReduceMeanDense(fact.weight, fact.bias, fact.input_channels, fact.output_channels)
    file_name = os.path.join(fact.cur_dir, "CKPT-2_1.ckpt")
    fact.checkpoint_save_load_check(net1, net2, 2, 2, 2, "CKPT", fact.cur_dir, file_name, True,
                                    async_save=async_save)
    fact.checkpoint_files_num_check("CKPT", fact.cur_dir, 1)


@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("integrated_save", [True, False])
def test_checkpoint_save_checkpoint_param_save_obj_set_init_net_obj(context_mode, integrated_save):
    """
    Feature: test checkpoint.
    Description: test checkpoint.
    Expectation: the result match with expected result.
    """
    ms.context.set_context(mode=context_mode)
    fact = CheckpointFactory(input_shape=(32, 2048, 7, 7), dense_input_c=2048, output_c=1000,
                             has_bias=True)
    net = ReLUReduceMeanDense(fact.weight, fact.bias, fact.input_channels, fact.output_channels)
    save_checkpoint(net, "save_init_net.ckpt", integrated_save=integrated_save)
    ckpt_dict = load_checkpoint("save_init_net.ckpt")
    np.allclose(net.dense.weight.asnumpy(), ckpt_dict["dense.weight"].data.asnumpy(), 0.0001,
                     0.0001)
