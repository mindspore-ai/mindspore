# Copyright 2026 Huawei Technologies Co., Ltd
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
Test module for parameter increment initialization.
"""
import numpy as np
import os
import torch
from torch import nn
import mindspore as ms
from mindspore import log as logger, context
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell, Conv2d, Dense, Momentum, SoftmaxCrossEntropyWithLogits, TrainOneStepCell, WithLossCell
from mindspore.train.callback import Callback, LossMonitor, ModelCheckpoint, RunContext, _InternalCallbackParam, _CallbackManager as _build_callbacks
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from tests.st.frontend.dataset.animal import create_animal_no_random_dataset
from tests.mark_utils import arg_mark


class Conv2dReduceMeanL2(Cell):
    def __init__(self, layer1):
        super().__init__()
        self.conv = layer1
        self.reduce = P.ReduceMean()

    def construct(self, inputs):
        out = self.conv(inputs)
        out = self.reduce(out, (2, 3))
        return out


class Conv2dReduceMeanL3(Cell):
    def __init__(self, layer1, layer3):
        super().__init__()
        self.conv = layer1
        self.reduce = P.ReduceMean()
        self.dense = layer3

    def construct(self, inputs):
        out = self.conv(inputs)
        out = self.reduce(out, (2, 3))
        out = self.dense(out)
        return out


class PtConvReduceNetL2(nn.Module):
    def __init__(self, layer1):
        super().__init__()
        self.conv = layer1

    def forward(self, x):
        out = self.conv(x)
        out = torch.mean(out, (2, 3))
        return out


class PtConvReduceNetL3(nn.Module):
    def __init__(self, net1, layer3):
        super().__init__()
        self.net = net1
        self.dense = layer3

    def forward(self, x):
        out = self.net(x)
        out = self.dense(out)
        return out


class ParamCallback(Callback):
    def __init__(self):
        super().__init__()
        self.loss_list = []

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        logger.info("me loss callback: {}".format(cb_params.net_outputs.asnumpy()))
        self.loss_list.append(cb_params.net_outputs.asnumpy())


class MetaFactory:
    """
    define class MetaFactory
    """
    def __init__(self):
        """
        define init for MetaFactory
        """
        self.device_target = context.get_context('device_target')
        self.rank_size = None
        self.device_id = None
        self.global_rank_id = None


class ParameterFactory(MetaFactory):
    def __init__(self, epoch_size=10, batch_size=32):
        super().__init__()
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.num_classes = 12
        self.me_layer1 = Conv2d(in_channels=3, out_channels=12, kernel_size=1, pad_mode='valid',
                                has_bias=False, weight_init='ones')
        self.me_layer3 = Dense(in_channels=12, out_channels=12, weight_init='ones', has_bias=False)
        self.pt_layer1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=1, bias=False)
        self.pt_layer3 = nn.Linear(in_features=12, out_features=12, bias=False)
        self.conv_weight = torch.from_numpy(np.ones([12, 3, 1, 1]).astype(np.float32))
        self.dense_weight = torch.from_numpy(np.ones([12, 12]).astype(np.float32))

    def me_model_train_dataset(self):
        dataset = image_data_proc(epoch=self.epoch_size)
        net = Conv2dReduceMeanL2(self.me_layer1)
        loss = SoftmaxCrossEntropyWithLogits(sparse=False, reduction='mean')
        opt = Momentum(learning_rate=0.1, momentum=2.0, params=net.get_parameters())
        model = Model(net, loss, opt)
        loss_cb = ParamCallback()
        model.train(self.epoch_size, dataset, callbacks=[loss_cb], dataset_sink_mode=True)

        logger.info("me conv_reduce net loss: ", loss_cb.loss_list)
        return net, loss_cb.loss_list

    def me_model_train_dataset_train_eval(self):
        dataset_train = image_data_proc(epoch=self.epoch_size)
        dataset_eval = image_data_proc(epoch=self.epoch_size)
        net1 = Conv2dReduceMeanL2(self.me_layer1)
        net2 = Conv2dReduceMeanL3(self.me_layer1, self.me_layer3)

        loss = SoftmaxCrossEntropyWithLogits(sparse=False, reduction='mean')
        opt = Momentum(learning_rate=0.1, momentum=2.0, params=net1.get_parameters())
        model1 = Model(net1, loss, opt)
        loss_cb = ParamCallback()
        model1.train(self.epoch_size, dataset_train, callbacks=[LossMonitor(), loss_cb])

        model2 = Model(net2, loss, metrics={'acc', 'loss'})
        out = model2.eval(dataset_eval)
        logger.info("me model eval result:", out)
        return out

    @staticmethod
    def net_train(train_network, epoch, dataset, cb_list=None):
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
        i = 0
        for i in range(epoch):
            for data in dataset:
                loss = train_network(Tensor(data[0]), Tensor(data[1]))
                loss_list.append(loss.asnumpy())
            callbacks.epoch_begin(run_context)
            callbacks.step_begin(run_context)
            cb_params.net_outputs = loss
            cb_params.train_network = train_network
            cb_params.cur_epoch_num = i + 1
            cb_params.cur_step_num = (i + 1) * cb_params.batch_num
            callbacks.step_end(run_context)
            callbacks.epoch_end(run_context)
            _should_stop = _should_stop or run_context.get_stop_requested()
            if _should_stop:
                break
            i += 1
        callbacks.end(run_context)
        return loss_list

    def train_dataset_onestepcell_get_loss(self, repeat_flag=False, checkpoint_file=None):
        dataset = image_data_proc(epoch=self.epoch_size)
        net1 = Conv2dReduceMeanL2(self.me_layer1)
        criterion = SoftmaxCrossEntropyWithLogits(sparse=False, reduction='mean')
        if repeat_flag is True:
            dicts = load_checkpoint(checkpoint_file)
            optimizer = Momentum(learning_rate=0.1, momentum=2.0, params=net1.get_parameters())
            load_param_into_net(optimizer, dicts)
        else:
            optimizer = Momentum(learning_rate=0.1, momentum=2.0, params=net1.get_parameters())

        net_with_criterion = WithLossCell(net1, criterion)
        train_network = TrainOneStepCell(net_with_criterion, optimizer)
        train_network.set_train()
        losses = self.net_train(train_network, self.epoch_size, dataset, cb_list=None)
        logger.info("me train one step loss: ", losses)
        return losses


def image_data_proc(epoch=3):
    dataset = create_animal_no_random_dataset(epoch_size=epoch, batch_size=32)
    dataset.num_classes()
    return dataset


def get_data_loader():
    ds = create_animal_no_random_dataset(epoch_size=1)
    for data in ds.create_tuple_iterator(output_numpy=True):
        images = torch.from_numpy(data[0])
        target = torch.from_numpy(np.argmax(data[1], axis=1))
        dataset = torch.utils.data.dataset.TensorDataset(images, target)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    return loader


def _count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me) * rtol)
    nan_diff = np.not_equal(np.isnan(data_expected), np.isnan(data_me))
    inf_diff = np.not_equal(np.isinf(data_expected), np.isinf(data_me))
    neginf_diff = np.not_equal(np.isneginf(data_expected), np.isneginf(data_me))
    greater = greater + nan_diff + inf_diff + neginf_diff
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
            format(data_expected[greater], data_me[greater], error[greater])


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert np.array(data_expected).shape == np.array(data_me).shape


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', mem_peak=512.00)
def test_parameter_increm_init():
    """
    Feature: Parameter checkpoint loading and inference consistency
    Description: Train two identical models with different epochs, save checkpoint from first model,
                 then load parameters into second model and verify inference consistency
    Expectation: Result is correct
    """
    ms.runtime.launch_blocking()

    class Conv2dReduceMean1(Cell):
        def __init__(self):
            super().__init__()
            self.conv = Conv2d(in_channels=3, out_channels=12, kernel_size=1, pad_mode='valid',
                               has_bias=False, weight_init='ones')
            self.reduce = P.ReduceMean()

        def construct(self, inputs):
            out = self.conv(inputs)
            out = self.reduce(out, (2, 3))
            return out

    ds = image_data_proc(1)
    for data, _ in ds.create_tuple_iterator(output_numpy=True):
        inputs = data

    # Train net1 for 3 epochs and perform prediction
    fact = ParameterFactory(epoch_size=3)
    dataset_train = image_data_proc(fact.epoch_size)
    net1 = Conv2dReduceMean1()
    net2 = Conv2dReduceMean1()
    loss = SoftmaxCrossEntropyWithLogits(sparse=False, reduction='mean')
    opt = Momentum(learning_rate=0.1, momentum=2.0, params=net1.get_parameters())
    model1 = Model(net1, loss, opt)
    loss_cb = ParamCallback()
    ckpoint_cb = ModelCheckpoint(prefix="train", directory="./")
    model1.train(fact.epoch_size, dataset_train, callbacks=[LossMonitor(), loss_cb, ckpoint_cb])
    out1 = model1.predict(Tensor(inputs))

    # Train net2 for 5 epochs and perform prediction
    fact2 = ParameterFactory(epoch_size=5)
    dataset_train2 = image_data_proc(fact2.epoch_size)
    loss2 = SoftmaxCrossEntropyWithLogits(sparse=False, reduction='mean')
    opt2 = Momentum(learning_rate=0.1, momentum=2.0, params=net2.get_parameters())
    model2 = Model(net2, loss2, opt2)
    model2.train(fact2.epoch_size, dataset_train2, callbacks=[LossMonitor()])
    model2.predict(Tensor(inputs))

    # Load parameters into net2 and perform prediction
    dicts = load_checkpoint(os.path.join("./", "train-3_1.ckpt"), net=net2)
    load_param_into_net(opt2, dicts)
    out3 = model2.predict(Tensor(inputs))

    allclose_nparray(out1.asnumpy(), out3.asnumpy(), 0, 0)
