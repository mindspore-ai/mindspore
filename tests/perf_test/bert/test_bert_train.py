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

"""Bert test."""

# pylint: disable=missing-docstring, arguments-differ, W0612

import os

import mindspore.common.dtype as mstype
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.nn.optim import AdamWeightDecay
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.nn import learning_rate_schedule as lr_schedules
from tests.models.official.nlp.bert.src import BertConfig, BertNetworkWithLoss, BertTrainOneStepCell, BertTrainOneStepWithLossScaleCell
from ...dataset_mock import MindData
from ...ops_common import nn, np, batch_tuple_tensor, build_construct_graph


_current_dir = os.path.dirname(os.path.realpath(__file__)) + "/../python/test_data"
context.set_context(mode=context.GRAPH_MODE)


def get_dataset(batch_size=1):
    dataset_types = (np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32)
    dataset_shapes = ((batch_size, 128), (batch_size, 128), (batch_size, 128), (batch_size, 1), \
                      (batch_size, 20), (batch_size, 20), (batch_size, 20))

    dataset = MindData(size=2, batch_size=batch_size,
                       np_types=dataset_types,
                       output_shapes=dataset_shapes,
                       input_indexs=(0, 1))
    return dataset


def load_test_data(batch_size=1):
    dataset = get_dataset(batch_size)
    ret = dataset.next()
    ret = batch_tuple_tensor(ret, batch_size)
    return ret


def get_config(version='base'):
    """
    get_config definition
    """
    if version == 'base':
        return BertConfig(
            seq_length=128,
            vocab_size=21128,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            use_relative_positions=True,
            dtype=mstype.float32,
            compute_type=mstype.float32)
    if version == 'large':
        return BertConfig(
            seq_length=128,
            vocab_size=21128,
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            use_relative_positions=True,
            dtype=mstype.float32,
            compute_type=mstype.float32)
    return BertConfig()


class BertLearningRate(lr_schedules.LearningRateSchedule):
    def __init__(self, decay_steps, warmup_steps=100, learning_rate=0.1, end_learning_rate=0.0001, power=1.0):
        super(BertLearningRate, self).__init__()
        self.warmup_lr = lr_schedules.WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = lr_schedules.PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps, power)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

        self.greater = P.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = P.Cast()

    def construct(self, global_step):
        is_warmup = self.cast(self.greater(self.warmup_steps, global_step), mstype.float32)
        warmup_lr = self.warmup_lr(global_step)
        decay_lr = self.decay_lr(global_step)
        lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        return lr


def test_bert_train():
    """
    the main function
    """

    class ModelBert(nn.Cell):
        """
        ModelBert definition
        """

        def __init__(self, network, optimizer=None):
            super(ModelBert, self).__init__()
            self.optimizer = optimizer
            self.train_network = BertTrainOneStepCell(network, self.optimizer)
            self.train_network.set_train()

        def construct(self, arg0, arg1, arg2, arg3, arg4, arg5, arg6):
            return self.train_network(arg0, arg1, arg2, arg3, arg4, arg5, arg6)

    version = os.getenv('VERSION', 'large')
    batch_size = int(os.getenv('BATCH_SIZE', '1'))
    inputs = load_test_data(batch_size)

    config = get_config(version=version)
    netwithloss = BertNetworkWithLoss(config, True)
    lr = BertLearningRate(10)
    optimizer = AdamWeightDecay(netwithloss.trainable_params(), lr)
    net = ModelBert(netwithloss, optimizer=optimizer)
    net.set_train()
    build_construct_graph(net, *inputs, execute=False)


def test_bert_withlossscale_train():
    class ModelBert(nn.Cell):
        def __init__(self, network, optimizer=None):
            super(ModelBert, self).__init__()
            self.optimizer = optimizer
            self.train_network = BertTrainOneStepWithLossScaleCell(network, self.optimizer)
            self.train_network.set_train()

        def construct(self, arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7):
            return self.train_network(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)

    version = os.getenv('VERSION', 'base')
    batch_size = int(os.getenv('BATCH_SIZE', '1'))
    scaling_sens = Tensor(np.ones([1]).astype(np.float32))
    inputs = load_test_data(batch_size) + (scaling_sens,)

    config = get_config(version=version)
    netwithloss = BertNetworkWithLoss(config, True)
    lr = BertLearningRate(10)
    optimizer = AdamWeightDecay(netwithloss.trainable_params(), lr)
    net = ModelBert(netwithloss, optimizer=optimizer)
    net.set_train()
    build_construct_graph(net, *inputs, execute=True)


def bert_withlossscale_manager_train():
    class ModelBert(nn.Cell):
        def __init__(self, network, optimizer=None):
            super(ModelBert, self).__init__()
            self.optimizer = optimizer
            manager = DynamicLossScaleManager()
            update_cell = LossScaleUpdateCell(manager)
            self.train_network = BertTrainOneStepWithLossScaleCell(network, self.optimizer,
                                                                   scale_update_cell=update_cell)
            self.train_network.set_train()

        def construct(self, arg0, arg1, arg2, arg3, arg4, arg5, arg6):
            return self.train_network(arg0, arg1, arg2, arg3, arg4, arg5, arg6)

    version = os.getenv('VERSION', 'base')
    batch_size = int(os.getenv('BATCH_SIZE', '1'))
    inputs = load_test_data(batch_size)

    config = get_config(version=version)
    netwithloss = BertNetworkWithLoss(config, True)
    lr = BertLearningRate(10)
    optimizer = AdamWeightDecay(netwithloss.trainable_params(), lr)
    net = ModelBert(netwithloss, optimizer=optimizer)
    net.set_train()
    build_construct_graph(net, *inputs, execute=True)


def bert_withlossscale_manager_train_feed():
    class ModelBert(nn.Cell):
        def __init__(self, network, optimizer=None):
            super(ModelBert, self).__init__()
            self.optimizer = optimizer
            manager = DynamicLossScaleManager()
            update_cell = LossScaleUpdateCell(manager)
            self.train_network = BertTrainOneStepWithLossScaleCell(network, self.optimizer,
                                                                   scale_update_cell=update_cell)
            self.train_network.set_train()

        def construct(self, arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7):
            return self.train_network(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)

    version = os.getenv('VERSION', 'base')
    batch_size = int(os.getenv('BATCH_SIZE', '1'))
    scaling_sens = Tensor(np.ones([1]).astype(np.float32))
    inputs = load_test_data(batch_size) + (scaling_sens,)

    config = get_config(version=version)
    netwithloss = BertNetworkWithLoss(config, True)
    lr = BertLearningRate(10)
    optimizer = AdamWeightDecay(netwithloss.trainable_params(), lr)
    net = ModelBert(netwithloss, optimizer=optimizer)
    net.set_train()
    build_construct_graph(net, *inputs, execute=True)
