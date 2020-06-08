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

"""train bert network without lossscale"""

import os
import time

import numpy as np
import pytest
from src.bert_for_pre_training import BertNetworkWithLoss, BertTrainOneStepWithLossScaleCell
from src.bert_model import BertConfig

import mindspore.common.dtype as mstype
import mindspore.dataset.engine.datasets as de
import mindspore.dataset.transforms.c_transforms as C
from mindspore import context
from mindspore import log as logger
from mindspore.common.tensor import Tensor
from mindspore.nn.optim import Lamb
from mindspore.train.callback import Callback
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.train.model import Model

_current_dir = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = ["/home/workspace/mindspore_dataset/bert/example/examples.tfrecord"]
SCHEMA_DIR = "/home/workspace/mindspore_dataset/bert/example/datasetSchema.json"


def get_config(version='base', batch_size=1):
    """get config"""
    if version == 'base':
        bert_config = BertConfig(
            batch_size=batch_size,
            seq_length=128,
            vocab_size=21136,
            hidden_size=768,
            num_hidden_layers=2,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            use_relative_positions=True,
            input_mask_from_dataset=True,
            token_type_ids_from_dataset=True,
            dtype=mstype.float32,
            compute_type=mstype.float32)
    elif version == 'large':
        bert_config = BertConfig(
            batch_size=batch_size,
            seq_length=128,
            vocab_size=21136,
            hidden_size=1024,
            num_hidden_layers=2,
            num_attention_heads=16,
            intermediate_size=4096,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            use_relative_positions=False,
            input_mask_from_dataset=True,
            token_type_ids_from_dataset=True,
            dtype=mstype.float32,
            compute_type=mstype.float16,
            enable_fused_layernorm=False)
    else:
        bert_config = BertConfig(batch_size=batch_size)
    return bert_config


def me_de_train_dataset(sink_mode=False):
    """test me de train dataset"""
    # apply repeat operations
    repeat_count = 1
    batch_size = 16
    ds = de.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["input_ids", "input_mask", "segment_ids",
                                                                "next_sentence_labels", "masked_lm_positions",
                                                                "masked_lm_ids", "masked_lm_weights"], shuffle=False)
    type_cast_op = C.TypeCast(mstype.int32)
    new_repeat_count = repeat_count
    if sink_mode:
        repeat_count = 30
        sink_steps = 100
        ori_dataaet_size = ds.get_dataset_size()
        new_size = sink_steps * batch_size
        ds.set_dataset_size(new_size)
        new_repeat_count = int(repeat_count * ori_dataaet_size // ds.get_dataset_size())
    ds = ds.map(input_columns="masked_lm_ids", operations=type_cast_op)
    ds = ds.map(input_columns="masked_lm_positions", operations=type_cast_op)
    ds = ds.map(input_columns="next_sentence_labels", operations=type_cast_op)
    ds = ds.map(input_columns="segment_ids", operations=type_cast_op)
    ds = ds.map(input_columns="input_mask", operations=type_cast_op)
    ds = ds.map(input_columns="input_ids", operations=type_cast_op)
    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.repeat(repeat_count)
    logger.info("data size: {}".format(ds.get_dataset_size()))
    logger.info("repeat_count: {}".format(ds.get_repeat_count()))
    return ds, new_repeat_count


def weight_variable(shape):
    """weight variable"""
    np.random.seed(1)
    ones = np.random.uniform(-0.1, 0.1, size=shape).astype(np.float32)
    return Tensor(ones)


class ModelCallback(Callback):
    def __init__(self):
        super(ModelCallback, self).__init__()
        self.loss_list = []
        self.overflow_list = []
        self.lossscale_list = []

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        self.loss_list.append(cb_params.net_outputs[0].asnumpy()[0])
        self.overflow_list.append(cb_params.net_outputs[1].asnumpy())
        self.lossscale_list.append(cb_params.net_outputs[2].asnumpy())
        print("epoch: {}, outputs are: {}".format(cb_params.cur_epoch_num, str(cb_params.net_outputs)))

class TimeMonitor(Callback):
    """Time Monitor."""
    def __init__(self, data_size):
        super(TimeMonitor, self).__init__()
        self.data_size = data_size
        self.epoch_mseconds_list = []
        self.per_step_mseconds_list = []
    def epoch_begin(self, run_context):
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        self.epoch_mseconds_list.append(epoch_mseconds)
        self.per_step_mseconds_list.append(epoch_mseconds / self.data_size)

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_bert_percision():
    """test bert percision"""
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", reserve_class_name_in_scope=False)
    ds, new_repeat_count = me_de_train_dataset()
    version = os.getenv('VERSION', 'large')
    batch_size = 16
    config = get_config(version=version, batch_size=batch_size)
    netwithloss = BertNetworkWithLoss(config, True)
    optimizer = Lamb(netwithloss.trainable_params(), decay_steps=ds.get_dataset_size()*new_repeat_count,
                     start_learning_rate=5e-5, end_learning_rate=1e-9,
                     power=10.0, warmup_steps=0, weight_decay=0.01)
    scale_window = 3
    scale_manager = DynamicLossScaleManager(2 ** 16, 2, scale_window)
    netwithgrads = BertTrainOneStepWithLossScaleCell(netwithloss, optimizer=optimizer,
                                                     scale_update_cell=scale_manager.get_update_cell())
    netwithgrads.set_train(True)
    model = Model(netwithgrads)
    callback = ModelCallback()
    params = netwithloss.trainable_params()
    for param in params:
        param.init_data()
        value = param.default_input
        name = param.name
        if isinstance(value, Tensor):
            if name.split('.')[-1] in ['weight']:
                if name.split('.')[-3] in ['cls2']:
                    logger.info("***************** BERT param name is 1 {}".format(name))
                    param.default_input = weight_variable(value.asnumpy().shape)
                else:
                    logger.info("***************** BERT param name is 2 {}".format(name))
                    tempshape = value.asnumpy().shape
                    shape = (tempshape[1], tempshape[0])
                    weight_value = weight_variable(shape).asnumpy()
                    param.default_input = Tensor(np.transpose(weight_value, [1, 0]))
            else:
                logger.info("***************** BERT param name is 3 {}".format(name))
                param.default_input = weight_variable(value.asnumpy().shape)
    model.train(new_repeat_count, ds, callbacks=callback, dataset_sink_mode=False)

    # assertion occurs while the loss value, overflow state or loss_scale value is wrong
    loss_value = np.array(callback.loss_list)
    assert np.allclose(loss_value[0], 12.207198, 0, 0.000001)

    expect_loss_value = [12.207198, 11.980881, 11.984844, 11.879381, 11.832978, 12.411333, 12.009284,
                         12.621277, 12.223178, 12.427385]
    print("loss value: {}".format(loss_value))
    assert np.allclose(loss_value, expect_loss_value, 0, 0.0005)

    overflow = np.array(callback.overflow_list)
    expect_overflow = [True, True, False, False, False, True, False, False, False, True]
    print("overflow: {}".format(overflow))
    assert (overflow == expect_overflow).all()

    loss_scale = np.array(callback.lossscale_list)
    expect_loss_scale = [32768.0, 16384.0, 16384.0, 16384.0, 32768.0, 16384.0, 16384.0, 16384.0, 32768.0, 16384.0]
    print("loss scale: {}".format(loss_scale))
    assert np.allclose(loss_scale, expect_loss_scale, 0, 0)

def test_bert_performance():
    """test bert performance"""
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", reserve_class_name_in_scope=False)
    ds, new_repeat_count = me_de_train_dataset(sink_mode=True)
    version = os.getenv('VERSION', 'large')
    batch_size = 16
    config = get_config(version=version, batch_size=batch_size)
    netwithloss = BertNetworkWithLoss(config, True)
    optimizer = Lamb(netwithloss.trainable_params(), decay_steps=ds.get_dataset_size()*new_repeat_count,
                     start_learning_rate=5e-5, end_learning_rate=1e-9,
                     power=10.0, warmup_steps=0, weight_decay=0.01)
    scale_window = 3
    scale_manager = DynamicLossScaleManager(2 ** 16, 2, scale_window)
    netwithgrads = BertTrainOneStepWithLossScaleCell(netwithloss, optimizer=optimizer,
                                                     scale_update_cell=scale_manager.get_update_cell())
    netwithgrads.set_train(True)
    model = Model(netwithgrads)
    callback = ModelCallback()
    params = netwithloss.trainable_params()
    for param in params:
        param.init_data()
        value = param.default_input
        name = param.name
        if isinstance(value, Tensor):
            if name.split('.')[-1] in ['weight']:
                if name.split('.')[-3] in ['cls2']:
                    logger.info("***************** BERT param name is 1 {}".format(name))
                    param.default_input = weight_variable(value.asnumpy().shape)
                else:
                    logger.info("***************** BERT param name is 2 {}".format(name))
                    tempshape = value.asnumpy().shape
                    shape = (tempshape[1], tempshape[0])
                    weight_value = weight_variable(shape).asnumpy()
                    param.default_input = Tensor(np.transpose(weight_value, [1, 0]))
            else:
                logger.info("***************** BERT param name is 3 {}".format(name))
                param.default_input = weight_variable(value.asnumpy().shape)
    time_monitor_callback = TimeMonitor(ds.get_dataset_size())
    model.train(new_repeat_count, ds, callbacks=[time_monitor_callback, callback],
                dataset_sink_mode=True)

    # assertion occurs while the loss value, overflow state or loss_scale value is wrong
    loss_value = np.array(callback.loss_list)
    expect_loss_value = [10.237753, 10.213153, 10.212972]
    print("loss value: {}".format(loss_value))
    assert np.allclose(loss_value, expect_loss_value, 0, 0.0005)

    overflow = np.array(callback.overflow_list)
    expect_overflow = [False, False, False]
    print("overflow: {}".format(overflow))
    assert (overflow == expect_overflow).all()

    loss_scale = np.array(callback.lossscale_list)
    expect_loss_scale = [16384.0, 16384.0, 16384.0]
    print("loss scale: {}".format(loss_scale))
    assert np.allclose(loss_scale, expect_loss_scale, 0, 0)

    epoch_mseconds = np.array(time_monitor_callback.epoch_mseconds_list)[2]
    expect_epoch_mseconds = 1726
    print("epoch mseconds: {}".format(epoch_mseconds))
    assert epoch_mseconds <= expect_epoch_mseconds + 5

    per_step_mseconds = np.array(time_monitor_callback.per_step_mseconds_list)[2]
    expect_per_step_mseconds = 17
    print("per step mseconds: {}".format(per_step_mseconds))
    assert per_step_mseconds <= expect_per_step_mseconds + 1

if __name__ == '__main__':
    test_bert_percision()
    test_bert_performance()
