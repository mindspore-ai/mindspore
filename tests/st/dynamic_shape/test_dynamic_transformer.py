# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""Transformer training script."""
import time
import numpy as np
from easydict import EasyDict as edict
import pytest

import mindspore as ms
from mindspore.common.tensor import Tensor
from mindspore.nn.optim import Adam
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.train import Callback, TimeMonitor, Model
from mindspore.common import set_seed
import mindspore.dataset as de
from transformer.transformer_for_train import TransformerNetworkWithLoss, TransformerTrainOneStepWithLossScaleCell


set_seed(1)


def get_ms_timestamp():
    t = time.time()
    return int(round(t * 1000))

TIME_STAMP_INIT = False
TIME_STAMP_FIRST = 0
EPOCH_SIZE = 1
BATCH_SIZE = 32
LR_BETA2 = 0.997

lr_schedule = edict({'learning_rate': 2.0, 'warmup_steps': 8000, 'start_decay_step': 16000, 'min_lr': 0.0,})


class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss is NAN or INF terminating training.
    Note:
        If per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, per_print_times=1, rank_id=0):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.rank_id = rank_id
        self.loss_list = []
        global TIME_STAMP_INIT, TIME_STAMP_FIRST
        if not TIME_STAMP_INIT:
            TIME_STAMP_FIRST = get_ms_timestamp()
            TIME_STAMP_INIT = True

    def step_end(self, run_context):
        """Monitor the loss in training."""
        global TIME_STAMP_FIRST
        time_stamp_current = get_ms_timestamp()
        cb_params = run_context.original_args()
        print("time: {}, epoch: {}, step: {}, outputs are {}".format(time_stamp_current - TIME_STAMP_FIRST,
                                                                     cb_params.cur_epoch_num,
                                                                     cb_params.cur_step_num,
                                                                     str(cb_params.net_outputs)))
        result = cb_params.net_outputs[0].asnumpy()
        self.loss_list.append(result)


def linear_warmup(warmup_steps, current_step):
    return min([1.0, float(current_step)/float(warmup_steps)])


def rsqrt_decay(warmup_steps, current_step):
    return float(max([current_step, warmup_steps])) ** -0.5


def rsqrt_hidden(hidden_size):
    return float(hidden_size) ** -0.5


def create_dynamic_lr(schedule, training_steps, learning_rate, warmup_steps, hidden_size,
                      start_decay_step=0, min_lr=0.):
    """
    Generate dynamic learning rate.
    """
    if start_decay_step < warmup_steps:
        start_decay_step = warmup_steps
    lr = []
    for current_step in range(1, training_steps+1):
        cur_lr = 1.0
        for name in schedule.split("*"):
            if name == "constant":
                cur_lr *= float(learning_rate)
            elif name == "rsqrt_hidden":
                cur_lr *= rsqrt_hidden(hidden_size)
            elif name == "linear_warmup":
                cur_lr *= linear_warmup(warmup_steps, current_step)
            elif name == "rsqrt_decay":
                cur_lr *= rsqrt_decay(warmup_steps, current_step-start_decay_step+warmup_steps)
            else:
                raise ValueError("unknown learning rate schedule")
        if warmup_steps < current_step < start_decay_step:
            cur_lr = lr[-1]
        if current_step > warmup_steps:
            cur_lr = max([cur_lr, min_lr])
        lr.append(cur_lr)
    return lr


def fun(data, shape):
    data = data.reshape(shape)
    return data[0], data[1], data[2], data[3], data[4], data[5]


def create_transformer_dynamic_dataset(rank_size=1, rank_id=0, do_shuffle="true"):
    dataset = de.MindDataset(
        "/home/workspace/mindspore_dataset/transformer/test-dynamic-mindrecord",
        columns_list=["batch_data", "batch_shape"],
        shuffle=(do_shuffle == "true"), num_shards=rank_size, shard_id=rank_id)

    dataset = dataset.map(fun, input_columns=["batch_data", "batch_shape"],
                          output_columns=["source_eos_ids", "source_eos_mask",
                                          "target_sos_ids", "target_sos_mask",
                                          "target_eos_ids", "target_eos_mask"],
                          )
    dataset = dataset.project(["source_eos_ids", "source_eos_mask",
                               "target_sos_ids", "target_sos_mask",
                               "target_eos_ids", "target_eos_mask"])
    return dataset


def get_train_loss(is_graph_mode, device_target, device_id=0):
    """
    Transformer training.
    """
    if is_graph_mode:
        mode = ms.GRAPH_MODE
    else:
        mode = ms.PYNATIVE_MODE

    ms.set_context(mode=mode, device_target=device_target, reserve_class_name_in_scope=False,
                   enable_graph_kernel=False)

    # Set mempool block size in PYNATIVE_MODE for improving memory utilization, which will not take effect in GRAPH_MODE
    if ms.get_context("mode") == ms.PYNATIVE_MODE:
        ms.set_context(mempool_block_size="31GB")

    device_num = 1
    rank_id = 0

    dataset = create_transformer_dynamic_dataset(rank_size=device_num, rank_id=rank_id, do_shuffle=True)
    netwithloss = TransformerNetworkWithLoss(True, is_graph_mode=is_graph_mode)

    hidden_size = 1024
    learning_rate = 1.0
    lr = Tensor(create_dynamic_lr(schedule="constant*rsqrt_hidden*linear_warmup*rsqrt_decay",
                                  training_steps=dataset.get_dataset_size() * EPOCH_SIZE,
                                  learning_rate=learning_rate,
                                  warmup_steps=lr_schedule.warmup_steps,
                                  hidden_size=hidden_size,
                                  start_decay_step=lr_schedule.start_decay_step,
                                  min_lr=lr_schedule.min_lr), ms.float32)


    optimizer = Adam(netwithloss.trainable_params(), lr, beta2=LR_BETA2)
    loss_callback = LossCallBack(rank_id=rank_id)
    callbacks = [TimeMonitor(dataset.get_dataset_size()), loss_callback]

    scale_manager = DynamicLossScaleManager(init_loss_scale=1024, scale_factor=2, scale_window=2000)
    update_cell = scale_manager.get_update_cell()
    netwithgrads = TransformerTrainOneStepWithLossScaleCell(netwithloss, optimizer=optimizer,
                                                            scale_update_cell=update_cell)
    if is_graph_mode:
        data_col_int64 = Tensor(shape=[BATCH_SIZE, None], dtype=ms.int64)
        data_col = Tensor(shape=[BATCH_SIZE, None], dtype=ms.float32)
        netwithgrads.set_inputs(data_col_int64, data_col_int64, data_col_int64,
                                data_col_int64, data_col_int64, data_col_int64,
                                data_col)

    netwithgrads.set_train(True)
    model = Model(netwithgrads)
    model.train(EPOCH_SIZE, dataset, callbacks=callbacks, dataset_sink_mode=True)
    loss_list = loss_callback.loss_list
    return loss_list


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_train_graph_mode_gpu():
    """
    Feature: Test the simplified dynamic shape transformer network with small data.
    Description:  The sequence length of inputs is dynamic.
    Expectation: Assert that the training loss of fixed data is consistent with the expected loss.
    """
    graph_loss = get_train_loss(True, "GPU")
    expect_loss = [11.193909]
    assert np.allclose(graph_loss, expect_loss, 5e-3, 5e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_train_pynative_mode_gpu():
    """
    Feature: Test the simplified dynamic shape transformer network with small data.
    Description:  The sequence length of inputs is dynamic.
    Expectation: Assert that the training loss of fixed data is consistent with the expected loss.
    """
    graph_loss = get_train_loss(False, "GPU")
    expect_loss = [11.112342]
    assert np.allclose(graph_loss[0], expect_loss, 5e-3, 5e-3)
