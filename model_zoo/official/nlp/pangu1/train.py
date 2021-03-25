# Copyright 2021 Huawei Technologies Co., Ltd
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
PanGu1 train script
"""

import os
import math
import time
from mindspore import context
from mindspore.train.model import Model
import mindspore.communication.management as D
from mindspore.context import ParallelMode
import mindspore.nn as nn
from mindspore.train.callback import TimeMonitor, Callback
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
import mindspore.common.dtype as mstype
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from src.dataset import create_dataset
from src.pangu1 import PanGu1, PanGu1WithLoss, CrossEntropyLoss
from src.pangu1_wrapcell import PanGu1TrainOneStepWithLossScaleCell, VirtualDatasetOneInputCell
from src.utils import PanGu1Config, LearningRate, get_args


class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss in NAN or INF terminating training.
    Note:
        if per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """
    def __init__(self, dataset_size=-1, local_rank=0, has_trained_epoch=0, has_trained_step=0):
        super(LossCallBack, self).__init__()
        self._dataset_size = dataset_size
        self.local_rank = local_rank
        self.has_trained_epoch = has_trained_epoch
        self.has_trained_step = has_trained_step
        print("load has trained epoch :{} and step: {}".format(has_trained_epoch, has_trained_step), flush=True)
    def step_end(self, run_context):
        """
        Print loss after each step
        """
        cb_params = run_context.original_args()
        if self._dataset_size > 0 and self.local_rank % 8 == 0:
            percent, epoch_num = math.modf(cb_params.cur_step_num /
                                           self._dataset_size)
            if percent == 0:
                percent = 1
                epoch_num -= 1
            date = time.asctime(time.localtime(time.time()))
            print("time: {} local_rank: {}, epoch: {}, step: {}, output is {}, overflow is {}, scale is {}".\
            format(date, int(self.local_rank), int(epoch_num) + int(self.has_trained_epoch),\
            cb_params.cur_step_num + int(self.has_trained_step), cb_params.net_outputs[0].asnumpy(),\
            cb_params.net_outputs[1].asnumpy(), cb_params.net_outputs[2].asnumpy()))


project_root = os.path.abspath(
    os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "..")
print('project_root:', project_root)


def run_train():
    r"""
    The main training process.
    """
    args_opt = get_args()
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend",
                        device_id=device_id)
    context.set_context(variable_memory_max_size="30GB")
    full_batch = bool(args_opt.full_batch)
    if args_opt.distribute == "true":
        D.init()
        device_num = D.get_group_size()
        rank = D.get_rank()
        print("device_id is {}, rank_id is {}, device_num is {}".format(
            device_id, rank, device_num))

        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
            gradients_mean=False,
            device_num=device_num,
            full_batch=full_batch,
            enable_parallel_optimizer=True)
        auto_parallel_context().set_loss_repeated_mean(True)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        _set_multi_subgraphs()

    else:
        rank = 0
        device_num = 1
    model_parallel_num = args_opt.mp
    data_parallel_num = int(device_num / model_parallel_num)
    batch_size = args_opt.per_batch_size * device_num
    config = PanGu1Config(
        data_parallel_num=data_parallel_num,
        model_parallel_num=model_parallel_num,
        batch_size=batch_size,
        seq_length=1024,
        vocab_size=40000,
        embedding_size=5120,
        num_layers=40,
        num_heads=40,
        expand_ratio=4,
        dropout_rate=0.1,
        compute_dtype=mstype.float16,
        eod_reset=bool(args_opt.eod_reset))
    print("===config is: ", config, flush=True)
    pangu1 = PanGu1(config)
    loss = CrossEntropyLoss(config)
    pangu1_with_loss = PanGu1WithLoss(config, pangu1, loss)
    pangu1_with_loss = VirtualDatasetOneInputCell(pangu1_with_loss)

    print("=====args_opt is: ", args_opt, flush=True)
    lr = LearningRate(learning_rate=args_opt.start_lr,
                      end_learning_rate=args_opt.end_lr,
                      warmup_steps=args_opt.warmup_step,
                      decay_steps=200000,
                      lr_scale=1)

    decay_filter = lambda x: 'layernorm' not in x.name.lower() and "bias" not in x.name.lower()
    params = pangu1.trainable_params()
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = [{
        'params': decay_params,
        'weight_decay': 1e-1
    }, {
        'params': other_params,
        'weight_decay': 0.0
    }, {
        'order_params': params
    }]
    if args_opt.optimizer == "lamb":
        optimizer = nn.Lamb(group_params, learning_rate=lr)
    else:
        optimizer = nn.AdamWeightDecay(group_params, learning_rate=lr, eps=1e-8, beta1=0.9, beta2=0.95)
    has_trained_epoch = 0
    has_trained_step = 0
    data_start_index = 0
    loss_scale_value = math.pow(2, 32)

    ds = create_dataset(config.batch_size, data_path=args_opt.data_url,
                        data_start_index=data_start_index, eod_reset=config.eod_reset,
                        eod_id=args_opt.eod_id, device_num=device_num, rank=rank)

    epoch_num = args_opt.epoch_size
    step_per_epoch = ds.get_dataset_size()
    callback_size = args_opt.sink_size
    actual_epoch_num = int(epoch_num * step_per_epoch / callback_size)
    callback = [
        TimeMonitor(callback_size),
        LossCallBack(callback_size, rank, has_trained_epoch, has_trained_step)
    ]
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=loss_scale_value, scale_factor=2, scale_window=1000)
    pangu1_with_grads = PanGu1TrainOneStepWithLossScaleCell(
        pangu1_with_loss, optimizer=optimizer, scale_update_cell=update_cell, enable_global_norm=True, config=config)

    model = Model(pangu1_with_grads)
    print("=====dataset size: ", ds.get_dataset_size(), flush=True)
    print("=====actual_epoch_num: ", actual_epoch_num, flush=True)
    model.train(actual_epoch_num, ds, callbacks=callback, sink_size=callback_size, dataset_sink_mode=True)

if __name__ == "__main__":
    run_train()
