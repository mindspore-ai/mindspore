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
PanguAlpha train script
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
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore.nn.wrap.cell_wrapper import PipelineCell, _VirtualDatasetCell
from src.dataset import create_dataset
from src.pangu_alpha import PanguAlpha, PanguAlphaWithLoss,\
    PanguAlphaPipeline, PanguAlphaWithLossPipeline, CrossEntropyLoss
from src.pangu_alpha_wrapcell import PanguAlphaTrainOneStepWithLossScaleCell, PanguAlphaTrainPipelineWithLossScaleCell
from src.pangu_alpha_config import PANGUALPHAConfig, set_parse
from src.utils import LearningRate, get_args, FP32StateAdamWeightDecay
from src.utils import download_data


class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss in NAN or INF terminating training.
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
                epoch_num -= 1
            date = time.asctime(time.localtime(time.time()))
            print("time: {} local_rank: {}, epoch: {}, step: {}, output is {}, overflow is {}, scale is {}".
                  format(date, int(self.local_rank), int(epoch_num) + int(self.has_trained_epoch),
                         cb_params.cur_step_num + int(self.has_trained_step), cb_params.net_outputs[0].asnumpy(),
                         cb_params.net_outputs[1].asnumpy(), cb_params.net_outputs[2].asnumpy()))


project_root = os.path.abspath(
    os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "..")
print('project_root:', project_root)


def run_train(args_opt):
    r"""
    The main training process.
    """
    device_id = int(os.getenv('DEVICE_ID'))
    # Set execution mode
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend",
                        device_id=device_id)
    context.set_context(variable_memory_max_size="30GB")
    # Set parallel context
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
            full_batch=bool(args_opt.full_batch),
            enable_parallel_optimizer=bool(args_opt.optimizer_shard))
        set_algo_parameters(elementwise_op_strategy_follow=True)
        _set_multi_subgraphs()

    else:
        rank = 0
        device_num = 1

    # copy data from the cloud to the /cache/Data
    cache_url = '/cache/Data/'
    if args_opt.offline:
        cache_url = args_opt.data_url
    else:
        download_data(src_data_url=args_opt.data_url, tgt_data_path=cache_url, rank=rank)
    # Set model property
    model_parallel_num = args_opt.op_level_model_parallel_num
    data_parallel_num = int(device_num / model_parallel_num)
    batch_size = args_opt.per_batch_size * data_parallel_num
    config = PANGUALPHAConfig(
        data_parallel_num=data_parallel_num,
        model_parallel_num=model_parallel_num,
        batch_size=batch_size,
        seq_length=args_opt.seq_length,
        vocab_size=args_opt.vocab_size,
        embedding_size=args_opt.embedding_size,
        num_layers=args_opt.num_layers,
        num_heads=args_opt.num_heads,
        expand_ratio=4,
        dropout_rate=0.1,
        compute_dtype=mstype.float16,
        stage_num=args_opt.stage_num,
        micro_size=args_opt.micro_size,
        eod_reset=bool(args_opt.eod_reset),
        param_init_type=mstype.float32 if args_opt.param_init_type == 'fp32' else mstype.float16,
        word_emb_dp=bool(args_opt.word_emb_dp))
    print("===config is: ", config, flush=True)

    # Define network
    pangu_alpha = PanguAlpha(config)
    loss = CrossEntropyLoss(config)
    pangu_alpha_with_loss = PanguAlphaWithLoss(config, pangu_alpha, loss)
    pangu_alpha_with_loss = _VirtualDatasetCell(pangu_alpha_with_loss)

    print("=====args_opt is: ", args_opt, flush=True)

    # Warm-up and cosine decay learning rate
    lr = LearningRate(learning_rate=args_opt.start_lr,
                      end_learning_rate=args_opt.end_lr,
                      warmup_steps=args_opt.warmup_step,
                      decay_steps=200000)

    # Set weight decay coefficient, zero for bias and layernorm, 1e-1 for rest
    decay_filter = lambda x: 'layernorm' not in x.name.lower() and "bias" not in x.name.lower()
    params = pangu_alpha.trainable_params()
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
        optimizer = FP32StateAdamWeightDecay(group_params, learning_rate=lr, eps=1e-8, beta1=0.9, beta2=0.95)
    # Initial scaling sens
    loss_scale_value = math.pow(2, 32)
    epoch_num = args_opt.epoch_size
    # Dataset loading mindrecord files
    ds = create_dataset(config.batch_size, data_path=cache_url,
                        data_start_index=0, eod_reset=config.eod_reset, full_batch=bool(args_opt.full_batch),
                        eod_id=args_opt.eod_id, device_num=device_num, rank=rank,
                        column_name=args_opt.data_column_name, epoch=epoch_num)
    step_per_epoch = ds.get_dataset_size()
    callback_size = args_opt.sink_size
    actual_epoch_num = int(epoch_num * step_per_epoch / callback_size)
    callback = [
        TimeMonitor(callback_size),
        LossCallBack(callback_size, rank, 0, 0)
    ]
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=loss_scale_value, scale_factor=2, scale_window=1000)
    pangu_alpha_with_grads = PanguAlphaTrainOneStepWithLossScaleCell(
        pangu_alpha_with_loss, optimizer=optimizer, scale_update_cell=update_cell, enable_global_norm=True,
        config=config)
    model = Model(pangu_alpha_with_grads)
    print("Dataset size: {}, actual_epoch_num: {}".format(ds.get_dataset_size(), actual_epoch_num), flush=True)
    model.train(actual_epoch_num, ds, callbacks=callback, sink_size=callback_size, dataset_sink_mode=True)

def run_train_pipeline(args_opt):
    r"""
    The main training process in pipeline.
    """
    device_id = int(os.getenv("DEVICE_ID"))
    context.set_context(save_graphs=False,
                        mode=context.GRAPH_MODE,
                        device_target="Ascend",
                        device_id=device_id)
    context.set_context(variable_memory_max_size="31GB")
    if args_opt.distribute == "true":
        D.init()
        device_num = D.get_group_size()
        rank_id = D.get_rank()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
            gradients_mean=False,
            device_num=device_num,
            full_batch=True,
            loss_repeated_mean=True,
            enable_parallel_optimizer=bool(args_opt.optimizer_shard),
            pipeline_stages=args_opt.stage_num)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        _set_multi_subgraphs()
    else:
        rank_id = int(os.getenv("RANK_ID"))
        device_num = 1
    model_parallel_num = args_opt.op_level_model_parallel_num
    stage_device_num = int(device_num / args_opt.stage_num)
    data_parallel_num = int(stage_device_num / model_parallel_num)
    per_batch_size = args_opt.per_batch_size
    batch_size = per_batch_size * data_parallel_num * args_opt.micro_size
    config = PANGUALPHAConfig(
        data_parallel_num=data_parallel_num,
        model_parallel_num=model_parallel_num,
        batch_size=batch_size,
        seq_length=args_opt.seq_length,
        vocab_size=args_opt.vocab_size,
        embedding_size=args_opt.embedding_size,
        num_layers=args_opt.num_layers,
        num_heads=args_opt.num_heads,
        expand_ratio=4,
        post_layernorm_residual=False,
        dropout_rate=0.1,
        compute_dtype=mstype.float16,
        use_past=False,
        self_layernorm=True,
        stage_num=args_opt.stage_num,
        micro_size=args_opt.micro_size,
        word_emb_dp=bool(args_opt.word_emb_dp))
    print("===config is: ", config, flush=True)
    pangu_alpha = PanguAlphaPipeline(config)
    loss = CrossEntropyLoss(config)
    pangu_alpha_with_loss = PipelineCell(PanguAlphaWithLossPipeline(config, pangu_alpha, loss), config.micro_size)
    pangu_alpha_with_loss = _VirtualDatasetCell(pangu_alpha_with_loss)
    print("=====args_opt is: ", args_opt, flush=True)
    lr = LearningRate(learning_rate=args_opt.start_lr,
                      end_learning_rate=args_opt.end_lr,
                      warmup_steps=args_opt.warmup_step,
                      decay_steps=args_opt.decay_steps)
    params = pangu_alpha.infer_param_pipeline_stage()
    decay_filter = lambda x: 'layernorm' not in x.name.lower() and "bias" not in x.name.lower()
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
        optimizer = nn.AdamWeightDecay(group_params, learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)
    ds = create_dataset(config.batch_size, data_path=args_opt.data_url, eod_reset=True,
                        data_start_index=0, full_batch=True, column_name=args_opt.data_column_name)
    epoch_num = args_opt.epoch_size
    step_per_epoch = ds.get_dataset_size()
    callback_size = args_opt.sink_size
    actual_epoch_num = int(epoch_num * step_per_epoch / callback_size)
    callback = [
        TimeMonitor(callback_size),
        LossCallBack(callback_size, rank_id, config.stage_num)
    ]
    loss_scale_value = math.pow(2, 32)
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=loss_scale_value,
                                             scale_factor=2,
                                             scale_window=1000)
    pangu_alpha_with_grads = PanguAlphaTrainPipelineWithLossScaleCell(
        pangu_alpha_with_loss, optimizer=optimizer, config=config, scale_update_cell=update_cell)
    model = Model(pangu_alpha_with_grads)
    model.train(actual_epoch_num,
                ds,
                callbacks=callback,
                sink_size=callback_size,
                dataset_sink_mode=True)

if __name__ == "__main__":
    opt = get_args()
    set_parse(opt)
    if opt.stage_num > 1:
        run_train_pipeline(opt)
    else:
        run_train(opt)
