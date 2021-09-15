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
from mindspore import context
from mindspore.train.model import Model
import mindspore.communication.management as D
from mindspore.context import ParallelMode
import mindspore.nn as nn
from mindspore.train.callback import TimeMonitor
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
import mindspore.common.dtype as mstype
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore.nn.wrap.cell_wrapper import PipelineCell, _VirtualDatasetCell
from mindspore.parallel.nn import TransformerOpParallelConfig, CrossEntropyLoss
from src.adam import AdamWeightDecayOp
from src.dataset import create_dataset
from src.pangu_alpha import PanGUAlphaWithLoss, PanguAlphaModel
from src.pangu_alpha_wrapcell import PanguAlphaTrainOneStepWithLossScaleCell, PanguAlphaTrainPipelineWithLossScaleCell
from src.pangu_alpha_config import set_parse, PanguAlphaConfig
from src.utils import LearningRate, get_args, FP32StateAdamWeightDecay
from src.utils import download_data
from src.callbacks import EvalCallBack, LossCallBack
from src.metrics import PPLMetric

project_root = os.path.abspath(
    os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "..")
print('project_root:', project_root)


def set_weight_decay(params):
    """
    Set weight decay coefficient, zero for bias and layernorm, 1e-1 for rest
    """
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
    return group_params


def run_train(args_opt):
    r"""
    The main training process.
    """
    # Set execution mode
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    context.set_context(variable_memory_max_size="31GB")
    # Set parallel context
    if args_opt.distribute == "true":
        D.init()
        device_num = D.get_group_size()
        rank = D.get_rank()
        print("rank_id is {}, device_num is {}".format(rank, device_num))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, gradients_mean=False,
            full_batch=bool(args_opt.full_batch), strategy_ckpt_load_file=args_opt.strategy_load_ckpt_path,
            enable_parallel_optimizer=bool(args_opt.optimizer_shard))
        set_algo_parameters(elementwise_op_strategy_follow=True)
        _set_multi_subgraphs()
    else:
        rank = 0
        device_num = 1
    context.set_context(save_graphs=False, save_graphs_path="./graphs_of_device_id_" + str(rank))
    # copy data from the cloud to the /cache/Data
    cache_url = '/cache/Data/'
    eval_cache_url = '/cache/EvalData/'
    if args_opt.offline:
        cache_url = args_opt.data_url
        eval_cache_url = args_opt.eval_data_url
    else:
        download_data(src_data_url=args_opt.data_url, tgt_data_path=cache_url, rank=rank)
        download_data(src_data_url=args_opt.eval_data_url, tgt_data_path=eval_cache_url, rank=rank)
    # Set model property
    model_parallel_num = args_opt.op_level_model_parallel_num
    data_parallel_num = int(device_num / model_parallel_num)
    batch_size = args_opt.per_batch_size * data_parallel_num
    parallel_config = TransformerOpParallelConfig(data_parallel=data_parallel_num,
                                                  model_parallel=model_parallel_num,
                                                  pipeline_stage=args_opt.stage_num,
                                                  micro_batch_num=args_opt.micro_size,
                                                  optimizer_shard=bool(args_opt.optimizer_shard),
                                                  vocab_emb_dp=bool(args_opt.word_emb_dp),
                                                  recompute=True)
    config = PanguAlphaConfig(batch_size=batch_size, num_heads=args_opt.num_heads,
                              hidden_size=args_opt.embedding_size, seq_length=args_opt.seq_length,
                              vocab_size=args_opt.vocab_size, num_layers=args_opt.num_layers,
                              ffn_hidden_size=args_opt.embedding_size * 4,
                              eod_token=bool(args_opt.eod_reset),
                              load_ckpt_path=args_opt.load_ckpt_path,
                              param_init_type=mstype.float32 if args_opt.param_init_type == 'fp32' else mstype.float16,
                              enable_offload=bool(args_opt.opt_offload),
                              parallel_config=parallel_config)
    print("===config is: ", config, flush=True)

    # Define network
    pangu_alpha = PanguAlphaModel(config=config)
    loss = CrossEntropyLoss(config.parallel_config.dp_mp_config)
    pangu_alpha_with_loss_net = PanGUAlphaWithLoss(config, pangu_alpha, loss)
    pangu_alpha_with_loss = _VirtualDatasetCell(pangu_alpha_with_loss_net)

    print("=====args_opt is: ", args_opt, flush=True)
    # Warm-up and cosine decay learning rate
    lr = LearningRate(learning_rate=args_opt.start_lr,
                      end_learning_rate=args_opt.end_lr,
                      warmup_steps=args_opt.warmup_step,
                      decay_steps=200000)

    params = pangu_alpha_with_loss.trainable_params()
    group_params = set_weight_decay(params)
    if args_opt.optimizer == "lamb":
        optimizer = nn.Lamb(group_params, learning_rate=lr)
    elif args_opt.opt_offload:
        optimizer = AdamWeightDecayOp(group_params, learning_rate=lr, eps=1e-8, beta1=0.9, beta2=0.95,
                                      param_init_type=config.param_init_type)
    else:
        optimizer = FP32StateAdamWeightDecay(group_params, learning_rate=lr, eps=1e-8, beta1=0.9, beta2=0.95)
    # Initial scaling sens
    loss_scale_value = math.pow(2, 32)
    epoch_num = args_opt.epoch_size
    # Dataset loading mindrecord files
    ds = create_dataset(config.batch_size, data_path=cache_url, data_start_index=0, eod_reset=config.eod_reset,
                        full_batch=bool(args_opt.full_batch), eod_id=args_opt.eod_id, device_num=device_num,
                        rank=rank, column_name=args_opt.data_column_name, epoch=epoch_num)
    actual_epoch_num = int(epoch_num * ds.get_dataset_size() / args_opt.sink_size)
    callback = [TimeMonitor(args_opt.sink_size), LossCallBack(args_opt.sink_size, rank, 0, 0)]
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=loss_scale_value, scale_factor=2, scale_window=1000)
    pangu_alpha_with_grads = PanguAlphaTrainOneStepWithLossScaleCell(
        pangu_alpha_with_loss, optimizer=optimizer, scale_update_cell=update_cell, enable_global_norm=True,
        config=config)
    if args_opt.train_and_eval_mode:
        ds_eval = create_dataset(config.batch_size, data_path=eval_cache_url,
                                 data_start_index=0, eod_reset=config.eod_reset, full_batch=bool(args_opt.full_batch),
                                 eod_id=args_opt.eod_id, device_num=device_num, rank=rank,
                                 column_name=args_opt.data_column_name, epoch=epoch_num,
                                 num_samples=args_opt.eval_steps * config.batch_size)
        ppl_metric = PPLMetric(config.seq_length)
        model = Model(pangu_alpha_with_grads, eval_network=pangu_alpha_with_loss, metrics={"ppl": ppl_metric})
        callback.append(EvalCallBack(model, ds_eval, ppl_metric))
    else:
        model = Model(pangu_alpha_with_grads)
    if args_opt.incremental_training:
        from mindspore.train.serialization import load_distributed_checkpoint
        strategy = model.infer_train_layout(train_dataset=ds, sink_size=args_opt.sink_size)
        print("======start load_distributed checkpoint", flush=True)
        # For 2.6B and 13B models, the number of ckpt files is 512.
        ckpt_file_list = [os.path.join(args_opt.load_ckpt_path, f"filerted_{ckpt_rank}.ckpt") for ckpt_rank in
                          range(0, 512)]
        print(f"Loading from path {ckpt_file_list[0]}", flush=True)
        load_distributed_checkpoint(model.train_network, ckpt_file_list, strategy)
    print("Dataset size: {}, actual_epoch_num: {}".format(ds.get_dataset_size(), actual_epoch_num), flush=True)
    model.train(actual_epoch_num, ds, callbacks=callback, sink_size=args_opt.sink_size, dataset_sink_mode=True)


def run_train_pipeline(args_opt):
    r"""
    The main training process in pipeline.
    """
    context.set_context(save_graphs=False, mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    context.set_context(variable_memory_max_size="31GB")
    if args_opt.distribute == "true":
        D.init()
        device_num = D.get_group_size()
        rank_id = D.get_rank()
        print("rank_id is {}, device_num is {}".format(rank_id, device_num))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, gradients_mean=False,
            full_batch=bool(args_opt.full_batch), loss_repeated_mean=True,
            device_num=device_num, enable_parallel_optimizer=bool(args_opt.optimizer_shard),
            pipeline_stages=args_opt.stage_num)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        _set_multi_subgraphs()
    else:
        rank_id = int(os.getenv("RANK_ID"))
        device_num = 1
    # copy data from the cloud to the /cache/Data
    cache_url = '/cache/Data/'
    eval_cache_url = '/cache/EvalData/'
    if args_opt.offline:
        cache_url = args_opt.data_url
        eval_cache_url = args_opt.eval_data_url
    else:
        download_data(src_data_url=args_opt.data_url, tgt_data_path=cache_url, rank=rank_id)
        download_data(src_data_url=args_opt.eval_data_url, tgt_data_path=eval_cache_url, rank=rank_id)
    model_parallel_num = args_opt.op_level_model_parallel_num
    stage_device_num = int(device_num / args_opt.stage_num)
    data_parallel_num = int(stage_device_num / model_parallel_num)
    if data_parallel_num <= 1 and args_opt.optimizer_shard == 1:
        raise ValueError("The dp must large than 1 when applying optimizer shard.")
    per_batch_size = args_opt.per_batch_size
    batch_size = per_batch_size * data_parallel_num * args_opt.micro_size

    parallel_config = TransformerOpParallelConfig(data_parallel=data_parallel_num,
                                                  model_parallel=model_parallel_num,
                                                  pipeline_stage=args_opt.stage_num,
                                                  micro_batch_num=args_opt.micro_size,
                                                  optimizer_shard=bool(args_opt.optimizer_shard),
                                                  vocab_emb_dp=bool(args_opt.word_emb_dp),
                                                  recompute=True)
    config = PanguAlphaConfig(batch_size=batch_size // parallel_config.micro_batch_num,
                              num_heads=args_opt.num_heads, hidden_size=args_opt.embedding_size,
                              seq_length=args_opt.seq_length, vocab_size=args_opt.vocab_size,
                              num_layers=args_opt.num_layers, ffn_hidden_size=args_opt.embedding_size * 4,
                              eod_token=bool(args_opt.eod_reset), load_ckpt_path=args_opt.load_ckpt_path,
                              param_init_type=mstype.float32 if args_opt.param_init_type == 'fp32' else mstype.float16,
                              enable_offload=bool(args_opt.opt_offload), parallel_config=parallel_config)

    print("===config is: ", config, flush=True)
    pangu_alpha = PanguAlphaModel(config=config)
    loss = CrossEntropyLoss(config.parallel_config.dp_mp_config)
    pangu_alpha_with_loss_net = PipelineCell(PanGUAlphaWithLoss(config, pangu_alpha, loss),
                                             config.parallel_config.micro_batch_num)
    pangu_alpha_with_loss = _VirtualDatasetCell(pangu_alpha_with_loss_net)
    print("=====args_opt is: ", args_opt, flush=True)
    lr = LearningRate(learning_rate=args_opt.start_lr, end_learning_rate=args_opt.end_lr,
                      warmup_steps=args_opt.warmup_step, decay_steps=args_opt.decay_steps)
    params = pangu_alpha.infer_param_pipeline_stage()
    group_params = set_weight_decay(params)
    if args_opt.optimizer == "lamb":
        optimizer = nn.Lamb(group_params, learning_rate=lr)
    elif args_opt.opt_offload:
        optimizer = AdamWeightDecayOp(group_params, learning_rate=lr, eps=1e-8, beta1=0.9, beta2=0.95,
                                      param_init_type=config.param_init_type)
    else:
        optimizer = nn.AdamWeightDecay(group_params, learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)

    ds = create_dataset(config.batch_size * parallel_config.micro_batch_num, data_path=cache_url,
                        device_num=stage_device_num,
                        rank=rank_id % stage_device_num, eod_reset=True, data_start_index=0,
                        full_batch=context.get_auto_parallel_context("full_batch"),
                        column_name=args_opt.data_column_name)
    epoch_num = args_opt.epoch_size
    step_per_epoch = ds.get_dataset_size()
    callback_size = args_opt.sink_size
    actual_epoch_num = int(epoch_num * step_per_epoch / callback_size)
    callback = [TimeMonitor(callback_size), LossCallBack(callback_size, rank_id,
                                                         micro_size=parallel_config.micro_batch_num)]
    loss_scale_value = math.pow(2, 32)
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=loss_scale_value, scale_factor=2, scale_window=1000)
    pangu_alpha_with_grads = PanguAlphaTrainPipelineWithLossScaleCell(
        pangu_alpha_with_loss, optimizer=optimizer, config=config, scale_update_cell=update_cell)
    if args_opt.train_and_eval_mode:
        ds_eval = create_dataset(config.batch_size * parallel_config.micro_batch_num, data_path=eval_cache_url,
                                 device_num=stage_device_num, rank=rank_id % stage_device_num, eod_reset=True,
                                 data_start_index=0, full_batch=bool(args_opt.full_batch),
                                 column_name=args_opt.data_column_name,
                                 num_samples=args_opt.eval_steps * config.batch_size)
        ppl_metric = PPLMetric(config.seq_length)
        pangu_alpha_with_loss_eval_net = _VirtualDatasetCell(PanGUAlphaWithLoss(config, pangu_alpha, loss))
        model = Model(pangu_alpha_with_grads, eval_network=pangu_alpha_with_loss_eval_net, metrics={"ppl": ppl_metric})
        model.build(ds, ds_eval, sink_size=callback_size)
        eval_callback = EvalCallBack(model, ds_eval, ppl_metric)
        callback.append(eval_callback)
    else:
        model = Model(pangu_alpha_with_grads)
    model.train(actual_epoch_num, ds, callbacks=callback,
                sink_size=callback_size, dataset_sink_mode=True)


if __name__ == "__main__":
    opt = get_args()
    set_parse(opt)
    if opt.per_batch_size == 0:
        raise ValueError("The per_batch_size has not been configured.")
    if opt.stage_num > 1:
        run_train_pipeline(opt)
    else:
        run_train(opt)
