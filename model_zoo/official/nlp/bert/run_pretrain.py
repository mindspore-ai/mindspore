# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#################pre_train bert example on zh-wiki########################
python run_pretrain.py
"""
import os
import mindspore.communication.management as D
from mindspore.communication.management import get_rank
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.train_thor import ConvertModelUtils
from mindspore.nn.optim import Lamb, Momentum, AdamWeightDecay, thor
from mindspore import log as logger
from mindspore.common import set_seed
from src import BertNetworkWithLoss, BertNetworkMatchBucket, \
    BertTrainOneStepCell, \
    BertTrainOneStepWithLossScaleCell, \
    BertTrainAccumulationAllReduceEachWithLossScaleCell, \
    BertTrainAccumulationAllReducePostWithLossScaleCell, \
    BertTrainOneStepWithLossScaleCellForAdam, \
    BertPretrainEval, \
    AdamWeightDecayForBert, AdamWeightDecayOp
from src.dataset import create_bert_dataset, create_eval_dataset
from src.utils import LossCallBack, BertLearningRate, EvalCallBack, BertMetric
from src.model_utils.config import config as cfg, bert_net_cfg
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_device_num
_current_dir = os.path.dirname(os.path.realpath(__file__))


def _set_bert_all_reduce_split():
    """set bert all_reduce fusion split, support num_hidden_layers is 12 and 24."""
    device_target = context.get_context('device_target')
    enable_graph_kernel = context.get_context('enable_graph_kernel')
    device_num = context.get_auto_parallel_context('device_num')
    if bert_net_cfg.num_hidden_layers == 12:
        if bert_net_cfg.use_relative_positions:
            context.set_auto_parallel_context(all_reduce_fusion_config=[29, 58, 87, 116, 145, 174, 203, 217])
        else:
            context.set_auto_parallel_context(all_reduce_fusion_config=[28, 55, 82, 109, 136, 163, 190, 205])
            if device_target == 'GPU' and enable_graph_kernel and device_num == 8:
                context.set_auto_parallel_context(all_reduce_fusion_config=[180, 205])
            elif device_target == 'GPU' and enable_graph_kernel and device_num == 16:
                context.set_auto_parallel_context(all_reduce_fusion_config=[120, 205])
    elif bert_net_cfg.num_hidden_layers == 24:
        if bert_net_cfg.use_relative_positions:
            context.set_auto_parallel_context(all_reduce_fusion_config=[30, 90, 150, 210, 270, 330, 390, 421])
        else:
            context.set_auto_parallel_context(all_reduce_fusion_config=[38, 93, 148, 203, 258, 313, 368, 397])
            if device_target == 'Ascend' and enable_graph_kernel and device_num == 8:
                context.set_auto_parallel_context(all_reduce_fusion_config=[
                    0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 50, 70, 93, 148, 203, 258, 313, 368, 397])


def _get_optimizer(args_opt, network):
    """get bert optimizer, support Lamb, Momentum, AdamWeightDecay."""
    if cfg.optimizer == 'Lamb':
        lr_schedule = BertLearningRate(learning_rate=cfg.Lamb.learning_rate,
                                       end_learning_rate=cfg.Lamb.end_learning_rate,
                                       warmup_steps=cfg.Lamb.warmup_steps,
                                       decay_steps=args_opt.train_steps,
                                       power=cfg.Lamb.power)
        params = network.trainable_params()
        decay_params = list(filter(cfg.Lamb.decay_filter, params))
        other_params = list(filter(lambda x: not cfg.Lamb.decay_filter(x), params))
        group_params = [{'params': decay_params, 'weight_decay': cfg.Lamb.weight_decay},
                        {'params': other_params},
                        {'order_params': params}]
        optimizer = Lamb(group_params, learning_rate=lr_schedule, beta1=cfg.Lamb.beta1, beta2=cfg.Lamb.beta2,
                         eps=cfg.Lamb.eps)
    elif cfg.optimizer == 'Momentum':
        optimizer = Momentum(network.trainable_params(), learning_rate=cfg.Momentum.learning_rate,
                             momentum=cfg.Momentum.momentum)
    elif cfg.optimizer == 'AdamWeightDecay':
        lr_schedule = BertLearningRate(learning_rate=cfg.AdamWeightDecay.learning_rate,
                                       end_learning_rate=cfg.AdamWeightDecay.end_learning_rate,
                                       warmup_steps=cfg.AdamWeightDecay.warmup_steps,
                                       decay_steps=args_opt.train_steps,
                                       power=cfg.AdamWeightDecay.power)
        params = network.trainable_params()
        decay_params = list(filter(cfg.AdamWeightDecay.decay_filter, params))
        other_params = list(filter(lambda x: not cfg.AdamWeightDecay.decay_filter(x), params))
        group_params = [{'params': decay_params, 'weight_decay': cfg.AdamWeightDecay.weight_decay},
                        {'params': other_params, 'weight_decay': 0.0},
                        {'order_params': params}]
        if args_opt.enable_lossscale == "true" and args_opt.device_target == 'GPU':
            optimizer = AdamWeightDecayForBert(group_params, learning_rate=lr_schedule, eps=cfg.AdamWeightDecay.eps)
        elif context.get_context("mode") == context.PYNATIVE_MODE and args_opt.device_target == 'GPU':
            optimizer = AdamWeightDecayOp(group_params, learning_rate=lr_schedule, eps=cfg.AdamWeightDecay.eps)
        else:
            optimizer = AdamWeightDecay(group_params, learning_rate=lr_schedule, eps=cfg.AdamWeightDecay.eps)
    elif cfg.optimizer == "Thor":
        from src.utils import get_bert_thor_lr, get_bert_thor_damping
        lr = get_bert_thor_lr(cfg.Thor.lr_max, cfg.Thor.lr_min, cfg.Thor.lr_power, cfg.Thor.lr_total_steps)
        damping = get_bert_thor_damping(cfg.Thor.damping_max, cfg.Thor.damping_min, cfg.Thor.damping_power,
                                        cfg.Thor.damping_total_steps)
        split_indices = None
        if bert_net_cfg.num_hidden_layers == 12 and not bert_net_cfg.use_relative_positions:
            split_indices = [28, 55, 77]
        elif bert_net_cfg.num_hidden_layers == 24 and not bert_net_cfg.use_relative_positions:
            split_indices = [38, 93, 149]
        optimizer = thor(network, lr, damping, cfg.Thor.momentum,
                         cfg.Thor.weight_decay, cfg.Thor.loss_scale, cfg.batch_size,
                         decay_filter=lambda x: 'layernorm' not in x.name.lower() and 'bias' not in x.name.lower(),
                         split_indices=split_indices, enable_clip_grad=True, frequency=cfg.Thor.frequency)
    else:
        raise ValueError("Don't support optimizer {}, only support [Lamb, Momentum, AdamWeightDecay, Thor]".
                         format(cfg.optimizer))
    return optimizer


def _set_graph_kernel_context(device_target):
    """Add suitable graph kernel context for different configs."""
    if device_target == 'GPU':
        if cfg.bert_network == 'base':
            context.set_context(enable_graph_kernel=True,
                                graph_kernel_flags="--enable_stitch_fusion=true "
                                                   "--enable_parallel_fusion=true "
                                                   "--enable_cluster_ops=BatchMatMul")
        else:
            context.set_context(enable_graph_kernel=True)
    else:
        logger.warning('Graph kernel only supports GPU back-end now, run with graph kernel off.')


def _check_compute_type(args_opt):
    if args_opt.device_target == 'GPU' and bert_net_cfg.compute_type != mstype.float32 and cfg.bert_network != 'base':
        warning_message = 'Gpu only support fp32 temporarily, run with fp32.'
        bert_net_cfg.compute_type = mstype.float32
        if args_opt.enable_lossscale == "true":
            args_opt.enable_lossscale = "false"
            warning_message = 'Gpu only support fp32 temporarily, run with fp32 and disable lossscale.'
        logger.warning(warning_message)


def modelarts_pre_process():
    '''modelarts pre process function.'''
    cfg.device_id = get_device_id()
    cfg.device_num = get_device_num()
    cfg.data_dir = cfg.data_path
    cfg.save_checkpoint_path = os.path.join(cfg.output_path, cfg.save_checkpoint_path)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_pretrain():
    """pre-train bert_clue"""
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target, device_id=cfg.device_id)
    context.set_context(reserve_class_name_in_scope=False)
    _set_graph_kernel_context(cfg.device_target)
    ckpt_save_dir = cfg.save_checkpoint_path
    if cfg.distribute == "true":
        if cfg.device_target == 'Ascend':
            D.init()
            device_num = cfg.device_num
            rank = cfg.device_id % device_num
        else:
            D.init()
            device_num = D.get_group_size()
            rank = D.get_rank()
        ckpt_save_dir = os.path.join(cfg.save_checkpoint_path, 'ckpt_' + str(get_rank()))

        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)
        _set_bert_all_reduce_split()
    else:
        rank = 0
        device_num = 1

    _check_compute_type(cfg)

    if cfg.accumulation_steps > 1:
        logger.info("accumulation steps: {}".format(cfg.accumulation_steps))
        logger.info("global batch size: {}".format(cfg.batch_size * cfg.accumulation_steps))
        if cfg.enable_data_sink == "true":
            cfg.data_sink_steps *= cfg.accumulation_steps
            logger.info("data sink steps: {}".format(cfg.data_sink_steps))
        if cfg.enable_save_ckpt == "true":
            cfg.save_checkpoint_steps *= cfg.accumulation_steps
            logger.info("save checkpoint steps: {}".format(cfg.save_checkpoint_steps))

    ds = create_bert_dataset(device_num, rank, cfg.do_shuffle, cfg.data_dir, cfg.schema_dir, cfg.batch_size,
                             cfg.bucket_list)
    net_with_loss = BertNetworkWithLoss(bert_net_cfg, True)

    new_repeat_count = cfg.epoch_size * ds.get_dataset_size() // cfg.data_sink_steps
    if cfg.train_steps > 0:
        train_steps = cfg.train_steps * cfg.accumulation_steps
        new_repeat_count = min(new_repeat_count, train_steps // cfg.data_sink_steps)
    else:
        cfg.train_steps = cfg.epoch_size * ds.get_dataset_size() // cfg.accumulation_steps
        logger.info("train steps: {}".format(cfg.train_steps))

    optimizer = _get_optimizer(cfg, net_with_loss)
    callback = [TimeMonitor(cfg.data_sink_steps), LossCallBack(ds.get_dataset_size())]
    if cfg.enable_save_ckpt == "true" and cfg.device_id % min(8, device_num) == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                     keep_checkpoint_max=cfg.save_checkpoint_num)
        ckpoint_cb = ModelCheckpoint(prefix='checkpoint_bert',
                                     directory=None if ckpt_save_dir == "" else ckpt_save_dir, config=config_ck)
        callback.append(ckpoint_cb)

    if cfg.load_checkpoint_path:
        param_dict = load_checkpoint(cfg.load_checkpoint_path)
        load_param_into_net(net_with_loss, param_dict)

    if cfg.enable_lossscale == "true":
        update_cell = DynamicLossScaleUpdateCell(loss_scale_value=cfg.loss_scale_value,
                                                 scale_factor=cfg.scale_factor,
                                                 scale_window=cfg.scale_window)
        accumulation_steps = cfg.accumulation_steps
        enable_global_norm = cfg.enable_global_norm
        if accumulation_steps <= 1:
            if cfg.optimizer == 'AdamWeightDecay' and cfg.device_target == 'GPU':
                net_with_grads = BertTrainOneStepWithLossScaleCellForAdam(net_with_loss, optimizer=optimizer,
                                                                          scale_update_cell=update_cell)
            else:
                net_with_grads = BertTrainOneStepWithLossScaleCell(net_with_loss, optimizer=optimizer,
                                                                   scale_update_cell=update_cell)
        else:
            allreduce_post = cfg.distribute == "false" or cfg.allreduce_post_accumulation == "true"
            net_with_accumulation = (BertTrainAccumulationAllReducePostWithLossScaleCell if allreduce_post else
                                     BertTrainAccumulationAllReduceEachWithLossScaleCell)
            net_with_grads = net_with_accumulation(net_with_loss, optimizer=optimizer,
                                                   scale_update_cell=update_cell,
                                                   accumulation_steps=accumulation_steps,
                                                   enable_global_norm=enable_global_norm)
    else:
        net_with_grads = BertTrainOneStepCell(net_with_loss, optimizer=optimizer, enable_clip_grad=True)
        if cfg.optimizer == "Thor":
            net_with_grads = BertTrainOneStepCell(net_with_loss, optimizer=optimizer, sens=cfg.Thor.loss_scale,
                                                  enable_clip_grad=False)

    if cfg.bucket_list:
        net_with_grads = BertNetworkMatchBucket(net_with_grads, bert_net_cfg.seq_length, cfg.bucket_list)

    model = Model(net_with_grads)

    if cfg.train_with_eval == 'true':
        net_eval = BertPretrainEval(bert_net_cfg, network=net_with_loss.bert)
        eval_ds = create_eval_dataset(cfg.batch_size, device_num, rank, cfg.eval_data_dir, cfg.schema_dir)
        model = Model(net_with_grads, eval_network=net_eval, metrics={'bert_acc': BertMetric(cfg.batch_size)})
        eval_callback = EvalCallBack(model, eval_ds, device_num * cfg.batch_size, cfg.eval_samples)
        callback.append(eval_callback)

    model = ConvertModelUtils().convert_to_thor_model(model, network=net_with_grads, optimizer=optimizer)
    model.train(new_repeat_count, ds, callbacks=callback,
                dataset_sink_mode=(cfg.enable_data_sink == "true"), sink_size=cfg.data_sink_steps)


if __name__ == '__main__':
    set_seed(0)
    run_pretrain()
