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

'''
dynamic ranker train and evaluation script.
'''

import os
import mindspore.communication.management as D
from mindspore import context
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.optim import AdamWeightDecay, Lamb, Momentum
from mindspore.train.model import Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.context import ParallelMode
from mindspore.common import set_seed
from src.dynamic_ranker import DynamicRankerPredict, DynamicRankerFinetuneCell, DynamicRankerBase, DynamicRanker
from src.dataset import create_dyr_base_dataset, create_dyr_dataset_predict, create_dyr_dataset
from src.utils import make_directory, LossCallBack, LoadNewestCkpt, DynamicRankerLearningRate, MRR
from src.model_utils.config import config as args_opt, optimizer_cfg, dyr_net_cfg
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id


_cur_dir = os.getcwd()


def do_train(dataset=None, network=None, load_checkpoint_path="", save_checkpoint_path="", epoch_num=1, prefix="dyr"):
    """ do train """
    if load_checkpoint_path == "":
        raise ValueError("Pretrain model missed, finetune task must load pretrain model!")
    steps_per_epoch = dataset.get_dataset_size()
    # optimizer
    if optimizer_cfg.optimizer == 'AdamWeightDecay':
        lr_schedule = DynamicRankerLearningRate(learning_rate=optimizer_cfg.AdamWeightDecay.learning_rate,
                                                end_learning_rate=optimizer_cfg.AdamWeightDecay.end_learning_rate,
                                                warmup_steps=int(steps_per_epoch * epoch_num * 0.1),
                                                decay_steps=steps_per_epoch * epoch_num,
                                                power=optimizer_cfg.AdamWeightDecay.power)
        params = network.trainable_params()
        decay_params = list(filter(optimizer_cfg.AdamWeightDecay.decay_filter, params))
        other_params = list(filter(lambda x: not optimizer_cfg.AdamWeightDecay.decay_filter(x), params))
        group_params = [{'params': decay_params, 'weight_decay': optimizer_cfg.AdamWeightDecay.weight_decay},
                        {'params': other_params, 'weight_decay': 0.0}]

        optimizer = AdamWeightDecay(group_params, lr_schedule, eps=optimizer_cfg.AdamWeightDecay.eps)
    elif optimizer_cfg.optimizer == 'Lamb':
        lr_schedule = DynamicRankerLearningRate(learning_rate=optimizer_cfg.Lamb.learning_rate,
                                                end_learning_rate=optimizer_cfg.Lamb.end_learning_rate,
                                                warmup_steps=int(steps_per_epoch * epoch_num * 0.1),
                                                decay_steps=steps_per_epoch * epoch_num,
                                                power=optimizer_cfg.Lamb.power)
        optimizer = Lamb(network.trainable_params(), learning_rate=lr_schedule)
    elif optimizer_cfg.optimizer == 'Momentum':
        optimizer = Momentum(network.trainable_params(), learning_rate=optimizer_cfg.Momentum.learning_rate,
                             momentum=optimizer_cfg.Momentum.momentum)
    else:
        raise Exception("Optimizer not supported. support: [AdamWeightDecay, Lamb, Momentum]")

    # load checkpoint into network
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix=prefix,
                                 directory=None if save_checkpoint_path == "" else save_checkpoint_path,
                                 config=ckpt_config)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(network, param_dict)

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2**16, scale_factor=2, scale_window=1000)
    netwithgrads = DynamicRankerFinetuneCell(network, optimizer=optimizer, scale_update_cell=update_cell)
    model = Model(netwithgrads)
    callbacks = [TimeMonitor(dataset.get_dataset_size()), LossCallBack(dataset.get_dataset_size()), ckpoint_cb]
    model.train(epoch_num, dataset, callbacks=callbacks)

def do_predict(rank_id=0, dataset=None, network=None, load_checkpoint_path="",
               eval_ids_path="", eval_qrels_path="", save_score_path=""):
    """ do eval """
    if load_checkpoint_path == "":
        raise ValueError("Finetune model missed, evaluation task must load finetune model!")
    net_for_pretraining = network(dyr_net_cfg, False, dropout_prob=0.0)
    net_for_pretraining.set_train(False)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(net_for_pretraining, param_dict)
    model = Model(net_for_pretraining)
    columns_list = ["input_ids", "input_mask", "segment_ids"]
    loss = []
    for data in dataset.create_dict_iterator(num_epochs=1):
        input_data = []
        for i in columns_list:
            input_data.append(data[i])
        input_ids, input_mask, token_type_id = input_data
        logits = model.predict(input_ids, input_mask, token_type_id)
        print(logits)
        logits = logits[0][0].asnumpy()
        loss.append(logits)
    pred_qids = []
    pred_pids = []
    with open(eval_ids_path) as f:
        for l in f:
            q, p = l.split()
            pred_qids.append(q)
            pred_pids.append(p)
    if len(pred_qids) != len(loss):
        raise ValueError("len(pred_qids) != len(loss)!")

    with open(save_score_path, "w") as writer:
        for qid, pid, score in zip(pred_qids, pred_pids, loss):
            writer.write(f'{qid}\t{pid}\t{score}\n')

    mrr = MRR()
    mrr.accuracy(qrels_path=eval_qrels_path, scores_path=save_score_path)


def modelarts_pre_process():
    '''modelarts pre process function.'''
    args_opt.device_id = get_device_id()
    _file_dir = os.path.dirname(os.path.abspath(__file__))
    args_opt.load_pretrain_checkpoint_path = os.path.join(_file_dir, args_opt.load_pretrain_checkpoint_path)
    args_opt.load_finetune_checkpoint_path = os.path.join(args_opt.output_path, args_opt.load_finetune_checkpoint_path)
    args_opt.save_finetune_checkpoint_path = os.path.join(args_opt.output_path, args_opt.save_finetune_checkpoint_path)
    if args_opt.schema_file_path:
        args_opt.schema_file_path = os.path.join(args_opt.data_path, args_opt.schema_file_path)
    args_opt.train_data_file_path = os.path.join(args_opt.data_path, args_opt.train_data_file_path)
    args_opt.eval_data_file_path = os.path.join(args_opt.data_path, args_opt.eval_data_file_path)
    args_opt.save_score_path = os.path.join(args_opt.output_path, args_opt.save_score_path)
    args_opt.eval_ids_path = os.path.join(args_opt.data_path, args_opt.eval_ids_path)
    args_opt.eval_qrels_path = os.path.join(args_opt.data_path, args_opt.eval_qrels_path)

@moxing_wrapper(pre_process=modelarts_pre_process)
def run_dyr():
    """run dyr task"""
    set_seed(1234)
    if args_opt.do_train.lower() == "false" and args_opt.do_eval.lower() == "false":
        raise ValueError("At least one of 'do_train' or 'do_eval' must be true")
    if args_opt.do_train.lower() == "true" and args_opt.train_data_file_path == "":
        raise ValueError("'train_data_file_path' must be set when do finetune task")
    if args_opt.do_eval.lower() == "true" and args_opt.eval_data_file_path == "":
        raise ValueError("'eval_data_file_path' must be set when do evaluation task")
    epoch_num = args_opt.epoch_num
    load_pretrain_checkpoint_path = args_opt.load_pretrain_checkpoint_path
    save_finetune_checkpoint_path = args_opt.save_finetune_checkpoint_path
    load_finetune_checkpoint_path = args_opt.load_finetune_checkpoint_path
    eval_ids_path = args_opt.eval_ids_path
    eval_qrels_path = args_opt.eval_qrels_path
    save_score_path = args_opt.save_score_path
    target = args_opt.device_target
    if target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",
                            device_id=args_opt.device_id, save_graphs=False)
    else:
        raise Exception("Target error, Ascend is supported.")
    D.init()
    device_num = D.get_group_size()
    rank_id = D.get_rank()
    save_finetune_checkpoint_path = os.path.join(args_opt.save_finetune_checkpoint_path, 'ckpt_' + str(rank_id))

    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                      device_num=device_num)

    create_dataset = create_dyr_base_dataset
    dyr_network = DynamicRankerBase
    if args_opt.dyr_version.lower() == "dyr":
        create_dataset = create_dyr_dataset
        dyr_network = DynamicRanker

    netwithloss = dyr_network(dyr_net_cfg, True, dropout_prob=0.1,
                              batch_size=args_opt.train_batch_size,
                              group_size=args_opt.group_size,
                              group_num=args_opt.group_num,
                              rank_id=rank_id,
                              device_num=device_num)

    if args_opt.do_train.lower() == "true":
        ds = create_dataset(device_num, rank_id, batch_size=args_opt.train_batch_size, repeat_count=1,
                            data_file_path=args_opt.train_data_file_path,
                            schema_file_path=args_opt.schema_file_path,
                            do_shuffle=(args_opt.train_data_shuffle.lower() == "true"),
                            group_size=args_opt.group_size, group_num=args_opt.group_num)
        do_train(ds, netwithloss, load_pretrain_checkpoint_path, save_finetune_checkpoint_path,
                 epoch_num, args_opt.dyr_version.lower())

        if args_opt.do_eval.lower() == "true":
            if save_finetune_checkpoint_path == "":
                load_finetune_checkpoint_dir = _cur_dir
            else:
                load_finetune_checkpoint_dir = make_directory(save_finetune_checkpoint_path)
            load_finetune_checkpoint_path = LoadNewestCkpt(load_finetune_checkpoint_dir,
                                                           ds.get_dataset_size(), epoch_num,
                                                           args_opt.dyr_version.lower())

    if args_opt.do_eval.lower() == "true":
        if rank_id == 0:
            ds = create_dyr_dataset_predict(batch_size=args_opt.eval_batch_size, repeat_count=1,
                                            data_file_path=args_opt.eval_data_file_path,
                                            schema_file_path=args_opt.schema_file_path,
                                            do_shuffle=(args_opt.eval_data_shuffle.lower() == "true"))
            do_predict(rank_id, ds, DynamicRankerPredict, load_finetune_checkpoint_path,
                       eval_ids_path, eval_qrels_path, save_score_path)


if __name__ == "__main__":
    run_dyr()
