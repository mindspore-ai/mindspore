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
ERNIE finetune and evaluation script.
'''

import os
import argparse
import time
import json
from src.ernie_for_finetune import ErnieFinetuneCell, ErnieNER
from src.finetune_eval_config import optimizer_cfg, ernie_net_cfg
from src.dataset import create_finetune_dataset
from src.utils import make_directory, LossCallBack, LoadNewestCkpt, ErnieLearningRate
from src.assessment_method import SpanF1
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore import log as logger
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.optim import AdamWeightDecay, Adam, Adagrad
from mindspore.train.model import Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

_cur_dir = os.getcwd()


def do_train(task_type, dataset=None, network=None, load_checkpoint_path="", save_checkpoint_path="", epoch_num=1):
    """ do train """
    if load_checkpoint_path == "":
        raise ValueError("Pretrain model missed, finetune task must load pretrain model!")
    steps_per_epoch = dataset.get_dataset_size()
    # optimizer
    if optimizer_cfg.optimizer == 'AdamWeightDecay':
        lr_schedule = ErnieLearningRate(learning_rate=optimizer_cfg.AdamWeightDecay.learning_rate,
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
    elif optimizer_cfg.optimizer == 'Adam':
        optimizer = Adam(network.trainable_params(), learning_rate=optimizer_cfg.Adam.learning_rate)
    elif optimizer_cfg.optimizer == 'Adagrad':
        optimizer = Adagrad(network.trainable_params(), learning_rate=optimizer_cfg.Adagrad.learning_rate)
    else:
        raise Exception("Optimizer not supported. support: [AdamWeightDecay, Lamb, Momentum]")

    # load checkpoint into network
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix=task_type,
                                 directory=None if save_checkpoint_path == "" else save_checkpoint_path,
                                 config=ckpt_config)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(network, param_dict)

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2**32, scale_factor=2, scale_window=1000)
    netwithgrads = ErnieFinetuneCell(network, optimizer=optimizer, scale_update_cell=update_cell)
    model = Model(netwithgrads)
    callbacks = [TimeMonitor(dataset.get_dataset_size()), LossCallBack(dataset.get_dataset_size()), ckpoint_cb]
    train_begin = time.time()
    model.train(epoch_num, dataset, callbacks=callbacks)
    train_end = time.time()
    print("latency: {:.6f} s".format(train_end - train_begin))

def eval_result_print(assessment_method="accuracy", callback=None):
    """print eval result"""
    print("Precision {:.6f} ".format(callback.TP / (callback.TP + callback.FP)))
    print("Recall {:.6f} ".format(callback.TP / (callback.TP + callback.FN)))
    print("F1 {:.6f} ".format(2 * callback.TP / (2 * callback.TP + callback.FP + callback.FN)))

def do_eval(dataset=None, network=None, num_class=41, assessment_method="accuracy", data_file="",
            load_checkpoint_path="", vocab_file="", label_file="", tag_to_index=None, batch_size=1):
    """ do eval """
    if load_checkpoint_path == "":
        raise ValueError("Finetune model missed, evaluation task must load finetune model!")
    net_for_pretraining = network(ernie_net_cfg, batch_size, False, num_class, tag_to_index=tag_to_index)
    net_for_pretraining.set_train(False)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(net_for_pretraining, param_dict)
    model = Model(net_for_pretraining)

    callback = SpanF1(tag_to_index)

    evaluate_times = []
    columns_list = ["input_ids", "input_mask", "token_type_id", "label_ids"]
    for data in dataset.create_dict_iterator(num_epochs=1):
        input_data = []
        for i in columns_list:
            input_data.append(data[i])
        input_ids, input_mask, token_type_id, label_ids = input_data
        time_begin = time.time()
        logits = model.predict(input_ids, input_mask, token_type_id, label_ids)
        time_end = time.time()
        evaluate_times.append(time_end - time_begin)
        callback.update(logits, label_ids)
    print("==============================================================")
    eval_result_print(assessment_method, callback)
    print("(w/o first and last) elapsed time: {}, per step time : {}".format(
        sum(evaluate_times), sum(evaluate_times)/len(evaluate_times)))
    print("==============================================================")

def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="run ner")
    parser.add_argument("--task_type", type=str, default="msra_ner", choices=["msra_ner"],
                        help="Task type, default is msra_ner")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["Ascend", "GPU"],
                        help="Device type, default is Ascend")
    parser.add_argument("--do_train", type=str, default="false", choices=["true", "false"],
                        help="Eable train, default is false")
    parser.add_argument("--do_eval", type=str, default="false", choices=["true", "false"],
                        help="Eable eval, default is false")
    parser.add_argument("--number_labels", type=int, default=0, help='Number of NER labels, default is 0')
    parser.add_argument("--label_map_config", type=str, default="", help="Label map file path")
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--epoch_num", type=int, default=5, help="Epoch number, default is 5.")
    parser.add_argument("--train_data_shuffle", type=str, default="true", choices=["true", "false"],
                        help="Enable train data shuffle, default is true")
    parser.add_argument("--eval_data_shuffle", type=str, default="false", choices=["true", "false"],
                        help="Enable eval data shuffle, default is false")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Train batch size, default is 32")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Eval batch size, default is 1")
    parser.add_argument("--vocab_file_path", type=str, default="", help="Vocab file path, used in clue benchmark")
    parser.add_argument("--label_file_path", type=str, default="", help="label file path, used in clue benchmark")
    parser.add_argument("--save_finetune_checkpoint_path", type=str, default="", help="Save checkpoint path")
    parser.add_argument("--load_pretrain_checkpoint_path", type=str, default="", help="Load checkpoint file path")
    parser.add_argument("--load_finetune_checkpoint_path", type=str, default="", help="Load checkpoint file path")
    parser.add_argument("--train_data_file_path", type=str, default="",
                        help="Data path, it is better to use absolute path")
    parser.add_argument("--eval_data_file_path", type=str, default="",
                        help="Data path, it is better to use absolute path")
    parser.add_argument("--eval_json_path", type=str, default="",
                        help="Json data path, it is better to use absolute path")
    parser.add_argument("--vocab_path", type=str, default="", help="vocab file")
    parser.add_argument("--dataset_format", type=str, default="mindrecord", choices=["mindrecord", "tfrecord"],
                        help="Dataset format, support mindrecord or tfrecord")
    parser.add_argument("--schema_file_path", type=str, default="",
                        help="Schema path, it is better to use absolute path")
    args_opt = parser.parse_args()
    if args_opt.do_train.lower() == "false" and args_opt.do_eval.lower() == "false":
        raise ValueError("At least one of 'do_train' or 'do_eval' must be true")
    if args_opt.do_train.lower() == "true" and args_opt.train_data_file_path == "":
        raise ValueError("'train_data_file_path' must be set when do finetune task")
    if args_opt.do_eval.lower() == "true" and args_opt.eval_data_file_path == "":
        raise ValueError("'eval_data_file_path' must be set when do evaluation task")
    return args_opt

def run_ner():
    """run ner task"""
    args_opt = parse_args()
    epoch_num = args_opt.epoch_num
    assessment_method = "f1"
    load_pretrain_checkpoint_path = args_opt.load_pretrain_checkpoint_path
    save_finetune_checkpoint_path = args_opt.save_finetune_checkpoint_path
    load_finetune_checkpoint_path = args_opt.load_finetune_checkpoint_path
    target = args_opt.device_target
    if target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args_opt.device_id)
    elif target == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
        if ernie_net_cfg.compute_type != mstype.float32:
            logger.warning('GPU only support fp32 temporarily, run with fp32.')
            ernie_net_cfg.compute_type = mstype.float32
        if optimizer_cfg.optimizer == 'AdamWeightDecay':
            context.set_context(enable_graph_kernel=True)
    else:
        raise Exception("Target error, GPU or Ascend is supported.")
    with open(args_opt.label_map_config) as f:
        tag_to_index = json.load(f)
    number_labels = args_opt.number_labels
    if args_opt.do_train.lower() == "true":
        netwithloss = ErnieNER(ernie_net_cfg, args_opt.train_batch_size, True, num_labels=number_labels,
                               tag_to_index=tag_to_index, dropout_prob=0.1)
        ds = create_finetune_dataset(batch_size=args_opt.train_batch_size,
                                     repeat_count=1,
                                     data_file_path=args_opt.train_data_file_path,
                                     do_shuffle=(args_opt.train_data_shuffle.lower() == "true"))
        print("==============================================================")
        print("processor_name: {}".format(args_opt.device_target))
        print("test_name: ERNIE Finetune Training")
        print("model_name: {}".format("ERNIE + MLP"))
        print("batch_size: {}".format(args_opt.train_batch_size))

        do_train(args_opt.task_type, ds, netwithloss, load_pretrain_checkpoint_path,
                 save_finetune_checkpoint_path, epoch_num)

        if args_opt.do_eval.lower() == "true":
            if save_finetune_checkpoint_path == "":
                load_finetune_checkpoint_dir = _cur_dir
            else:
                load_finetune_checkpoint_dir = make_directory(save_finetune_checkpoint_path)
            load_finetune_checkpoint_path = LoadNewestCkpt(load_finetune_checkpoint_dir,
                                                           ds.get_dataset_size(), epoch_num, args_opt.task_type)

    if args_opt.do_eval.lower() == "true":
        ds = create_finetune_dataset(batch_size=args_opt.eval_batch_size,
                                     repeat_count=1,
                                     data_file_path=args_opt.eval_data_file_path,
                                     do_shuffle=(args_opt.eval_data_shuffle.lower() == "true"))
        do_eval(ds, ErnieNER, number_labels, assessment_method,
                args_opt.eval_data_file_path, load_finetune_checkpoint_path, args_opt.vocab_file_path,
                args_opt.label_file_path, tag_to_index, args_opt.eval_batch_size)

if __name__ == "__main__":
    run_ner()
