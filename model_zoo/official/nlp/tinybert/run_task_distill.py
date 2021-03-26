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

"""task distill script"""

import os
import re
import argparse
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore.train.model import Model
from mindspore.train.callback import TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.optim import AdamWeightDecay
from mindspore import log as logger
from src.dataset import create_tinybert_dataset, DataType
from src.utils import LossCallBack, ModelSaveCkpt, EvalCallBack, BertLearningRate
from src.assessment_method import Accuracy, F1
from src.td_config import phase1_cfg, phase2_cfg, eval_cfg, td_teacher_net_cfg, td_student_net_cfg
from src.tinybert_for_gd_td import BertEvaluationWithLossScaleCell, BertNetworkWithLoss_td, BertEvaluationCell
from src.tinybert_model import BertModelCLS, BertModelNER

_cur_dir = os.getcwd()
td_phase1_save_ckpt_dir = os.path.join(_cur_dir, 'tinybert_td_phase1_save_ckpt')
td_phase2_save_ckpt_dir = os.path.join(_cur_dir, 'tinybert_td_phase2_save_ckpt')
if not os.path.exists(td_phase1_save_ckpt_dir):
    os.makedirs(td_phase1_save_ckpt_dir)
if not os.path.exists(td_phase2_save_ckpt_dir):
    os.makedirs(td_phase2_save_ckpt_dir)

def parse_args():
    """
    parse args
    """
    parser = argparse.ArgumentParser(description='tinybert task distill')
    parser.add_argument("--device_target", type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented. (Default: Ascend)')
    parser.add_argument("--do_train", type=str, default="true", choices=["true", "false"],
                        help="Do train task, default is true.")
    parser.add_argument("--do_eval", type=str, default="true", choices=["true", "false"],
                        help="Do eval task, default is true.")
    parser.add_argument("--td_phase1_epoch_size", type=int, default=10,
                        help="Epoch size for td phase 1, default is 10.")
    parser.add_argument("--td_phase2_epoch_size", type=int, default=3, help="Epoch size for td phase 2, default is 3.")
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--do_shuffle", type=str, default="true", choices=["true", "false"],
                        help="Enable shuffle for dataset, default is true.")
    parser.add_argument("--enable_data_sink", type=str, default="true", choices=["true", "false"],
                        help="Enable data sink, default is true.")
    parser.add_argument("--save_ckpt_step", type=int, default=100, help="Enable data sink, default is true.")
    parser.add_argument("--max_ckpt_num", type=int, default=1, help="Enable data sink, default is true.")
    parser.add_argument("--data_sink_steps", type=int, default=1, help="Sink steps for each epoch, default is 1.")
    parser.add_argument("--load_teacher_ckpt_path", type=str, default="", help="Load checkpoint file path")
    parser.add_argument("--load_gd_ckpt_path", type=str, default="", help="Load checkpoint file path")
    parser.add_argument("--load_td1_ckpt_path", type=str, default="", help="Load checkpoint file path")
    parser.add_argument("--train_data_dir", type=str, default="", help="Data path, it is better to use absolute path")
    parser.add_argument("--eval_data_dir", type=str, default="", help="Data path, it is better to use absolute path")
    parser.add_argument("--schema_dir", type=str, default="", help="Schema path, it is better to use absolute path")
    parser.add_argument("--task_type", type=str, default="classification", choices=["classification", "ner"],
                        help="The type of the task to train.")
    parser.add_argument("--task_name", type=str, default="", choices=["SST-2", "QNLI", "MNLI", "TNEWS", "CLUENER"],
                        help="The name of the task to train.")
    parser.add_argument("--assessment_method", type=str, default="accuracy", choices=["accuracy", "bf1", "mf1"],
                        help="assessment_method include: [accuracy, bf1, mf1], default is accuracy")
    parser.add_argument("--dataset_type", type=str, default="tfrecord",
                        help="dataset type tfrecord/mindrecord, default is tfrecord")
    args = parser.parse_args()
    if args.do_train.lower() != "true" and args.do_eval.lower() != "true":
        raise ValueError("do train or do eval must have one be true, please confirm your config")
    if args.task_name in ["SST-2", "QNLI", "MNLI", "TNEWS"] and args.task_type != "classification":
        raise ValueError(f"{args.task_name} is a classification dataset, please set --task_type=classification")
    if args.task_name in ["CLUENER"] and args.task_type != "ner":
        raise ValueError(f"{args.task_name} is a ner dataset, please set --task_type=ner")
    if args.task_name in ["SST-2", "QNLI", "MNLI"] and \
        (td_teacher_net_cfg.vocab_size != 30522 or td_student_net_cfg.vocab_size != 30522):
        logger.warning(f"{args.task_name} is an English dataset. Usually, we use 21128 for CN vocabs and 30522 for "\
                       "EN vocabs according to the origin paper.")
    if args.task_name in ["TNEWS", "CLUENER"] and \
            (td_teacher_net_cfg.vocab_size != 21128 or td_student_net_cfg.vocab_size != 21128):
        logger.warning(f"{args.task_name} is a Chinese dataset. Usually, we use 21128 for CN vocabs and 30522 for " \
                       "EN vocabs according to the origin paper.")
    return args


args_opt = parse_args()
if args_opt.dataset_type == "tfrecord":
    dataset_type = DataType.TFRECORD
elif args_opt.dataset_type == "mindrecord":
    dataset_type = DataType.MINDRECORD
else:
    raise Exception("dataset format is not supported yet")
DEFAULT_NUM_LABELS = 2
DEFAULT_SEQ_LENGTH = 128
task_params = {"SST-2": {"num_labels": 2, "seq_length": 64},
               "QNLI": {"num_labels": 2, "seq_length": 128},
               "MNLI": {"num_labels": 3, "seq_length": 128},
               "TNEWS": {"num_labels": 15, "seq_length": 128},
               "CLUENER": {"num_labels": 43, "seq_length": 128}}


class Task:
    """
    Encapsulation class of get the task parameter.
    """
    def __init__(self, task_name):
        self.task_name = task_name

    @property
    def num_labels(self):
        if self.task_name in task_params and "num_labels" in task_params[self.task_name]:
            return task_params[self.task_name]["num_labels"]
        return DEFAULT_NUM_LABELS

    @property
    def seq_length(self):
        if self.task_name in task_params and "seq_length" in task_params[self.task_name]:
            return task_params[self.task_name]["seq_length"]
        return DEFAULT_SEQ_LENGTH
task = Task(args_opt.task_name)


def run_predistill():
    """
    run predistill
    """
    cfg = phase1_cfg
    load_teacher_checkpoint_path = args_opt.load_teacher_ckpt_path
    load_student_checkpoint_path = args_opt.load_gd_ckpt_path
    netwithloss = BertNetworkWithLoss_td(teacher_config=td_teacher_net_cfg, teacher_ckpt=load_teacher_checkpoint_path,
                                         student_config=td_student_net_cfg, student_ckpt=load_student_checkpoint_path,
                                         is_training=True, task_type=args_opt.task_type,
                                         num_labels=task.num_labels, is_predistill=True)

    rank = 0
    device_num = 1
    dataset = create_tinybert_dataset('td', cfg.batch_size,
                                      device_num, rank, args_opt.do_shuffle,
                                      args_opt.train_data_dir, args_opt.schema_dir,
                                      data_type=dataset_type)

    dataset_size = dataset.get_dataset_size()
    print('td1 dataset size: ', dataset_size)
    print('td1 dataset repeatcount: ', dataset.get_repeat_count())
    if args_opt.enable_data_sink == 'true':
        repeat_count = args_opt.td_phase1_epoch_size * dataset_size // args_opt.data_sink_steps
        time_monitor_steps = args_opt.data_sink_steps
    else:
        repeat_count = args_opt.td_phase1_epoch_size
        time_monitor_steps = dataset_size

    optimizer_cfg = cfg.optimizer_cfg

    lr_schedule = BertLearningRate(learning_rate=optimizer_cfg.AdamWeightDecay.learning_rate,
                                   end_learning_rate=optimizer_cfg.AdamWeightDecay.end_learning_rate,
                                   warmup_steps=int(dataset_size / 10),
                                   decay_steps=int(dataset_size * args_opt.td_phase1_epoch_size),
                                   power=optimizer_cfg.AdamWeightDecay.power)
    params = netwithloss.trainable_params()
    decay_params = list(filter(optimizer_cfg.AdamWeightDecay.decay_filter, params))
    other_params = list(filter(lambda x: not optimizer_cfg.AdamWeightDecay.decay_filter(x), params))
    group_params = [{'params': decay_params, 'weight_decay': optimizer_cfg.AdamWeightDecay.weight_decay},
                    {'params': other_params, 'weight_decay': 0.0},
                    {'order_params': params}]

    optimizer = AdamWeightDecay(group_params, learning_rate=lr_schedule, eps=optimizer_cfg.AdamWeightDecay.eps)
    callback = [TimeMonitor(time_monitor_steps), LossCallBack(), ModelSaveCkpt(netwithloss.bert,
                                                                               args_opt.save_ckpt_step,
                                                                               args_opt.max_ckpt_num,
                                                                               td_phase1_save_ckpt_dir)]
    if enable_loss_scale:
        update_cell = DynamicLossScaleUpdateCell(loss_scale_value=cfg.loss_scale_value,
                                                 scale_factor=cfg.scale_factor,
                                                 scale_window=cfg.scale_window)
        netwithgrads = BertEvaluationWithLossScaleCell(netwithloss, optimizer=optimizer, scale_update_cell=update_cell)
    else:
        netwithgrads = BertEvaluationCell(netwithloss, optimizer=optimizer)

    model = Model(netwithgrads)
    model.train(repeat_count, dataset, callbacks=callback,
                dataset_sink_mode=(args_opt.enable_data_sink == 'true'),
                sink_size=args_opt.data_sink_steps)

def run_task_distill(ckpt_file):
    """
    run task distill
    """
    if ckpt_file == '':
        raise ValueError("Student ckpt file should not be None")
    cfg = phase2_cfg

    load_teacher_checkpoint_path = args_opt.load_teacher_ckpt_path
    load_student_checkpoint_path = ckpt_file
    netwithloss = BertNetworkWithLoss_td(teacher_config=td_teacher_net_cfg, teacher_ckpt=load_teacher_checkpoint_path,
                                         student_config=td_student_net_cfg, student_ckpt=load_student_checkpoint_path,
                                         is_training=True, task_type=args_opt.task_type,
                                         num_labels=task.num_labels, is_predistill=False)

    rank = 0
    device_num = 1
    train_dataset = create_tinybert_dataset('td', cfg.batch_size,
                                            device_num, rank, args_opt.do_shuffle,
                                            args_opt.train_data_dir, args_opt.schema_dir,
                                            data_type=dataset_type)

    dataset_size = train_dataset.get_dataset_size()
    print('td2 train dataset size: ', dataset_size)
    print('td2 train dataset repeatcount: ', train_dataset.get_repeat_count())
    if args_opt.enable_data_sink == 'true':
        repeat_count = args_opt.td_phase2_epoch_size * train_dataset.get_dataset_size() // args_opt.data_sink_steps
        time_monitor_steps = args_opt.data_sink_steps
    else:
        repeat_count = args_opt.td_phase2_epoch_size
        time_monitor_steps = dataset_size

    optimizer_cfg = cfg.optimizer_cfg

    lr_schedule = BertLearningRate(learning_rate=optimizer_cfg.AdamWeightDecay.learning_rate,
                                   end_learning_rate=optimizer_cfg.AdamWeightDecay.end_learning_rate,
                                   warmup_steps=int(dataset_size * args_opt.td_phase2_epoch_size / 10),
                                   decay_steps=int(dataset_size * args_opt.td_phase2_epoch_size),
                                   power=optimizer_cfg.AdamWeightDecay.power)
    params = netwithloss.trainable_params()
    decay_params = list(filter(optimizer_cfg.AdamWeightDecay.decay_filter, params))
    other_params = list(filter(lambda x: not optimizer_cfg.AdamWeightDecay.decay_filter(x), params))
    group_params = [{'params': decay_params, 'weight_decay': optimizer_cfg.AdamWeightDecay.weight_decay},
                    {'params': other_params, 'weight_decay': 0.0},
                    {'order_params': params}]

    optimizer = AdamWeightDecay(group_params, learning_rate=lr_schedule, eps=optimizer_cfg.AdamWeightDecay.eps)

    eval_dataset = create_tinybert_dataset('td', eval_cfg.batch_size,
                                           device_num, rank, args_opt.do_shuffle,
                                           args_opt.eval_data_dir, args_opt.schema_dir,
                                           data_type=dataset_type)
    print('td2 eval dataset size: ', eval_dataset.get_dataset_size())

    if args_opt.do_eval.lower() == "true":
        callback = [TimeMonitor(time_monitor_steps), LossCallBack(),
                    EvalCallBack(netwithloss.bert, eval_dataset)]
    else:
        callback = [TimeMonitor(time_monitor_steps), LossCallBack(),
                    ModelSaveCkpt(netwithloss.bert,
                                  args_opt.save_ckpt_step,
                                  args_opt.max_ckpt_num,
                                  td_phase2_save_ckpt_dir)]
    if enable_loss_scale:
        update_cell = DynamicLossScaleUpdateCell(loss_scale_value=cfg.loss_scale_value,
                                                 scale_factor=cfg.scale_factor,
                                                 scale_window=cfg.scale_window)

        netwithgrads = BertEvaluationWithLossScaleCell(netwithloss, optimizer=optimizer, scale_update_cell=update_cell)
    else:
        netwithgrads = BertEvaluationCell(netwithloss, optimizer=optimizer)
    model = Model(netwithgrads)
    model.train(repeat_count, train_dataset, callbacks=callback,
                dataset_sink_mode=(args_opt.enable_data_sink == 'true'),
                sink_size=args_opt.data_sink_steps)

def eval_result_print(assessment_method="accuracy", callback=None):
    """print eval result"""
    if assessment_method == "accuracy":
        print("============== acc is {}".format(callback.acc_num / callback.total_num))
    elif assessment_method == "bf1":
        print("Precision {:.6f} ".format(callback.TP / (callback.TP + callback.FP)))
        print("Recall {:.6f} ".format(callback.TP / (callback.TP + callback.FN)))
        print("F1 {:.6f} ".format(2 * callback.TP / (2 * callback.TP + callback.FP + callback.FN)))
    elif assessment_method == "mf1":
        print("F1 {:.6f} ".format(callback.eval()))
    else:
        raise ValueError("Assessment method not supported, support: [accuracy, f1]")

def do_eval_standalone():
    """
    do eval standalone
    """
    ckpt_file = args_opt.load_td1_ckpt_path
    if ckpt_file == '':
        raise ValueError("Student ckpt file should not be None")
    if args_opt.task_type == "classification":
        eval_model = BertModelCLS(td_student_net_cfg, False, task.num_labels, 0.0, phase_type="student")
    elif args_opt.task_type == "ner":
        eval_model = BertModelNER(td_student_net_cfg, False, task.num_labels, 0.0, phase_type="student")
    else:
        raise ValueError(f"Not support the task type {args_opt.task_type}")
    param_dict = load_checkpoint(ckpt_file)
    new_param_dict = {}
    for key, value in param_dict.items():
        new_key = re.sub('tinybert_', 'bert_', key)
        new_key = re.sub('^bert.', '', new_key)
        new_param_dict[new_key] = value
    load_param_into_net(eval_model, new_param_dict)
    eval_model.set_train(False)

    eval_dataset = create_tinybert_dataset('td', batch_size=eval_cfg.batch_size,
                                           device_num=1, rank=0, do_shuffle="false",
                                           data_dir=args_opt.eval_data_dir,
                                           schema_dir=args_opt.schema_dir,
                                           data_type=dataset_type)
    print('eval dataset size: ', eval_dataset.get_dataset_size())
    print('eval dataset batch size: ', eval_dataset.get_batch_size())
    if args_opt.assessment_method == "accuracy":
        callback = Accuracy()
    elif args_opt.assessment_method == "bf1":
        callback = F1(num_labels=task.num_labels)
    elif args_opt.assessment_method == "mf1":
        callback = F1(num_labels=task.num_labels, mode="MultiLabel")
    else:
        raise ValueError("Assessment method not supported, support: [accuracy, f1]")
    columns_list = ["input_ids", "input_mask", "segment_ids", "label_ids"]
    for data in eval_dataset.create_dict_iterator(num_epochs=1):
        input_data = []
        for i in columns_list:
            input_data.append(data[i])
        input_ids, input_mask, token_type_id, label_ids = input_data
        logits = eval_model(input_ids, token_type_id, input_mask)
        callback.update(logits, label_ids)
    print("==============================================================")
    eval_result_print(args_opt.assessment_method, callback)
    print("==============================================================")


if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target,
                        reserve_class_name_in_scope=False)
    if args_opt.device_target == "Ascend":
        context.set_context(device_id=args_opt.device_id)
    enable_loss_scale = True
    if args_opt.device_target == "GPU":
        if td_student_net_cfg.compute_type != mstype.float32:
            logger.warning('Compute about the student only support float32 temporarily, run with float32.')
            td_student_net_cfg.compute_type = mstype.float32
        # Backward of the network are calculated using fp32,
        # and the loss scale is not necessary
        enable_loss_scale = False

    if args_opt.device_target == "CPU":
        logger.warning('CPU only support float32 temporarily, run with float32.')
        td_teacher_net_cfg.dtype = mstype.float32
        td_teacher_net_cfg.compute_type = mstype.float32
        td_student_net_cfg.dtype = mstype.float32
        td_student_net_cfg.compute_type = mstype.float32
        enable_loss_scale = False

    td_teacher_net_cfg.seq_length = task.seq_length
    td_student_net_cfg.seq_length = task.seq_length

    if args_opt.do_train == "true":
        # run predistill
        run_predistill()
        lists = os.listdir(td_phase1_save_ckpt_dir)
        if lists:
            lists.sort(key=lambda fn: os.path.getmtime(td_phase1_save_ckpt_dir+'/'+fn))
            name_ext = os.path.splitext(lists[-1])
            if name_ext[-1] != ".ckpt":
                raise ValueError("Invalid file, checkpoint file should be .ckpt file")
            newest_ckpt_file = os.path.join(td_phase1_save_ckpt_dir, lists[-1])
            # run task distill
            run_task_distill(newest_ckpt_file)
        else:
            raise ValueError("Checkpoint file not exists, please make sure ckpt file has been saved")
    else:
        do_eval_standalone()
