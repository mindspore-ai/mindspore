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
# ===========================================================================

"""q8bert train"""

import argparse
import os
import ast

from mindspore import context
from mindspore import set_seed
from mindspore.nn.optim import AdamWeightDecay
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import TimeMonitor
from mindspore.train.model import Model
import mindspore.common.dtype as mstype

from src.dataset import create_dataset
from src.q8bert import BertEvaluationWithLossScaleCell, BertNetworkWithLoss_td, BertEvaluationCell
from src.config import train_cfg, eval_cfg, model_cfg, glue_output_modes, task_params
from src.utils import LossCallBack, ModelSaveCkpt, EvalCallBack, BertLearningRate


def parse_args():
    """
    parse args
    """
    parser = argparse.ArgumentParser(description="Q8Bert task train")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["Ascend", "GPU"],
                        help="device where the code will be implemented. (Default: Ascend)")
    parser.add_argument("--do_eval", type=ast.literal_eval, default=True, help="Do eval task, default is True.")
    parser.add_argument("--epoch_num", type=int, default=3, help="default is 3.")
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--do_shuffle", type=ast.literal_eval, default=True,
                        help="Enable shuffle for dataset, default is True.")
    parser.add_argument("--enable_data_sink", type=ast.literal_eval, default=True,
                        help="Enable data sink, default is True.")
    parser.add_argument("--save_ckpt_step", type=int, default=100, help="Enable save ckpt.")
    parser.add_argument("--max_ckpt_num", type=int, default=1, help="Number of saved pretrained models, default is 1.")
    parser.add_argument("--data_sink_steps", type=int, default=1, help="Sink steps for each epoch, default is 1.")
    parser.add_argument("--load_ckpt_path", type=str, default="", help="Load checkpoint file path")
    parser.add_argument("--train_data_dir", type=str, default="",
                        help="Train data path, it is better to use absolute path")
    parser.add_argument("--eval_data_dir", type=str, default="",
                        help="Eval data path, it is better to use absolute path")
    parser.add_argument("--do_quant", type=ast.literal_eval, default=True, help="Do quant for model")
    parser.add_argument("--logging_step", type=int, default=100, help="Do evalate each logging step")
    parser.add_argument("--task_name", type=str, default="STS-B", choices=["STS-B", "QNLI", "SST-2"],
                        help="The name of the task to train.")
    parser.add_argument("--dataset_type", type=str, default="tfrecord",
                        help="dataset type tfrecord/mindrecord, default is tfrecord")
    args = parser.parse_args()
    return args


args_opt = parse_args()
_cur_dir = os.getcwd()
save_ckpt_dir = os.path.join(_cur_dir, "Q8Bert_" + args_opt.task_name + "_model")
if not os.path.exists(save_ckpt_dir):
    os.makedirs(save_ckpt_dir)

DEFAULT_NUM_LABELS = 2
DEFAULT_SEQ_LENGTH = 128


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


def do_train():
    """
    do train
    """
    ckpt_file = args_opt.load_ckpt_path

    if not ckpt_file:
        raise ValueError("Student ckpt file should not be None")
    cfg = train_cfg

    if args_opt.device_target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, device_id=args_opt.device_id)
        model_cfg.compute_type = mstype.float16
        train_cfg.loss_scale_value = 2 ** 10
    elif args_opt.device_target == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    else:
        raise Exception("Target error, GPU or Ascend is supported.")

    load_student_checkpoint_path = ckpt_file
    netwithloss = BertNetworkWithLoss_td(student_config=model_cfg, student_ckpt=load_student_checkpoint_path,
                                         do_quant=args_opt.do_quant, is_training=True,
                                         task_type=glue_output_modes[args_opt.task_name.lower()],
                                         num_labels=task.num_labels, is_predistill=False)
    rank = 0
    device_num = 1
    train_dataset = create_dataset(cfg.batch_size,
                                   device_num, rank, args_opt.do_shuffle,
                                   args_opt.train_data_dir,
                                   data_type=args_opt.dataset_type,
                                   seq_length=task.seq_length,
                                   drop_remainder=True)

    dataset_size = train_dataset.get_dataset_size()
    print("train dataset size: ", dataset_size)
    if args_opt.enable_data_sink:
        repeat_count = args_opt.epoch_num * train_dataset.get_dataset_size() // args_opt.data_sink_steps
        time_monitor_steps = args_opt.data_sink_steps
    else:
        repeat_count = args_opt.epoch_num
        time_monitor_steps = dataset_size

    optimizer_cfg = cfg.optimizer_cfg

    lr_schedule = BertLearningRate(learning_rate=optimizer_cfg.AdamWeightDecay.learning_rate,
                                   end_learning_rate=optimizer_cfg.AdamWeightDecay.end_learning_rate,
                                   warmup_steps=int(dataset_size * args_opt.epoch_num / 10),
                                   decay_steps=int(dataset_size * args_opt.epoch_num),
                                   power=optimizer_cfg.AdamWeightDecay.power)
    params = netwithloss.trainable_params()
    decay_params = list(filter(optimizer_cfg.AdamWeightDecay.decay_filter, params))
    other_params = list(filter(lambda x: not optimizer_cfg.AdamWeightDecay.decay_filter(x), params))
    group_params = [{"params": decay_params, "weight_decay": optimizer_cfg.AdamWeightDecay.weight_decay},
                    {"params": other_params, "weight_decay": 0.0},
                    {"order_params": params}]

    optimizer = AdamWeightDecay(group_params, learning_rate=lr_schedule, eps=optimizer_cfg.AdamWeightDecay.eps)

    eval_dataset = create_dataset(eval_cfg.batch_size,
                                  device_num, rank, args_opt.do_shuffle,
                                  args_opt.eval_data_dir,
                                  data_type=args_opt.dataset_type,
                                  seq_length=task.seq_length,
                                  drop_remainder=False)
    print("eval dataset size: ", eval_dataset.get_dataset_size())

    if args_opt.do_eval:
        callback = [TimeMonitor(time_monitor_steps), LossCallBack(),
                    EvalCallBack(netwithloss.bert, eval_dataset, args_opt.task_name, args_opt.logging_step,
                                 save_ckpt_dir)]
    else:
        callback = [TimeMonitor(time_monitor_steps), LossCallBack(),
                    ModelSaveCkpt(netwithloss.bert,
                                  args_opt.save_ckpt_step,
                                  args_opt.max_ckpt_num,
                                  save_ckpt_dir)]
    if enable_loss_scale:
        update_cell = DynamicLossScaleUpdateCell(loss_scale_value=cfg.loss_scale_value,
                                                 scale_factor=cfg.scale_factor,
                                                 scale_window=cfg.scale_window)
        netwithgrads = BertEvaluationWithLossScaleCell(netwithloss, optimizer=optimizer, scale_update_cell=update_cell)
    else:
        netwithgrads = BertEvaluationCell(netwithloss, optimizer=optimizer)
    model = Model(netwithgrads)
    model.train(repeat_count, train_dataset, callbacks=callback,
                dataset_sink_mode=args_opt.enable_data_sink,
                sink_size=args_opt.data_sink_steps)


if __name__ == "__main__":
    set_seed(1)
    enable_loss_scale = True
    model_cfg.seq_length = task.seq_length
    do_train()
