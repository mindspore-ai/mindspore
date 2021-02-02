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

"""task distill script"""

import os
import argparse
from mindspore import context
from mindspore.train.model import Model
from mindspore.nn.optim import AdamWeightDecay
from mindspore import set_seed
from src.dataset import create_dataset
from src.utils import StepCallBack, ModelSaveCkpt, EvalCallBack, BertLearningRate
from src.config import train_cfg, eval_cfg, teacher_net_cfg, student_net_cfg, task_cfg
from src.cell_wrapper import BertNetworkWithLoss, BertTrainCell

WEIGHTS_NAME = 'eval_model.ckpt'
EVAL_DATA_NAME = 'eval.tf_record'
TRAIN_DATA_NAME = 'train.tf_record'


def parse_args():
    """
    parse args
    """
    parser = argparse.ArgumentParser(description='ternarybert task distill')
    parser.add_argument('--device_target', type=str, default='GPU', choices=['Ascend', 'GPU'],
                        help='Device where the code will be implemented. (Default: GPU)')
    parser.add_argument('--do_eval', type=str, default='true', choices=['true', 'false'],
                        help='Do eval task during training or not. (Default: true)')
    parser.add_argument('--epoch_size', type=int, default=3, help='Epoch size for train phase. (Default: 3)')
    parser.add_argument('--device_id', type=int, default=0, help='Device id. (Default: 0)')
    parser.add_argument('--do_shuffle', type=str, default='true', choices=['true', 'false'],
                        help='Enable shuffle for train dataset. (Default: true)')
    parser.add_argument('--enable_data_sink', type=str, default='true', choices=['true', 'false'],
                        help='Enable data sink. (Default: true)')
    parser.add_argument('--save_ckpt_step', type=int, default=50,
                        help='If do_eval is false, the checkpoint will be saved every save_ckpt_step. (Default: 50)')
    parser.add_argument('--eval_ckpt_step', type=int, default=50,
                        help='If do_eval is true, the evaluation will be ran every eval_ckpt_step. (Default: 50)')
    parser.add_argument('--max_ckpt_num', type=int, default=10,
                        help='The number of checkpoints will not be larger than max_ckpt_num. (Default: 10)')
    parser.add_argument('--data_sink_steps', type=int, default=1, help='Sink steps for each epoch. (Default: 1)')
    parser.add_argument('--teacher_model_dir', type=str, default='', help='The checkpoint directory of teacher model.')
    parser.add_argument('--student_model_dir', type=str, default='', help='The checkpoint directory of student model.')
    parser.add_argument('--data_dir', type=str, default='', help='Data directory.')
    parser.add_argument('--output_dir', type=str, default='./', help='The output checkpoint directory.')
    parser.add_argument('--task_name', type=str, default='sts-b', choices=['sts-b', 'qnli', 'mnli'],
                        help='The name of the task to train. (Default: sts-b)')
    parser.add_argument('--dataset_type', type=str, default='tfrecord', choices=['tfrecord', 'mindrecord'],
                        help='The name of the task to train. (Default: tfrecord)')
    parser.add_argument('--seed', type=int, default=1, help='The random seed')
    parser.add_argument('--train_batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Eval Batch size in callback')
    return parser.parse_args()


def run_task_distill(args_opt):
    """
    run task distill
    """
    task = task_cfg[args_opt.task_name]
    teacher_net_cfg.seq_length = task.seq_length
    student_net_cfg.seq_length = task.seq_length
    train_cfg.batch_size = args_opt.train_batch_size
    eval_cfg.batch_size = args_opt.eval_batch_size
    teacher_ckpt = os.path.join(args_opt.teacher_model_dir, args_opt.task_name, WEIGHTS_NAME)
    student_ckpt = os.path.join(args_opt.student_model_dir, args_opt.task_name, WEIGHTS_NAME)
    train_data_dir = os.path.join(args_opt.data_dir, args_opt.task_name, TRAIN_DATA_NAME)
    eval_data_dir = os.path.join(args_opt.data_dir, args_opt.task_name, EVAL_DATA_NAME)
    save_ckpt_dir = os.path.join(args_opt.output_dir, args_opt.task_name)

    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, device_id=args.device_id)

    rank = 0
    device_num = 1
    train_dataset = create_dataset(batch_size=train_cfg.batch_size,
                                   device_num=device_num,
                                   rank=rank,
                                   do_shuffle=args_opt.do_shuffle,
                                   data_dir=train_data_dir,
                                   data_type=args_opt.dataset_type,
                                   seq_length=task.seq_length,
                                   task_type=task.task_type,
                                   drop_remainder=True)
    dataset_size = train_dataset.get_dataset_size()
    print('train dataset size:', dataset_size)
    eval_dataset = create_dataset(batch_size=eval_cfg.batch_size,
                                  device_num=device_num,
                                  rank=rank,
                                  do_shuffle=args_opt.do_shuffle,
                                  data_dir=eval_data_dir,
                                  data_type=args_opt.dataset_type,
                                  seq_length=task.seq_length,
                                  task_type=task.task_type,
                                  drop_remainder=False)
    print('eval dataset size:', eval_dataset.get_dataset_size())

    if args_opt.enable_data_sink == 'true':
        repeat_count = args_opt.epoch_size * dataset_size // args_opt.data_sink_steps
    else:
        repeat_count = args_opt.epoch_size

    netwithloss = BertNetworkWithLoss(teacher_config=teacher_net_cfg, teacher_ckpt=teacher_ckpt,
                                      student_config=student_net_cfg, student_ckpt=student_ckpt,
                                      is_training=True, task_type=task.task_type, num_labels=task.num_labels)
    params = netwithloss.trainable_params()
    optimizer_cfg = train_cfg.optimizer_cfg
    lr_schedule = BertLearningRate(learning_rate=optimizer_cfg.AdamWeightDecay.learning_rate,
                                   end_learning_rate=optimizer_cfg.AdamWeightDecay.end_learning_rate,
                                   warmup_steps=int(dataset_size * args_opt.epoch_size *
                                                    optimizer_cfg.AdamWeightDecay.warmup_ratio),
                                   decay_steps=int(dataset_size * args_opt.epoch_size),
                                   power=optimizer_cfg.AdamWeightDecay.power)
    decay_params = list(filter(optimizer_cfg.AdamWeightDecay.decay_filter, params))
    other_params = list(filter(lambda x: not optimizer_cfg.AdamWeightDecay.decay_filter(x), params))
    group_params = [{'params': decay_params, 'weight_decay': optimizer_cfg.AdamWeightDecay.weight_decay},
                    {'params': other_params, 'weight_decay': 0.0},
                    {'order_params': params}]

    optimizer = AdamWeightDecay(group_params, learning_rate=lr_schedule, eps=optimizer_cfg.AdamWeightDecay.eps)

    netwithgrads = BertTrainCell(netwithloss, optimizer=optimizer)

    if args_opt.do_eval == 'true':
        eval_dataset = list(eval_dataset.create_dict_iterator())
        callback = [EvalCallBack(network=netwithloss.bert,
                                 dataset=eval_dataset,
                                 eval_ckpt_step=args_opt.eval_ckpt_step,
                                 save_ckpt_dir=save_ckpt_dir,
                                 embedding_bits=student_net_cfg.embedding_bits,
                                 weight_bits=student_net_cfg.weight_bits,
                                 clip_value=student_net_cfg.weight_clip_value,
                                 metrics=task.metrics)]
    else:
        callback = [StepCallBack(),
                    ModelSaveCkpt(network=netwithloss.bert,
                                  save_ckpt_step=args_opt.save_ckpt_step,
                                  max_ckpt_num=args_opt.max_ckpt_num,
                                  output_dir=save_ckpt_dir,
                                  embedding_bits=student_net_cfg.embedding_bits,
                                  weight_bits=student_net_cfg.weight_bits,
                                  clip_value=student_net_cfg.weight_clip_value)]
    model = Model(netwithgrads)
    model.train(repeat_count, train_dataset, callbacks=callback,
                dataset_sink_mode=(args_opt.enable_data_sink == 'true'),
                sink_size=args_opt.data_sink_steps)


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    run_task_distill(args)
