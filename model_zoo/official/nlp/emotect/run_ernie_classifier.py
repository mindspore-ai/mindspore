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
Ernie finetune and evaluation script.
'''

import os
import time
import argparse
from src.ernie_for_finetune import ErnieFinetuneCell, ErnieCLS
from src.finetune_eval_config import optimizer_cfg, ernie_net_cfg
from src.dataset import create_classification_dataset
from src.assessment_method import Accuracy
from src.utils import make_directory, LossCallBack, LoadNewestCkpt, ErnieLearningRate
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore import log as logger
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.optim import Adam, AdamWeightDecay, Adagrad
from mindspore.train.model import Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

_cur_dir = os.getcwd()

def do_train(dataset=None, network=None, load_checkpoint_path="", save_checkpoint_path="", epoch_num=1):
    """ do train """
    if load_checkpoint_path == "":
        raise ValueError("Pretrain model missed, finetune task must load pretrain model!")
    steps_per_epoch = 500
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
    # load checkpoint into network
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=10)
    ckpoint_cb = ModelCheckpoint(prefix="classifier",
                                 directory=None if save_checkpoint_path == "" else save_checkpoint_path,
                                 config=ckpt_config)
    param_dict = load_checkpoint(load_checkpoint_path)
    unloaded_params = load_param_into_net(network, param_dict)
    if len(unloaded_params) > 2:
        print(unloaded_params)
        logger.warning('Loading ernie model failed, please check the checkpoint file.')

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2**32, scale_factor=2, scale_window=1000)
    netwithgrads = ErnieFinetuneCell(network, optimizer=optimizer, scale_update_cell=update_cell)
    model = Model(netwithgrads)
    callbacks = [TimeMonitor(dataset.get_dataset_size()), LossCallBack(dataset.get_dataset_size()), ckpoint_cb]
    model.train(epoch_num, dataset, callbacks=callbacks)

def do_eval(dataset=None, network=None, num_class=2, load_checkpoint_path=""):
    """ do eval """
    if load_checkpoint_path == "":
        raise ValueError("Finetune model missed, evaluation task must load finetune model!")
    net_for_pretraining = network(ernie_net_cfg, False, num_class)
    net_for_pretraining.set_train(False)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(net_for_pretraining, param_dict)

    callback = Accuracy()

    evaluate_times = []
    columns_list = ["input_ids", "input_mask", "segment_ids", "label_ids"]
    for data in dataset.create_dict_iterator(num_epochs=1):
        input_data = []
        for i in columns_list:
            input_data.append(data[i])
        input_ids, input_mask, token_type_id, label_ids = input_data
        time_begin = time.time()
        logits = net_for_pretraining(input_ids, input_mask, token_type_id, label_ids)
        time_end = time.time()
        evaluate_times.append(time_end - time_begin)
        callback.update(logits, label_ids)
    print("==============================================================")
    print("acc_num {} , total_num {}, accuracy {:.6f}".format(callback.acc_num, callback.total_num,
                                                              callback.acc_num / callback.total_num))
    print("(w/o first and last) elapsed time: {}, per step time : {}".format(
        sum(evaluate_times[1:-1]), sum(evaluate_times[1:-1])/(len(evaluate_times) - 2)))
    print("==============================================================")

def run_classifier():
    """run classifier task"""
    parser = argparse.ArgumentParser(description="run classifier")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["Ascend", "GPU"],
                        help="Device type, default is Ascend")
    parser.add_argument("--do_train", type=str, default="false", choices=["true", "false"],
                        help="Enable train, default is false")
    parser.add_argument("--do_eval", type=str, default="false", choices=["true", "false"],
                        help="Enable eval, default is false")
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--epoch_num", type=int, default=3, help="Epoch number, default is 3.")
    parser.add_argument("--num_class", type=int, default=3, help="The number of class, default is 3.")
    parser.add_argument("--train_data_shuffle", type=str, default="true", choices=["true", "false"],
                        help="Enable train data shuffle, default is true")
    parser.add_argument("--eval_data_shuffle", type=str, default="false", choices=["true", "false"],
                        help="Enable eval data shuffle, default is false")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Train batch size, default is 32")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Eval batch size, default is 1")
    parser.add_argument("--save_finetune_checkpoint_path", type=str, default="", help="Save checkpoint path")
    parser.add_argument("--load_pretrain_checkpoint_path", type=str, default="", help="Load checkpoint file path")
    parser.add_argument("--local_pretrain_checkpoint_path", type=str, default="",
                        help="Local pretrain checkpoint file path")
    parser.add_argument("--load_finetune_checkpoint_path", type=str, default="", help="Load checkpoint file path")
    parser.add_argument("--train_data_file_path", type=str, default="",
                        help="Data path, it is better to use absolute path")
    parser.add_argument("--eval_data_file_path", type=str, default="",
                        help="Data path, it is better to use absolute path")
    parser.add_argument("--schema_file_path", type=str, default="",
                        help="Schema path, it is better to use absolute path")
    parser.add_argument('--data_url', type=str, default=None, help='Dataset path for ModelArts')
    parser.add_argument('--train_url', type=str, default=None, help='Train output path for ModelArts')
    parser.add_argument('--modelarts', type=str, default='false',
                        help='train on modelarts or not, default is false')
    args_opt = parser.parse_args()

    epoch_num = args_opt.epoch_num
    load_pretrain_checkpoint_path = args_opt.load_pretrain_checkpoint_path
    save_finetune_checkpoint_path = args_opt.save_finetune_checkpoint_path
    load_finetune_checkpoint_path = args_opt.load_finetune_checkpoint_path

    if args_opt.modelarts.lower() == 'true':
        import moxing as mox
        mox.file.copy_parallel(args_opt.data_url, '/cache/data')
        mox.file.copy_parallel(args_opt.load_pretrain_checkpoint_path, args_opt.local_pretrain_checkpoint_path)
        load_pretrain_checkpoint_path = args_opt.local_pretrain_checkpoint_path
        if args_opt.do_train.lower() == "false" and args_opt.do_eval.lower() == "true":
            mox.file.copy_parallel(args_opt.save_finetune_checkpoint_path, args_opt.load_finetune_checkpoint_path)

    if args_opt.do_train.lower() == "false" and args_opt.do_eval.lower() == "false":
        raise ValueError("At least one of 'do_train' or 'do_eval' must be true")
    if args_opt.do_train.lower() == "true" and args_opt.train_data_file_path == "":
        raise ValueError("'train_data_file_path' must be set when do finetune task")
    if args_opt.do_eval.lower() == "true" and args_opt.eval_data_file_path == "":
        raise ValueError("'eval_data_file_path' must be set when do evaluation task")

    target = args_opt.device_target
    if target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args_opt.device_id)
    elif target == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
        if ernie_net_cfg.compute_type != mstype.float32:
            logger.warning('GPU only support fp32 temporarily, run with fp32.')
            ernie_net_cfg.compute_type = mstype.float32
    else:
        raise Exception("Target error, GPU or Ascend is supported.")

    netwithloss = ErnieCLS(ernie_net_cfg, True, num_labels=args_opt.num_class, dropout_prob=0.1)

    if args_opt.do_train.lower() == "true":
        ds = create_classification_dataset(batch_size=args_opt.train_batch_size, repeat_count=1,
                                           data_file_path=args_opt.train_data_file_path,
                                           schema_file_path=args_opt.schema_file_path,
                                           do_shuffle=(args_opt.train_data_shuffle.lower() == "true"))
        do_train(ds, netwithloss, load_pretrain_checkpoint_path, save_finetune_checkpoint_path, epoch_num)

        if args_opt.do_eval.lower() == "true":
            if save_finetune_checkpoint_path == "":
                load_finetune_checkpoint_dir = _cur_dir
            else:
                load_finetune_checkpoint_dir = make_directory(save_finetune_checkpoint_path)
            load_finetune_checkpoint_path = LoadNewestCkpt(load_finetune_checkpoint_dir,
                                                           ds.get_dataset_size(), epoch_num, "classifier")

    if args_opt.do_eval.lower() == "true":
        ds = create_classification_dataset(batch_size=args_opt.eval_batch_size, repeat_count=1,
                                           data_file_path=args_opt.eval_data_file_path,
                                           schema_file_path=args_opt.schema_file_path,
                                           do_shuffle=(args_opt.eval_data_shuffle.lower() == "true"),
                                           drop_remainder=False)
        do_eval(ds, ErnieCLS, args_opt.num_class, load_finetune_checkpoint_path)

    if args_opt.modelarts.lower() == 'true' and args_opt.do_train.lower() == "true":
        mox.file.copy_parallel(save_finetune_checkpoint_path, args_opt.train_url)
if __name__ == "__main__":
    run_classifier()
