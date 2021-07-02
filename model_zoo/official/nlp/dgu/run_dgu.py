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
Bert finetune and evaluation script.
'''

import os
import time
import mindspore.common.dtype as mstype
import mindspore.ops as P

from mindspore import context
from mindspore import log as logger
from mindspore.nn import Accuracy
from mindspore.nn.optim import AdamWeightDecay
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import (CheckpointConfig, ModelCheckpoint,
                                      TimeMonitor)
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

import src.dataset as data
import src.metric as metric
from src.args import parse_args, set_default_args
from src.bert_for_finetune import BertCLS, BertFinetuneCell
from src.finetune_eval_config import (bert_net_cfg, bert_net_udc_cfg,
                                      optimizer_cfg)
from src.utils import (CustomWarmUpLR, GetAllCkptPath, LossCallBack,
                       create_classification_dataset, make_directory)


def do_train(dataset=None, network=None, load_checkpoint_path="base-BertCLS-111.ckpt",
             save_checkpoint_path="", epoch_num=1):
    """ do train """
    if load_checkpoint_path == "":
        raise ValueError("Pretrain model missed, finetune task must load pretrain model!")
    print("load pretrain model: ", load_checkpoint_path)
    steps_per_epoch = args_opt.save_steps
    num_examples = dataset.get_dataset_size() * args_opt.train_batch_size
    max_train_steps = epoch_num * dataset.get_dataset_size()
    warmup_steps = int(max_train_steps * args_opt.warmup_proportion)
    print("Num train examples: %d" % num_examples)
    print("Max train steps: %d" % max_train_steps)
    print("Num warmup steps: %d" % warmup_steps)
    #warmup and optimizer
    lr_schedule = CustomWarmUpLR(learning_rate=args_opt.learning_rate, \
            warmup_steps=warmup_steps, max_train_steps=max_train_steps)
    params = network.trainable_params()
    decay_params = list(filter(optimizer_cfg.AdamWeightDecay.decay_filter, params))
    other_params = list(filter(lambda x: not optimizer_cfg.AdamWeightDecay.decay_filter(x), params))
    group_params = [{'params': decay_params, 'weight_decay': optimizer_cfg.AdamWeightDecay.weight_decay},
                    {'params': other_params, 'weight_decay': 0.0}]
    optimizer = AdamWeightDecay(group_params, lr_schedule, eps=optimizer_cfg.AdamWeightDecay.eps)
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2**32, scale_factor=2, scale_window=1000)
    #ckpt config
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=10)
    ckpoint_cb = ModelCheckpoint(prefix=args_opt.task_name,
                                 directory=None if save_checkpoint_path == "" else save_checkpoint_path,
                                 config=ckpt_config)
    # load checkpoint into network
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(network, param_dict)

    netwithgrads = BertFinetuneCell(network, optimizer=optimizer, scale_update_cell=update_cell)
    model = Model(netwithgrads)
    callbacks = [TimeMonitor(dataset.get_dataset_size()), LossCallBack(dataset.get_dataset_size()), ckpoint_cb]
    model.train(epoch_num, dataset, callbacks=callbacks)

def eval_result_print(eval_metric, result):
    if args_opt.task_name.lower() in ['atis_intent', 'mrda', 'swda']:
        metric_name = "Accuracy"
    else:
        metric_name = eval_metric.name()
    print(metric_name, " :", result)
    if args_opt.task_name.lower() == "udc":
        print("R1@10: ", result[0])
        print("R2@10: ", result[1])
        print("R5@10: ", result[2])

def do_eval(dataset=None, network=None, num_class=5, eval_metric=None, load_checkpoint_path=""):
    """ do eval """
    if load_checkpoint_path == "":
        raise ValueError("Finetune model missed, evaluation task must load finetune model!")
    print("eval model: ", load_checkpoint_path)
    print("loading... ")
    net_for_pretraining = network(eval_net_cfg, False, num_class)
    net_for_pretraining.set_train(False)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(net_for_pretraining, param_dict)
    model = Model(net_for_pretraining)

    print("evaling... ")
    columns_list = ["input_ids", "input_mask", "segment_ids", "label_ids"]
    eval_metric.clear()
    evaluate_times = []
    for data_item in dataset.create_dict_iterator(num_epochs=1):
        input_data = []
        for i in columns_list:
            input_data.append(data_item[i])
        input_ids, input_mask, token_type_id, label_ids = input_data
        squeeze = P.Squeeze(-1)
        label_ids = squeeze(label_ids)
        time_begin = time.time()
        logits = model.predict(input_ids, input_mask, token_type_id, label_ids)
        time_end = time.time()
        evaluate_times.append(time_end - time_begin)
        eval_metric.update(logits, label_ids)
    print("==============================================================")
    print("(w/o first and last) elapsed time: {}, per step time : {}".format(
        sum(evaluate_times[1:-1]), sum(evaluate_times[1:-1])/(len(evaluate_times) - 2)))
    print("==============================================================")
    result = eval_metric.eval()
    eval_result_print(eval_metric, result)
    return result


def run_dgu(args_input):
    """run_dgu main function """
    dataset_class, metric_class = TASK_CLASSES[args_input.task_name]
    epoch_num = args_input.epochs
    num_class = dataset_class.num_classes()

    target = args_input.device_target
    if target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args_input.device_id)
    elif target == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=args_input.device_id)
        if net_cfg.compute_type != mstype.float32:
            logger.warning('GPU only support fp32 temporarily, run with fp32.')
            net_cfg.compute_type = mstype.float32
    else:
        raise Exception("Target error, GPU or Ascend is supported.")

    if args_input.do_train.lower() == "true":
        netwithloss = BertCLS(net_cfg, True, num_labels=num_class, dropout_prob=0.1)
        train_ds = create_classification_dataset(batch_size=args_input.train_batch_size, repeat_count=1, \
                        data_file_path=args_input.train_data_file_path, \
                        do_shuffle=(args_input.train_data_shuffle.lower() == "true"), drop_remainder=True)
        do_train(train_ds, netwithloss, load_pretrain_checkpoint_path, save_finetune_checkpoint_path, epoch_num)

    if args_input.do_eval.lower() == "true":
        eval_ds = create_classification_dataset(batch_size=args_input.eval_batch_size, repeat_count=1, \
                    data_file_path=args_input.eval_data_file_path, \
                    do_shuffle=(args_input.eval_data_shuffle.lower() == "true"), drop_remainder=True)
        if args_input.task_name in ['atis_intent', 'mrda', 'swda']:
            eval_metric = metric_class("classification")
        else:
            eval_metric = metric_class()
        #load model from path and eval
        if args_input.eval_ckpt_path:
            do_eval(eval_ds, BertCLS, num_class, eval_metric, args_input.eval_ckpt_path)
        #eval all saved models
        else:
            ckpt_list = GetAllCkptPath(save_finetune_checkpoint_path)
            print("saved models:", ckpt_list)
            for filepath in ckpt_list:
                eval_result = do_eval(eval_ds, BertCLS, num_class, eval_metric, filepath)
                eval_file_dict[filepath] = str(eval_result)
            print(eval_file_dict)
        if args_input.is_modelarts_work == 'true':
            for filename in eval_file_dict:
                ckpt_result = eval_file_dict[filename].replace('[', '').replace(']', '').replace(', ', '_', 2)
                save_file_name = args_input.train_url + ckpt_result + "_" + filename.split('/')[-1]
                mox.file.copy_parallel(filename, save_file_name)
                print("upload model " + filename + " to " + save_file_name)

def print_args_input(args_input):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args_input).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')

def set_bert_cfg():
    """set bert cfg"""
    global net_cfg
    global eval_net_cfg
    if args_opt.task_name == 'udc':
        net_cfg = bert_net_udc_cfg
        eval_net_cfg = bert_net_udc_cfg
        print("use udc_bert_cfg")
    else:
        net_cfg = bert_net_cfg
        eval_net_cfg = bert_net_cfg
    return net_cfg, eval_net_cfg

if __name__ == '__main__':
    TASK_CLASSES = {
        'udc': (data.UDCv1, metric.RecallAtK),
        'atis_intent': (data.ATIS_DID, Accuracy),
        'mrda': (data.MRDA, Accuracy),
        'swda': (data.SwDA, Accuracy)
    }
    os.environ['GLOG_v'] = '3'
    eval_file_dict = {}
    args_opt = parse_args()
    set_default_args(args_opt)
    net_cfg, eval_net_cfg = set_bert_cfg()
    load_pretrain_checkpoint_path = args_opt.model_name_or_path
    save_finetune_checkpoint_path = args_opt.checkpoints_path + args_opt.task_name
    save_finetune_checkpoint_path = make_directory(save_finetune_checkpoint_path)
    if args_opt.is_modelarts_work == 'true':
        import moxing as mox
        local_load_pretrain_checkpoint_path = args_opt.local_model_name_or_path
        local_data_path = '/cache/data/' + args_opt.task_name
        mox.file.copy_parallel(args_opt.data_url + args_opt.task_name, local_data_path)
        mox.file.copy_parallel('obs:/' + load_pretrain_checkpoint_path, local_load_pretrain_checkpoint_path)
        load_pretrain_checkpoint_path = local_load_pretrain_checkpoint_path
        if not args_opt.train_data_file_path:
            args_opt.train_data_file_path = local_data_path + '/' + args_opt.task_name + '_train.mindrecord'
        if not args_opt.eval_data_file_path:
            args_opt.eval_data_file_path = local_data_path + '/' + args_opt.task_name + '_test.mindrecord'
    print_args_input(args_opt)
    run_dgu(args_opt)
