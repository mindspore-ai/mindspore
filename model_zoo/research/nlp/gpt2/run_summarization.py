# -*- coding: utf-8 -*-
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
"""
GPT-2 finetune and evaluation script for Summarization task.
"""

import time
import argparse

from mindspore import context
from mindspore.nn import AdamWeightDecay, Lamb, Momentum
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.model import Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor, LossMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.GPT2ForSummarization import GPT2SummarizationModel
from src.gpt2_for_finetune import GPT2Summarization, GPT2FinetuneCell
from src.finetune_eval_config import cfg, gpt2_net_cfg
from src.utils.metric_method import Rouge
from src.dataset import create_language_model_dataset
from src.utils.lr_schedule import GPT2LearningRate
from src.utils.tokenization import Tokenizer
from src.utils.task_utils import clean_hypo, modify_paramdict
from src.GPT2_generation import GenerateForSummarization
from src.utils.get_config_setting import get_train_setting, get_model_setting


def do_train(dataset=None, network=None, load_checkpoint_path="", save_checkpoint_path="", epoch_num=1):
    """
    Do train
    Args:
        dataset: the train dataset.
        network:  the network with loss
        load_checkpoint_path: the file path which saved pretrain model checkpoint.
        save_checkpoint_path:  the file path which will save finetune model checkpoint.
        epoch_num: the number of epoch
    """
    if load_checkpoint_path == "":
        raise ValueError("Pretrain model missed, finetune task must load pretrain model!")

    steps_per_epoch = dataset.get_dataset_size()

    # optimizer
    if cfg.optimizer == 'AdamWeightDecay':
        lr_schedule = GPT2LearningRate(learning_rate=cfg.AdamWeightDecay.learning_rate,
                                       end_learning_rate=cfg.AdamWeightDecay.end_learning_rate,
                                       warmup_steps=int(steps_per_epoch * epoch_num * 0.1),
                                       decay_steps=steps_per_epoch * epoch_num,
                                       power=cfg.AdamWeightDecay.power)
        params = network.trainable_params()

        decay_params = list(filter(cfg.AdamWeightDecay.decay_filter, params))
        other_params = list(
            filter(lambda x: not cfg.AdamWeightDecay.decay_filter(x), params))
        group_params = [{'params': decay_params, 'weight_decay': cfg.AdamWeightDecay.weight_decay},
                        {'params': other_params, 'weight_decay': 0.0}]
        optimizer = AdamWeightDecay(group_params, lr_schedule, eps=cfg.AdamWeightDecay.eps)
    elif cfg.optimizer == 'Lamb':
        lr_schedule = GPT2LearningRate(learning_rate=cfg.Lamb.learning_rate,
                                       end_learning_rate=cfg.Lamb.end_learning_rate,
                                       warmup_steps=int(steps_per_epoch * epoch_num * 0.1),
                                       decay_steps=steps_per_epoch * epoch_num,
                                       power=cfg.Lamb.power)
        optimizer = Lamb(network.trainable_params(), lr_schedule)
    elif cfg.optimizer == 'Momentum':
        optimizer = Momentum(network.trainable_params(), cfg.Momentum.learning_rate, cfg.Momentum.momentum)
    else:
        raise Exception("Optimizer not supported. support: [AdamWeightDecay, Lamb, Momentum]")

    # load checkpoint into network
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=1)
    prefix_name = "gpt2_summarization_" + str(cfg.gpt2_network) + "_" + str(cfg.optimizer) + "_" \
                  + str(epoch_num) + "_bs" + str(gpt2_net_cfg.batch_size)
    ckpoint_cb = ModelCheckpoint(prefix=prefix_name,
                                 directory=None if save_checkpoint_path == "" else save_checkpoint_path,
                                 config=ckpt_config)
    param_dict = load_checkpoint(load_checkpoint_path)

    final_param_dict = {}
    for name, _ in param_dict.items():
        final_param_dict['gpt2.gpt2.' + name] = param_dict[name]
    final_param_dict['gpt2.lm_head.weight'] = param_dict['gpt2_embedding_lookup.embedding_table']

    load_param_into_net(network, final_param_dict)
    print("Load pretrained parameter successfully!\n")

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2 ** 32, scale_factor=2, scale_window=1000)
    netwithgrads = GPT2FinetuneCell(network, optimizer=optimizer, scale_update_cell=update_cell)
    netwithgrads.set_train(True)

    loss_cb = LossMonitor(per_print_times=1)
    model = Model(netwithgrads)
    callbacks = [TimeMonitor(dataset.get_dataset_size()), loss_cb, ckpoint_cb]

    print("============== Starting Finetuning ==============")
    model.train(epoch_num, dataset, callbacks=callbacks, dataset_sink_mode=False)
    print("============== Finetuning Success ==============")


def eval_result_print(metric="Rouge", callback=None):
    """
    print eval result
    """
    if metric == "Rouge":
        print("Rouge-1 {:.8f}, Rouge-2 {:.8f}, Rouge-L {:.8f}, Rouge-AVG{:.8f}".
              format(callback.Rouge1 / callback.total_num,
                     callback.Rouge2 / callback.total_num,
                     callback.RougeL / callback.total_num,
                     (callback.Rouge1 + callback.Rouge2 + callback.RougeL) / (3.0 * callback.total_num)))
    else:
        raise ValueError("metric method '{}' not supported, support: [Rouge]. ".format(str(metric)))


def do_eval(dataset=None, network=None, metric=None, load_checkpoint_path="", eval_type=None, tokenizer_file="",
            top_k=None, top_p=None, temperature=None, generate_length=None):
    """
    Do evaluation on summarization
    """
    if load_checkpoint_path == "":
        raise ValueError("Finetune model missed, evaluation task must load finetune model!")
    if metric.lower() == "rouge":
        print("Prepare to calculate the Rouge score ...")
        callback = Rouge()

        gpt2_loss = network(config=gpt2_net_cfg,
                            is_training=False,
                            use_one_hot_embeddings=False)
        gpt2_loss.set_train(False)
        param_dict = load_checkpoint(load_checkpoint_path)

        reorganized_param_dict = modify_paramdict(param_dict, mode=eval_type, model_prefix="gpt2.")
        load_param_into_net(gpt2_loss, reorganized_param_dict)

        # load nn.Cell into Model and initiate tokenizer and Sample
        model = Model(gpt2_loss)
        tokenizer = Tokenizer(vocab_file=tokenizer_file + 'gpt2-vocab.json',
                              merge_file=tokenizer_file + 'gpt2-merges.txt')

        # load data and process text generation
        columns_list = ["input_ids", "input_mask", "label_ids"]

        summarization_generator = GenerateForSummarization(model,
                                                           config=gpt2_net_cfg,
                                                           tokenizer=tokenizer,
                                                           select_sentence=3,
                                                           eval_type=eval_type,
                                                           topk=top_k,
                                                           topp=float(top_p),
                                                           temperature=float(temperature),
                                                           generate_length=generate_length)
        num_data = 1
        print("==================== [Summrization] Testing ====================")
        for data in dataset.create_dict_iterator():
            input_data = []
            for value in columns_list:
                input_data.append(data[value])
            input_ids, _, label_ids = input_data
            print(" | [ROUGE] number : {} / {} ".format(num_data, dataset.get_dataset_size()))
            print("input_ids shape: {}".format(input_ids.shape))
            print("label_ids shape: {}".format(label_ids.shape))

            hypothesis, ref = summarization_generator.generate_for_summarization(input_ids)
            if ref[0] == '' or ref[0] is None:
                print("Sorry ref_list is None, skip it!")
                continue

            print("REF str:\n ", ref, "\nHYPO str:\n", hypothesis, "\n")
            for batch_idx in range(gpt2_net_cfg.batch_size):
                hypothesis[batch_idx] = clean_hypo(hypothesis[batch_idx])
            for batch_idx in range(gpt2_net_cfg.batch_size):
                hypothesis[batch_idx] = hypothesis[batch_idx].lower()
                ref[batch_idx] = ref[batch_idx].lower()

            callback.update(hypothesis, ref)
            num_data += 1

        print("\n\n")
        print("**********************************************************")
        eval_result_print(metric, callback)
        print("******************** Testing Finished ********************")

    else:
        raise ValueError("metric method not supported in summarization, support: [Rouge]")


def run_summarization():
    """
    Run Summarization task.
    """
    # set argument parser
    parser = argparse.ArgumentParser(description="Finetune and Evaluate Summrization")

    # context and task settings
    parser.add_argument("--device_target", type=str, default="Ascend",
                        help="Device type. Default: Ascend.")
    parser.add_argument("--device_id", type=int, default=4,
                        help="ID of target device.")
    parser.add_argument("--do_train", type=str, default="false",
                        help="Enable train. Default: false.")
    parser.add_argument("--do_eval", type=str, default="true",
                        help="Enable evaluation. Default: false.")
    parser.add_argument("--eval_type", type=str, default="finetuned",
                        help="The type of evaluation including [zero-shot, finetuned]. Default: zero-shot.")
    parser.add_argument("--metric_method", type=str, default="Rouge",
                        help="The eval method including [Rouge(Rouge1,Rouge2,RougeL,Rouge Avg)]. Default: Rouge.")
    parser.add_argument("--epoch_num", type=int, default=2,
                        help="Epoch number. Default: 2.")

    # dataset and params_dict file settings
    parser.add_argument("--train_data_shuffle", type=str, default="true",
                        help="Enable train data shuffle. Default: true.")
    parser.add_argument("--eval_data_shuffle", type=str, default="false",
                        help="Enable eval data shuffle. Default: false.")
    parser.add_argument("--save_finetune_ckpt_path", type=str, default="",
                        help="Save the checkpoint path.")
    parser.add_argument("--load_pretrain_ckpt_path", type=str, default="",
                        help="Load the checkpoint file path.")
    parser.add_argument("--load_finetune_ckpt_path", type=str, default="",
                        help="Load the checkpoint file path.")
    parser.add_argument("--train_data_file_path", type=str, default="",
                        help="Data path, it is better to use absolute path")
    parser.add_argument("--eval_data_file_path", type=str, default="",
                        help="Data path, it is better to use absolute path")

    # sampling settings
    parser.add_argument("--top_k", type=int, default=2,
                        help="top k tokens chosen for sampling")
    parser.add_argument("--top_p", type=str, default="1.0",
                        help="top p accumulated probability threshold for logit to be counted")
    parser.add_argument("--generate_length", type=int, default=100,
                        help="the number of generated tokens.")
    parser.add_argument("--temperature", type=str, default="1.0",
                        help="temperature on logits for sampling")
    parser.add_argument("--tokenizer_file_path", type=str, default="",
                        help="vocab & merge file path")
    args_opt = parser.parse_args()

    epoch_num = args_opt.epoch_num
    metric = args_opt.metric_method
    save_finetune_ckpt_path = args_opt.save_finetune_ckpt_path
    load_finetune_ckpt_path = args_opt.load_finetune_ckpt_path
    load_pretrain_ckpt_path = args_opt.load_pretrain_ckpt_path
    eval_type = args_opt.eval_type
    tokenizer_file = args_opt.tokenizer_file_path

    if args_opt.do_train.lower() == "false" and args_opt.do_eval.lower() == "false":
        raise ValueError("At least one of 'do_train' or 'do_eval' must be true")
    if args_opt.do_train.lower() == "true" and args_opt.train_data_file_path == "":
        raise ValueError("'train_data_file_path' must be set when do finetune task")
    if args_opt.do_eval.lower() == "true" and args_opt.eval_data_file_path == "":
        raise ValueError("'eval_data_file_path' must be set when do evaluation task")

    device = args_opt.device_target
    if device == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args_opt.device_id)
        context.set_auto_parallel_context(parallel_mode="stand_alone")
        print(" | Device: {}  | Device id: {}".format(device, args_opt.device_id))
    else:
        raise Exception("Device target error, Ascend is supported.")

    if args_opt.do_train.lower() == "true":
        get_train_setting(cfg)
        get_model_setting(cfg, gpt2_net_cfg)
        train_data_file_path = args_opt.train_data_file_path
        gpt2_loss = GPT2Summarization(config=gpt2_net_cfg,
                                      is_training=True,
                                      use_one_hot_embeddings=False)
        print("==============    Start Loading Train Dataset   ============")
        train_dataset = create_language_model_dataset(do_shuffle=(args_opt.train_data_shuffle.lower() == "true"),
                                                      dataset_path=train_data_file_path)
        do_train(train_dataset, gpt2_loss, load_pretrain_ckpt_path, save_finetune_ckpt_path, epoch_num)

    if args_opt.do_eval.lower() == "true":
        get_model_setting(cfg, gpt2_net_cfg)
        eval_dataset_file_path = args_opt.eval_data_file_path
        print("============== Start Loading Evaluation Dataset ============")
        eval_dataset = create_language_model_dataset(do_shuffle=(args_opt.train_data_shuffle.lower() == "true"),
                                                     dataset_path=eval_dataset_file_path)
        do_eval(eval_dataset, GPT2SummarizationModel, metric, load_finetune_ckpt_path, eval_type, tokenizer_file,
                args_opt.top_k, args_opt.top_p, args_opt.temperature, args_opt.generate_length)


if __name__ == "__main__":
    print("Start Time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    run_summarization()
    print("End Time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
