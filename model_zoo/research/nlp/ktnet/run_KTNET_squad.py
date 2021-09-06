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
"""Finetuning on SQuAD."""

import argparse
import collections
import os
import logging
import ast

from src.KTNET import KTNET
from src.dataset import create_train_dataset
from utils.util import CustomWarmUpLR
from utils.args import ArgumentGroup

from mindspore import context
from mindspore.train.model import Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor, LossMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn.optim import Adam
from mindspore.nn.wrap import TrainOneStepWithLossScaleCell
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.communication.management import init
from mindspore.context import ParallelMode

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

_cur_dir = os.getcwd()
bert_config = {
    "attention_probs_dropout_prob": 0.1,
    "directionality": "bidi",
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 4096,
    "max_position_embeddings": 512,
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "pooler_fc_size": 768,
    "pooler_num_attention_heads": 12,
    "pooler_num_fc_layers": 3,
    "pooler_size_per_head": 128,
    "pooler_type": "first_token_transform",
    "type_vocab_size": 2,
    "vocab_size": 28996
}
RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])

device_id = int(os.getenv('DEVICE_ID', "0"))


def parse_args():
    """init."""
    # yapf: disable
    parser = argparse.ArgumentParser()

    model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
    model_g.add_arg("bert_config_path", str, "data/cased_L-24_H-1024_A-16/bert_config.json",
                    "Path to the json file for bert model config.")
    model_g.add_arg("init_pretraining_params", str, "data/cased_L-24_H-1024_A-16/params",
                    "Init pre-training params which performs fine-tuning from. If the "
                    "arg 'init_checkpoint' has been set, this argument wouldn't be valid.")
    model_g.add_arg("checkpoints", str, "output/ 1>log/train.log", "Path to save checkpoints.")

    train_g = ArgumentGroup(parser, "training", "training options.")
    train_g.add_arg("epoch", int, 3, "Number of epoches for fine-tuning.")
    train_g.add_arg("learning_rate", float, 4e-5, "Learning rate used to train with warmup.")
    train_g.add_arg("lr_scheduler", str, "linear_warmup_decay",
                    "scheduler of learning rate.", choices=['linear_warmup_decay', 'noam_decay'])
    train_g.add_arg("weight_decay", float, 0.01, "Weight decay rate for L2 regularizer.")
    train_g.add_arg("warmup_proportion", float, 0.1,
                    "Proportion of training steps to perform linear learning rate warmup for.")
    train_g.add_arg("save_steps", int, 4000, "The steps interval to save checkpoints.")

    log_g = ArgumentGroup(parser, "logging", "logging related.")
    log_g.add_arg("skip_steps", int, 10, "The steps interval to print loss.")
    log_g.add_arg("verbose", ast.literal_eval, False, "Whether to output verbose log.")

    data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
    data_g.add_arg("train_file", str, "data/SQuAD/train-v1.1.json", "SQuAD json for training. E.g., train-v1.1.json.")
    data_g.add_arg("predict_file", str, "data/SQuAD/dev-v1.1.json",
                   "SQuAD json for predictions. E.g. dev-v1.1.json or test-v1.1.json.")
    data_g.add_arg("vocab_path", str, "data/cased_L-24_H-1024_A-16/vocab.txt", "Vocabulary path.")
    data_g.add_arg("version_2_with_negative", ast.literal_eval, False,
                   "If true, the SQuAD examples contain some that do not have an answer. If using squad v2.0, "
                   "it should be set true.")
    data_g.add_arg("max_seq_len", int, 384, "Number of words of the longest sequence.")
    data_g.add_arg("max_query_length", int, 64, "Max query length.")
    data_g.add_arg("max_answer_length", int, 30, "Max answer length.")
    data_g.add_arg("batch_size", int, 8, "Total examples' number in batch for training. see also --in_tokens.")
    data_g.add_arg("in_tokens", ast.literal_eval, False,
                   "If set, the batch size will be the maximum number of tokens in one batch. "
                   "Otherwise, it will be the maximum number of examples in one batch.")
    data_g.add_arg("do_lower_case", ast.literal_eval, False,
                   "Whether to lower case the input text. "
                   "Should be True for uncased models and False for cased models.")
    data_g.add_arg("doc_stride", int, 128,
                   "When splitting up a long document into chunks, how much stride to take between chunks.")
    data_g.add_arg("n_best_size", int, 20,
                   "The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    data_g.add_arg("null_score_diff_threshold", float, 0.0,
                   "If null_score - best_non_null is greater than the threshold predict null.")
    data_g.add_arg("random_seed", int, 45, "Random seed.")

    run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
    run_type_g.add_arg("do_train", ast.literal_eval, True, "Whether to perform training.")
    run_type_g.add_arg("do_val", ast.literal_eval, False, "Whether to perform validation during training.")
    run_type_g.add_arg("do_predict", ast.literal_eval, False, "Whether to perform prediction.")
    run_type_g.add_arg("freeze", ast.literal_eval, False, "freeze bert parameters")

    mem_settings_g = ArgumentGroup(parser, "memory", "memory settings.")
    mem_settings_g.add_arg('wn_concept_embedding_path', str, "KB_embeddings/wn_concept2vec.txt",
                           'path of wordnet pretrained concept file')
    mem_settings_g.add_arg('nell_concept_embedding_path', str, "KB_embeddings/nell_concept2vec.txt",
                           'path of nell pretrained concept file')
    mem_settings_g.add_arg('use_wordnet', ast.literal_eval, True, 'whether to use wordnet memory')
    mem_settings_g.add_arg('retrieved_synset_path', str,
                           'data/retrieve_wordnet/output_squad/retrived_synsets.data',
                           'path of retrieved synsets')
    mem_settings_g.add_arg('use_nell', ast.literal_eval, True, 'whether to use nell memory')
    mem_settings_g.add_arg('train_retrieved_nell_concept_path', str,
                           'data/retrieve_nell/output_squad/train.retrieved_nell_concepts.data',
                           'path of retrieved concepts for trainset')
    mem_settings_g.add_arg('dev_retrieved_nell_concept_path', str,
                           'data/retrieve_nell/output_squad/dev.retrieved_nell_concepts.data',
                           'path of retrieved concepts for devset')

    parser.add_argument('--device_target', type=str, default='Ascend', help='')
    parser.add_argument('--is_distribute', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--device_num', type=int, default=1, help='')
    parser.add_argument('--device_id', type=int, default=0, help='')
    parser.add_argument('--load_pretrain_checkpoint_path', type=str, default="data/cased_L-24_H-1024_A-16/roberta.ckpt",
                        help='')
    parser.add_argument('--train_mindrecord_file', type=str, default="SQuAD/train.mindrecord", help='')
    parser.add_argument('--predict_mindrecord_file', type=str, default="SQuAD/dev.mindrecord", help='')
    parser.add_argument('--save_finetune_checkpoint_path', type=str, default="/cache/output/finetune_checkpoint",
                        help='')
    parser.add_argument('--data_url', type=str, default="", help='')
    parser.add_argument('--train_url', type=str, default="", help='')
    parser.add_argument('--is_modelarts', type=str, default="true", help='')
    parser.add_argument('--save_url', type=str, default="/cache/", help='')
    parser.add_argument('--log_url', type=str, default="/tmp/log/", help='')

    args = parser.parse_args()
    return args


def do_train(dataset=None, network=None, load_checkpoint_path="", save_checkpoint_path="", epoch_num=1):
    """train KTNET model"""
    if load_checkpoint_path == "":
        raise ValueError("Pretrain model missed, finetune task must load pretrain model!")

    args = parse_args()
    step_per_epoch = dataset.get_dataset_size()
    # optimizer
    max_train_steps = epoch_num * dataset.get_dataset_size()
    warmup_steps = int(max_train_steps * args.warmup_proportion)
    lr_schedule = CustomWarmUpLR(learning_rate=args.learning_rate, warmup_steps=warmup_steps,
                                 max_train_steps=max_train_steps)
    optimizer = Adam(network.trainable_params(), learning_rate=lr_schedule, eps=1e-6)

    # load checkpoint into network
    ckpt_config = CheckpointConfig(save_checkpoint_steps=step_per_epoch, keep_checkpoint_max=10)
    ckpoint_cb = ModelCheckpoint(prefix="KTNET_squad",
                                 directory=None if save_checkpoint_path == "" else save_checkpoint_path,
                                 config=ckpt_config)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(network, param_dict)

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2 ** 32, scale_factor=2, scale_window=1000)
    netwithgrads = TrainOneStepWithLossScaleCell(network, optimizer, update_cell)
    model = Model(netwithgrads)

    if not os.path.exists(args.save_finetune_checkpoint_path):
        os.makedirs(args.save_finetune_checkpoint_path)

    if device_id == 0 or args.device_id == 0 or args.device_num == 1:
        model.train(epoch_num, dataset, callbacks=[TimeMonitor(dataset.get_dataset_size()), LossMonitor(), ckpoint_cb],
                    dataset_sink_mode=False)
    else:
        model.train(epoch_num, dataset, callbacks=[TimeMonitor(dataset.get_dataset_size()), LossMonitor()],
                    dataset_sink_mode=False)

    if args.is_modelarts.lower() == "true":
        import moxing as mox

        if device_id == 0:
            mox.file.copy_parallel(args.save_url, args.train_url)
            mox.file.copy_parallel(args.log_url, args.train_url + "/tmp")


def run_KTNET():
    """run ktnet task"""
    args = parse_args()
    epoch_num = args.epoch
    global device_id
    if args.is_modelarts.lower() == "false":
        device_id = args.device_id
    fp = args.save_url + 'data_' + str(device_id) + '/'

    if not (args.do_train or args.do_predict or args.do_val):
        raise ValueError("For args `do_train` and `do_predict`, at "
                         "least one of them must be True.")

    target = args.device_target
    if target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=device_id)
        if args.is_distribute:
            context.set_auto_parallel_context(device_num=args.device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()

    else:
        raise Exception("Target error, GPU or Ascend is supported.")

    save_dir = args.save_url + "model/"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if args.is_modelarts.lower() == "true":
        init()
        os.chdir('/home/work/user-job-dir/ktnet/')

    if args.is_modelarts.lower() == "true":
        context.set_auto_parallel_context(device_num=args.device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)

    if args.is_modelarts.lower() == "true":
        import moxing as mox

        mox.file.copy_parallel(args.data_url + "/", fp)
        ds = create_train_dataset(batch_size=args.batch_size,
                                  data_file=fp + args.train_mindrecord_file,
                                  do_shuffle=True,
                                  device_num=args.device_num, rank=device_id,
                                  num_parallel_workers=8)
    else:
        ds = create_train_dataset(batch_size=args.batch_size,
                                  data_file=args.train_mindrecord_file,
                                  do_shuffle=True,
                                  device_num=args.device_num, rank=device_id,
                                  num_parallel_workers=8)

    if args.do_train:
        netwithloss = KTNET(bert_config=bert_config,
                            max_wn_concept_length=49,
                            max_nell_concept_length=27,
                            wn_vocab_size=40944,
                            wn_embedding_size=112,
                            nell_vocab_size=288,
                            nell_embedding_size=112,
                            bert_size=1024,
                            is_training=True,
                            freeze=args.freeze)
        print("==============================================================")
        print("processor_name: {}".format(args.device_target))
        print("test_name: BERT Finetune Training")
        print("model_name: KTNET")
        print("batch_size: {}".format(args.batch_size))

        do_train(ds, netwithloss, args.load_pretrain_checkpoint_path, args.save_finetune_checkpoint_path, epoch_num)


if __name__ == "__main__":
    run_KTNET()
