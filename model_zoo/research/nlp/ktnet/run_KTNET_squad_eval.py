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
"""Evaluation on Squad."""

import argparse
import collections
import os
import logging
import numpy as np

from src.KTNET_eval import KTNET_eval
from src.reader.squad_twomemory import DataProcessor, write_predictions
from utils.args import ArgumentGroup

import mindspore
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.ops as ops

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


def parse_args():
    """init."""
    # yapf: disable
    parser = argparse.ArgumentParser()

    model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
    model_g.add_arg("bert_config_path", str, "data/cased_L-24_H-1024_A-16/bert_config.json",
                    "Path to the json file for bert model config.")
    model_g.add_arg("init_checkpoint", str, None, "Init checkpoint to resume training from.")
    model_g.add_arg("init_pretraining_params", str, "data/cased_L-24_H-1024_A-16/params",
                    "Init pre-training params which performs fine-tuning from. If the "
                    "arg 'init_checkpoint' has been set, this argument wouldn't be valid.")
    model_g.add_arg("checkpoints", str, "output/ 1>log/train.log", "Path to save checkpoints.")

    train_g = ArgumentGroup(parser, "training", "training options.")
    train_g.add_arg("epoch", int, 1, "Number of epoches for fine-tuning.")
    train_g.add_arg("learning_rate", float, 4e-5, "Learning rate used to train with warmup.")
    train_g.add_arg("lr_scheduler", str, "linear_warmup_decay",
                    "scheduler of learning rate.", choices=['linear_warmup_decay', 'noam_decay'])
    train_g.add_arg("weight_decay", float, 0.01, "Weight decay rate for L2 regularizer.")
    train_g.add_arg("warmup_proportion", float, 0.1,
                    "Proportion of training steps to perform linear learning rate warmup for.")
    train_g.add_arg("save_steps", int, 4000, "The steps interval to save checkpoints.")
    train_g.add_arg("validation_steps", int, 1000,
                    "The steps interval for validation (effective only when do_val is True).")
    train_g.add_arg("use_ema", bool, True, "Whether to use ema.")
    train_g.add_arg("ema_decay", float, 0.9999, "Decay rate for exponential moving average.")
    train_g.add_arg("use_fp16", bool, False, "Whether to use fp16 mixed precision training.")
    train_g.add_arg("loss_scaling", float, 1.0,
                    "Loss scaling factor for mixed precision training, only valid when use_fp16 is enabled.")

    log_g = ArgumentGroup(parser, "logging", "logging related.")
    log_g.add_arg("skip_steps", int, 10, "The steps interval to print loss.")
    log_g.add_arg("verbose", bool, False, "Whether to output verbose log.")

    data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
    data_g.add_arg("train_file", str, "data/SQuAD/train-v1.1.json", "SQuAD json for training. E.g., train-v1.1.json.")
    data_g.add_arg("predict_file", str, "./data/SQuAD/dev-v1.1.json",
                   "SQuAD json for predictions. E.g. dev-v1.1.json or test-v1.1.json.")
    data_g.add_arg("vocab_path", str, "data/cased_L-24_H-1024_A-16/vocab.txt", "Vocabulary path.")
    data_g.add_arg("version_2_with_negative", bool, False,
                   "If true, the SQuAD examples contain some that do not have an answer. If using squad v2.0, "
                   "it should be set true.")
    data_g.add_arg("max_seq_len", int, 384, "Number of words of the longest sequence.")
    data_g.add_arg("max_query_length", int, 64, "Max query length.")
    data_g.add_arg("max_answer_length", int, 30, "Max answer length.")
    data_g.add_arg("batch_size", int, 8, "Total examples' number in batch for training. see also --in_tokens.")
    data_g.add_arg("in_tokens", bool, False,
                   "If set, the batch size will be the maximum number of tokens in one batch. "
                   "Otherwise, it will be the maximum number of examples in one batch.")
    data_g.add_arg("do_lower_case", bool, False, "Whether to lower case the input text. Should be True for uncased "
                                                 "models and False for cased models.")
    data_g.add_arg("doc_stride", int, 128,
                   "When splitting up a long document into chunks, how much stride to take between chunks.")
    data_g.add_arg("n_best_size", int, 20,
                   "The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    data_g.add_arg("null_score_diff_threshold", float, 0.0,
                   "If null_score - best_non_null is greater than the threshold predict null.")
    data_g.add_arg("random_seed", int, 45, "Random seed.")

    run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
    run_type_g.add_arg("use_cuda", bool, True, "If set, use GPU for training.")
    run_type_g.add_arg("use_fast_executor", bool, False, "If set, use fast parallel executor (in experiment).")
    run_type_g.add_arg("num_iteration_per_drop_scope", int, 1,
                       "Ihe iteration intervals to clean up temporary variables.")
    run_type_g.add_arg("do_train", bool, False, "Whether to perform training.")
    run_type_g.add_arg("do_val", bool, True, "Whether to perform validation during training.")
    run_type_g.add_arg("do_predict", bool, True, "Whether to perform prediction.")
    run_type_g.add_arg("freeze", bool, False, "freeze bert parameters")

    mem_settings_g = ArgumentGroup(parser, "memory", "memory settings.")
    mem_settings_g.add_arg('wn_concept_embedding_path', str, "data/KB_embeddings/wn_concept2vec.txt",
                           'path of wordnet pretrained concept file')
    mem_settings_g.add_arg('nell_concept_embedding_path', str, "data/KB_embeddings/nell_concept2vec.txt",
                           'path of nell pretrained concept file')
    mem_settings_g.add_arg('use_wordnet', bool, True, 'whether to use wordnet memory')
    mem_settings_g.add_arg('retrieved_synset_path', str,
                           '/retrieve_wordnet/output_squad/retrived_synsets.data',
                           'path of retrieved synsets')
    mem_settings_g.add_arg('use_nell', bool, True, 'whether to use nell memory')
    mem_settings_g.add_arg('train_retrieved_nell_concept_path', str,
                           '/retrieve_nell/output_squad/train.retrieved_nell_concepts.data',
                           'path of retrieved concepts for trainset')
    mem_settings_g.add_arg('dev_retrieved_nell_concept_path', str,
                           '/retrieve_nell/output_squad/dev.retrieved_nell_concepts.data',
                           'path of retrieved concepts for devset')

    parser.add_argument('--device_target', type=str, default='Ascend', help='')
    parser.add_argument('--device_id', type=int, default=0, help='')
    parser.add_argument('--load_pretrain_checkpoint_path', type=str, default='', help='')
    parser.add_argument('--train_mindrecord_file', type=str, default='', help='')
    parser.add_argument('--predict_mindrecord_file', type=str, default='data/SQuAD/dev.mindrecord', help='')
    parser.add_argument('--save_finetune_checkpoint_path', type=str, default='', help='')
    parser.add_argument('--load_checkpoint_path', type=str,
                        default='output/finetune_checkpoint/KTNET_squad-1_11010.ckpt', help='')
    parser.add_argument('--data_url', type=str, default="data", help='')

    args = parser.parse_args()
    return args


def do_eval(processor, eval_concept_settings, eval_output_name='eval_result.json', network=None,
            load_checkpoint_path=""):
    """ do eval """
    args = parse_args()
    if load_checkpoint_path == "":
        raise ValueError("Finetune model missed, evaluation task must load finetune model!")

    eval_data = processor.data_generator(
        data_path=args.predict_file,
        batch_size=args.batch_size,
        phase='predict',
        shuffle=False,
        dev_count=1,
        epoch=1,
        **eval_concept_settings)

    net_for_pretraining = network(bert_config=bert_config,
                                  max_wn_concept_length=49,
                                  max_nell_concept_length=27,
                                  wn_vocab_size=40944,
                                  wn_embedding_size=112,
                                  nell_vocab_size=288,
                                  nell_embedding_size=112,
                                  bert_size=1024,
                                  is_training=True,
                                  freeze=args.freeze)
    net_for_pretraining.set_train(False)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(net_for_pretraining, param_dict)
    model = Model(net_for_pretraining)

    all_results = []

    for data in eval_data():
        src_ids = Tensor(np.squeeze(data[0]), mindspore.int32)
        pos_ids = Tensor(np.squeeze(data[1]), mindspore.int32)
        sent_ids = Tensor(np.squeeze(data[2]), mindspore.int32)
        wn_concept_ids = Tensor(data[3], mindspore.int32)
        nell_concept_ids = Tensor(data[4], mindspore.int32)
        input_mask = Tensor(np.squeeze(data[5]), mindspore.float32)
        unique_id = Tensor(data[6], mindspore.int32)

        pad = ops.Pad(((0, 0), (0, 0), (0, 3), (0, 0)))
        nell_concept_ids = pad(nell_concept_ids)

        logits_tensor = model.predict(input_mask, src_ids, pos_ids, sent_ids, wn_concept_ids,
                                      nell_concept_ids, unique_id)
        unstack = ops.Split(0, 2)
        start_logits_tensor, end_logits_tensor = unstack(logits_tensor)
        start_logits_tensor = np.squeeze(start_logits_tensor)
        end_logits_tensor = np.squeeze(end_logits_tensor)
        unique_ids_tensor = unique_id

        np_unique_ids = unique_ids_tensor.asnumpy()
        np_start_logits = start_logits_tensor.asnumpy()
        np_end_logits = end_logits_tensor.asnumpy()

        for idx in range(np_unique_ids.shape[0]):
            if len(all_results) % 1000 == 0:
                print("Processing example: %d" % len(all_results))
            unique_id = int(np_unique_ids[idx])
            start_logits = [float(x) for x in np_start_logits[idx].flat]
            end_logits = [float(x) for x in np_end_logits[idx].flat]

            all_results.append(RawResult(
                unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits))

    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)
    output_prediction_file = os.path.join(args.checkpoints, "predictions.json")
    output_nbest_file = os.path.join(args.checkpoints, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(args.checkpoints, "null_odds.json")
    output_evaluation_result_file = os.path.join(args.checkpoints, eval_output_name)
    features = processor.get_features(
        processor.predict_examples, is_training=False, **eval_concept_settings)
    eval_result = write_predictions(processor.predict_examples, features, all_results,
                                    args.n_best_size, args.max_answer_length,
                                    args.do_lower_case, output_prediction_file,
                                    output_nbest_file, output_null_log_odds_file,
                                    args.version_2_with_negative,
                                    args.null_score_diff_threshold, args.verbose, args.predict_file,
                                    output_evaluation_result_file)
    print("==============================================================")
    print(eval_result)
    print("==============================================================")


def read_concept_embedding(embedding_path):
    """read concept embedding"""
    fin = open(embedding_path, encoding='utf-8')
    info = [line.strip() for line in fin]
    dim = len(info[0].split(' ')[1:])
    embedding_mat = []
    id2concept, concept2id = [], {}
    # add padding concept into vocab
    id2concept.append('<pad_concept>')
    concept2id['<pad_concept>'] = 0
    embedding_mat.append([0.0 for _ in range(dim)])
    for line in info:
        concept_name = line.split(' ')[0]
        embedding = [float(value_str) for value_str in line.split(' ')[1:]]
        assert len(embedding) == dim and not np.any(np.isnan(embedding))
        embedding_mat.append(embedding)
        concept2id[concept_name] = len(id2concept)
        id2concept.append(concept_name)
    return concept2id


def run_KTNET():
    """run ktnet task"""
    args = parse_args()

    wn_concept2id = read_concept_embedding(args.wn_concept_embedding_path)
    nell_concept2id = read_concept_embedding(args.nell_concept_embedding_path)

    if not (args.do_train or args.do_predict or args.do_val):
        raise ValueError("For args `do_train` and `do_predict`, at "
                         "least one of them must be True.")

    target = args.device_target
    if target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args.device_id)
    else:
        raise Exception("Target error, GPU or Ascend is supported.")

    if args.do_predict:
        eval_concept_settings = {
            'tokenization_path': args.data_url + '/tokenization_squad/tokens/dev.tokenization.{}.data'.format(
                'uncased' if args.do_lower_case else 'cased'),
            'wn_concept2id': wn_concept2id,
            'nell_concept2id': nell_concept2id,
            'use_wordnet': args.use_wordnet,
            'retrieved_synset_path': args.data_url + args.retrieved_synset_path,
            'use_nell': args.use_nell,
            'retrieved_nell_concept_path': args.data_url + args.dev_retrieved_nell_concept_path,
        }
        processor = DataProcessor(
            vocab_path=args.vocab_path,
            do_lower_case=args.do_lower_case,
            max_seq_length=args.max_seq_len,
            in_tokens=args.in_tokens,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length)

        do_eval(processor, eval_concept_settings, network=KTNET_eval,
                load_checkpoint_path=args.load_checkpoint_path)


if __name__ == "__main__":
    run_KTNET()
