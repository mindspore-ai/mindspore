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
postprocess script.
'''

import argparse
import collections
import os
import numpy as np
from mindspore import Tensor
from src.reader.squad_twomemory import DataProcessor as SquadDataProcessor
from src.reader.squad_twomemory import write_predictions as write_predictions_squad
from src.reader.record_twomemory import DataProcessor as RecordDataProcessor
from src.reader.record_twomemory import write_predictions as write_predictions_record

parser = argparse.ArgumentParser(description="postprocess")
parser.add_argument("--batch_size", type=int, default=1, help="Eval batch size, default is 1")
parser.add_argument("--label_dir", type=str, default="", help="label data dir")
parser.add_argument("--result_dir", type=str, default="./result_files", help="infer result Files")
parser.add_argument("--dataset", type=str, default="squad", help="dataset")
parser.add_argument("--data_url", type=str, default="./data", help="data url")
parser.add_argument("--checkpoints", type=str, default="log/eval_310",
                    help="Path to save checkpoints.")

args, _ = parser.parse_known_args()

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


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


if __name__ == "__main__":

    wn_concept2id = read_concept_embedding(args.data_url + '/KB_embeddings/wn_concept2vec.txt')
    nell_concept2id = read_concept_embedding(args.data_url + '/KB_embeddings/nell_concept2vec.txt')

    eval_concept_settings = {
        'tokenization_path': args.data_url + '/tokenization_{}/tokens/dev.tokenization.cased.data'.format(args.dataset),
        'wn_concept2id': wn_concept2id,
        'nell_concept2id': nell_concept2id,
        'use_wordnet': True,
        'retrieved_synset_path':
            args.data_url + '/retrieve_wordnet/output_{}/retrived_synsets.data'.format(args.dataset),
        'use_nell': True,
        'retrieved_nell_concept_path':
            args.data_url + '/retrieve_nell/output_{}/dev.retrieved_nell_concepts.data'.format(args.dataset),
    }

    if args.dataset == 'squad':
        processor = SquadDataProcessor(
            vocab_path=args.data_url + '/cased_L-24_H-1024_A-16/vocab.txt',
            do_lower_case=False,
            max_seq_length=384,
            in_tokens=False,
            doc_stride=128,
            max_query_length=64)

        eval_data = processor.data_generator(
            data_path=args.data_url + '/SQuAD/dev-v1.1.json',
            batch_size=8,
            phase='predict',
            shuffle=False,
            dev_count=1,
            epoch=1,
            **eval_concept_settings)
    else:
        processor = RecordDataProcessor(
            vocab_path=args.data_url + '/cased_L-24_H-1024_A-16/vocab.txt',
            do_lower_case=False,
            max_seq_length=384,
            in_tokens=False,
            doc_stride=128,
            max_query_length=64)

        eval_data = processor.data_generator(
            data_path=args.data_url + '/ReCoRD/dev.json',
            batch_size=8,
            phase='predict',
            shuffle=False,
            dev_count=1,
            epoch=1,
            **eval_concept_settings)

    file_name = os.listdir(args.label_dir)
    all_results = []
    for f in file_name:
        f_name = os.path.join(args.result_dir, f.split('.')[0] + '_0.bin')
        print(f_name)
        logits = np.fromfile(f_name, np.float32)
        logits = logits.reshape((2, args.batch_size, 384))

        start_logits, end_logits = np.split(logits, 2, 0)
        start_logits_tensor = Tensor(np.squeeze(start_logits, axis=0))
        end_logits_tensor = Tensor(np.squeeze(end_logits, axis=0))

        unique_ids = np.fromfile(os.path.join(args.label_dir, f), np.int32)
        unique_ids_tensor = Tensor(unique_ids[0].reshape(args.batch_size, 1))

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
    output_evaluation_result_file = os.path.join(args.checkpoints, "eval_result.json")

    features = processor.get_features(
        processor.predict_examples, is_training=False, **eval_concept_settings)
    if args.dataset == 'squad':
        eval_result = write_predictions_squad(processor.predict_examples, features, all_results,
                                              20, 30, False, output_prediction_file,
                                              output_nbest_file, output_null_log_odds_file,
                                              False, 0.0, False, args.data_url + '/SQuAD/dev-v1.1.json',
                                              output_evaluation_result_file)
    else:
        eval_result = write_predictions_record(processor.predict_examples, features, all_results,
                                               20, 30, False, output_prediction_file,
                                               output_nbest_file, output_null_log_odds_file,
                                               False, 0.0, False, args.data_url + '/ReCoRD/dev.json',
                                               output_evaluation_result_file)
    print("==============================================================")
    print(eval_result)
    print("==============================================================")
