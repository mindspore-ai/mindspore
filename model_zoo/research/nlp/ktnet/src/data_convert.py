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
"""Convert data."""

import argparse
import numpy as np
from src.reader.record_twomemory import DataProcessor as RecordDataProcessor
from src.reader.squad_twomemory import DataProcessor as SquadDataProcessor
from mindspore.mindrecord import FileWriter


def parse_args():
    """init."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_url', type=str, default="./data", help='')

    args = parser.parse_args()
    return args


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


def convert_record_train_data():
    """convert record train data"""
    args = parse_args()
    wn_concept2id = read_concept_embedding(args.data_url + "/KB_embeddings/wn_concept2vec.txt")
    nell_concept2id = read_concept_embedding(args.data_url + "/KB_embeddings/nell_concept2vec.txt")

    processor = RecordDataProcessor(
        vocab_path=args.data_url + "/cased_L-24_H-1024_A-16/vocab.txt",
        do_lower_case=False,
        max_seq_length=384,
        in_tokens=False,
        doc_stride=128,
        max_query_length=64)

    print("record train data process begin")
    train_concept_settings = {
        'tokenization_path': args.data_url + '/tokenization_record/tokens/train.tokenization.cased.data',
        'wn_concept2id': wn_concept2id,
        'nell_concept2id': nell_concept2id,
        'use_wordnet': True,
        'retrieved_synset_path': args.data_url + "/retrieve_wordnet/output_record/retrived_synsets.data",
        'use_nell': True,
        'retrieved_nell_concept_path':
            args.data_url + "/retrieve_nell/output_record/train.retrieved_nell_concepts.data",
    }
    train_data_generator = processor.data_generator(
        data_path=args.data_url + "/ReCoRD/train.json",
        batch_size=1,
        phase='train',
        shuffle=True,
        dev_count=1,
        version_2_with_negative=False,
        epoch=1,
        **train_concept_settings)

    datalist = []
    for item in train_data_generator():
        sample = {
            "src_ids": item[0],
            "pos_ids": item[1],
            "sent_ids": item[2],
            "wn_concept_ids": item[3],
            "nell_concept_ids": item[4],
            "input_mask": item[5],
            "start_positions": item[6],
            "end_positions": item[7]
        }
        datalist.append(sample)

    print("record predict data process end")
    writer = FileWriter(file_name=args.data_url + "/ReCoRD/train.mindrecord", shard_num=1)
    nlp_schema = {
        "src_ids": {"type": "int64", "shape": [384]},
        "pos_ids": {"type": "int64", "shape": [384]},
        "sent_ids": {"type": "int64", "shape": [384]},
        "wn_concept_ids": {"type": "int64", "shape": [384, processor.train_wn_max_concept_length, 1]},
        "nell_concept_ids": {"type": "int64", "shape": [384, processor.train_nell_max_concept_length, 1]},
        "input_mask": {"type": "float32", "shape": [384]},
        "start_positions": {"type": "int64", "shape": [1]},
        "end_positions": {"type": "int64", "shape": [1]},
    }
    writer.add_schema(nlp_schema, "proprocessed classification dataset")
    writer.write_raw_data(datalist)
    writer.commit()
    print("record train data write success")


def convert_record_predict_data():
    """convert record predict data"""
    args = parse_args()
    wn_concept2id = read_concept_embedding(args.data_url + "/KB_embeddings/wn_concept2vec.txt")
    nell_concept2id = read_concept_embedding(args.data_url + "/KB_embeddings/nell_concept2vec.txt")

    processor = RecordDataProcessor(
        vocab_path=args.data_url + "/cased_L-24_H-1024_A-16/vocab.txt",
        do_lower_case=False,
        max_seq_length=384,
        in_tokens=False,
        doc_stride=128,
        max_query_length=64)

    print("record predict data process begin")
    eval_concept_settings = {
        'tokenization_path': args.data_url + '/tokenization_record/tokens/dev.tokenization.cased.data',
        'wn_concept2id': wn_concept2id,
        'nell_concept2id': nell_concept2id,
        'use_wordnet': True,
        'retrieved_synset_path': args.data_url + "/retrieve_wordnet/output_record/retrived_synsets.data",
        'use_nell': True,
        'retrieved_nell_concept_path': args.data_url + "/retrieve_nell/output_record/dev.retrieved_nell_concepts.data",
    }
    eval_data_generator = processor.data_generator(
        data_path=args.data_url + "/ReCoRD/dev.json",
        batch_size=1,
        phase='predict',
        shuffle=False,
        dev_count=1,
        epoch=1,
        **eval_concept_settings)

    datalist = []
    for item in eval_data_generator():
        sample = {
            "src_ids": item[0],
            "pos_ids": item[1],
            "sent_ids": item[2],
            "wn_concept_ids": item[3],
            "nell_concept_ids": item[4],
            "input_mask": item[5],
            "unique_id": item[6]
        }
        datalist.append(sample)

    print("squad predict data process end")
    writer = FileWriter(file_name=args.data_url + "/ReCoRD/dev.mindrecord", shard_num=1)
    nlp_schema = {
        "src_ids": {"type": "int64", "shape": [384]},
        "pos_ids": {"type": "int64", "shape": [384]},
        "sent_ids": {"type": "int64", "shape": [384]},
        "wn_concept_ids": {"type": "int64", "shape": [384, processor.predict_wn_max_concept_length, 1]},
        "nell_concept_ids": {"type": "int64", "shape": [384, processor.predict_nell_max_concept_length, 1]},
        "input_mask": {"type": "float32", "shape": [384]},
        "unique_id": {"type": "int64", "shape": [1]}
    }
    writer.add_schema(nlp_schema, "proprocessed classification dataset")
    writer.write_raw_data(datalist)
    writer.commit()
    print("record predict data write success")


def convert_squad_train_data():
    """convert squad train data"""
    args = parse_args()
    wn_concept2id = read_concept_embedding(args.data_url + "/KB_embeddings/wn_concept2vec.txt")
    nell_concept2id = read_concept_embedding(args.data_url + "/KB_embeddings/nell_concept2vec.txt")

    processor = SquadDataProcessor(
        vocab_path=args.data_url + "/cased_L-24_H-1024_A-16/vocab.txt",
        do_lower_case=False,
        max_seq_length=384,
        in_tokens=False,
        doc_stride=128,
        max_query_length=64)

    print("squad train data process begin")
    train_concept_settings = {
        'tokenization_path': args.data_url + '/tokenization_squad/tokens/train.tokenization.cased.data',
        'wn_concept2id': wn_concept2id,
        'nell_concept2id': nell_concept2id,
        'use_wordnet': True,
        'retrieved_synset_path': args.data_url + "/retrieve_wordnet/output_squad/retrived_synsets.data",
        'use_nell': True,
        'retrieved_nell_concept_path': args.data_url + "/retrieve_nell/output_squad/train.retrieved_nell_concepts.data",
    }
    train_data_generator = processor.data_generator(
        data_path=args.data_url + "/SQuAD/train-v1.1.json",
        batch_size=1,
        phase='train',
        shuffle=True,
        dev_count=1,
        version_2_with_negative=False,
        epoch=1,
        **train_concept_settings)

    datalist = []
    for item in train_data_generator():
        sample = {
            "src_ids": item[0],
            "pos_ids": item[1],
            "sent_ids": item[2],
            "wn_concept_ids": item[3],
            "nell_concept_ids": item[4],
            "input_mask": item[5],
            "start_positions": item[6],
            "end_positions": item[7]
        }
        datalist.append(sample)

    print("squad train data process end")
    writer = FileWriter(file_name=args.data_url + "/SQuAD/train.mindrecord", shard_num=1)
    nlp_schema = {
        "src_ids": {"type": "int64", "shape": [384]},
        "pos_ids": {"type": "int64", "shape": [384]},
        "sent_ids": {"type": "int64", "shape": [384]},
        "wn_concept_ids": {"type": "int64", "shape": [384, processor.train_wn_max_concept_length, 1]},
        "nell_concept_ids": {"type": "int64", "shape": [384, processor.train_nell_max_concept_length, 1]},
        "input_mask": {"type": "float32", "shape": [384]},
        "start_positions": {"type": "int64", "shape": [1]},
        "end_positions": {"type": "int64", "shape": [1]},
    }
    writer.add_schema(nlp_schema, "proprocessed classification dataset")
    writer.write_raw_data(datalist)
    writer.commit()
    print("squad train data write success")


def convert_squad_predict_data():
    """convert squad predict data"""
    args = parse_args()
    wn_concept2id = read_concept_embedding(args.data_url + "/KB_embeddings/wn_concept2vec.txt")
    nell_concept2id = read_concept_embedding(args.data_url + "/KB_embeddings/nell_concept2vec.txt")

    processor = SquadDataProcessor(
        vocab_path=args.data_url + "/cased_L-24_H-1024_A-16/vocab.txt",
        do_lower_case=False,
        max_seq_length=384,
        in_tokens=False,
        doc_stride=128,
        max_query_length=64)

    print("squad predict data process begin")
    eval_concept_settings = {
        'tokenization_path': args.data_url + '/tokenization_squad/tokens/dev.tokenization.cased.data',
        'wn_concept2id': wn_concept2id,
        'nell_concept2id': nell_concept2id,
        'use_wordnet': True,
        'retrieved_synset_path': args.data_url + "/retrieve_wordnet/output_squad/retrived_synsets.data",
        'use_nell': True,
        'retrieved_nell_concept_path': args.data_url + "/retrieve_nell/output_squad/dev.retrieved_nell_concepts.data",
    }
    eval_data_generator = processor.data_generator(
        data_path=args.data_url + "/SQuAD/dev-v1.1.json",
        batch_size=1,
        phase='predict',
        shuffle=False,
        dev_count=1,
        epoch=1,
        **eval_concept_settings)

    datalist = []
    for item in eval_data_generator():
        sample = {
            "src_ids": item[0],
            "pos_ids": item[1],
            "sent_ids": item[2],
            "wn_concept_ids": item[3],
            "nell_concept_ids": item[4],
            "input_mask": item[5],
            "unique_id": item[6]
        }
        datalist.append(sample)

    print("squad predict data process end")
    writer = FileWriter(file_name=args.data_url + "/SQuAD/dev.mindrecord", shard_num=1)
    nlp_schema = {
        "src_ids": {"type": "int64", "shape": [384]},
        "pos_ids": {"type": "int64", "shape": [384]},
        "sent_ids": {"type": "int64", "shape": [384]},
        "wn_concept_ids": {"type": "int64", "shape": [384, processor.predict_wn_max_concept_length, 1]},
        "nell_concept_ids": {"type": "int64", "shape": [384, processor.predict_nell_max_concept_length, 1]},
        "input_mask": {"type": "float32", "shape": [384]},
        "unique_id": {"type": "int64", "shape": [1]}
    }
    writer.add_schema(nlp_schema, "proprocessed classification dataset")
    writer.write_raw_data(datalist)
    writer.commit()
    print("squad predict data write success")


if __name__ == '__main__':
    convert_record_train_data()
    convert_record_predict_data()
    convert_squad_train_data()
    convert_squad_predict_data()
