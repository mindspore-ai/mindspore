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
"""get data from aclImdb and write the data to mindrecord file"""
import collections
import os
import re
import string
import numpy as np
from mindspore.mindrecord import FileWriter

ACLIMDB_DIR = "data/aclImdb"

MINDRECORD_FILE_NAME_TRAIN = "output/aclImdb_train.mindrecord"
MINDRECORD_FILE_NAME_TEST = "output/aclImdb_test.mindrecord"

def inputs(vectors, maxlen=50):
    """generate input_ids, mask, segemnt"""
    length = len(vectors)
    if length > maxlen:
        return vectors[0:maxlen], [1]*maxlen, [0]*maxlen
    input_ = vectors+[0]*(maxlen-length)
    mask = [1]*length + [0]*(maxlen-length)
    segment = [0]*maxlen
    return input_, mask, segment

def get_nlp_data(data_dir, vocab_dict):
    """get data from dir like aclImdb/train"""
    dir_list = [os.path.join(data_dir, "pos"),
                os.path.join(data_dir, "neg")]

    for index, exact_dir in enumerate(dir_list):
        if not os.path.exists(exact_dir):
            raise IOError("dir {} not exists".format(exact_dir))

        vocab_dict = load_vocab(os.path.join(data_dir, "../imdb.vocab"))

        for item in os.listdir(exact_dir):
            data = {}

            # file name like 4372_2.txt, we will get id: 4372, score: 2
            id_score = item.split("_", 1)
            score = id_score[1].split(".", 1)

            review_file = open(os.path.join(exact_dir, item), "r")
            review = review_file.read()
            review_file.close()

            vectors = []
            vectors += [vocab_dict.get(i) if i in vocab_dict else -1
                        for i in re.findall(r"[\w']+|[{}]".format(string.punctuation), review)]
            input_, mask, segment = inputs(vectors)
            input_ids = np.reshape(np.array(input_), [1, -1])
            input_mask = np.reshape(np.array(mask), [1, -1])
            segment_ids = np.reshape(np.array(segment), [1, -1])

            data = {
                "label": int(index),         # indicate pos: 0, neg: 1
                "id": int(id_score[0]),
                "score": int(score[0]),
                "input_ids": input_ids,      # raw content convert it
                "input_mask": input_mask,    # raw content convert it
                "segment_ids": segment_ids   # raw content convert it
            }
            yield data

def convert_to_uni(text):
    """convert bytes to text"""
    if isinstance(text, str):
        return text
    if isinstance(text, bytes):
        return text.decode('utf-8', 'ignore')
    raise Exception("The type %s does not convert!" % type(text))

def load_vocab(vocab_file):
    """load vocabulary to translate statement."""
    vocab = collections.OrderedDict()
    vocab.setdefault('blank', 2)
    index = 0
    with open(vocab_file) as reader:
        while True:
            tmp = reader.readline()
            if not tmp:
                break
            token = convert_to_uni(tmp)
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab

def gen_mindrecord(data_type):
    """gen mindreocrd according exactly schema"""
    if data_type == "train":
        fw = FileWriter(MINDRECORD_FILE_NAME_TRAIN)
    else:
        fw = FileWriter(MINDRECORD_FILE_NAME_TEST)

    schema = {"id": {"type": "int32"},
              "label": {"type": "int32"},
              "score": {"type": "int32"},
              "input_ids": {"type": "int32", "shape": [-1]},
              "input_mask": {"type": "int32", "shape": [-1]},
              "segment_ids": {"type": "int32", "shape": [-1]}}
    fw.add_schema(schema, "aclImdb preprocessed dataset")
    fw.add_index(["id", "label", "score"])

    vocab_dict = load_vocab(os.path.join(ACLIMDB_DIR, "imdb.vocab"))

    get_data_iter = get_nlp_data(os.path.join(ACLIMDB_DIR, data_type), vocab_dict)

    batch_size = 256
    transform_count = 0
    while True:
        data_list = []
        try:
            for _ in range(batch_size):
                data_list.append(get_data_iter.__next__())
                transform_count += 1
            fw.write_raw_data(data_list)
            print(">> transformed {} record...".format(transform_count))
        except StopIteration:
            if data_list:
                fw.write_raw_data(data_list)
                print(">> transformed {} record...".format(transform_count))
            break

    fw.commit()

def main():
    # generate mindrecord for train
    print(">> begin generate mindrecord by train data")
    gen_mindrecord("train")

    # generate mindrecord for test
    print(">> begin generate mindrecord by test data")
    gen_mindrecord("test")

if __name__ == "__main__":
    main()
