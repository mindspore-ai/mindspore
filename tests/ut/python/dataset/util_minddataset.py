# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
This module contains common utility functions for minddataset tests.
"""
import os
import re
import string
import collections

import pytest
import numpy as np

from mindspore.mindrecord import FileWriter

FILES_NUM = 4
CV_DIR_NAME = "../data/mindrecord/testImageNetData"


def get_data(dir_name):
    """
    usage: get data from imagenet dataset
    params:
    dir_name: directory containing folder images and annotation information

    """
    if not os.path.isdir(dir_name):
        raise IOError("Directory {} does not exist".format(dir_name))
    img_dir = os.path.join(dir_name, "images")
    ann_file = os.path.join(dir_name, "annotation.txt")
    with open(ann_file, "r") as file_reader:
        lines = file_reader.readlines()

    data_list = []
    for i, line in enumerate(lines):
        try:
            filename, label = line.split(",")
            label = label.strip("\n")
            with open(os.path.join(img_dir, filename), "rb") as file_reader:
                img = file_reader.read()
            data_json = {"id": i,
                         "file_name": filename,
                         "data": img,
                         "label": int(label)}
            data_list.append(data_json)
        except FileNotFoundError:
            continue
    return data_list


def inputs(vectors, maxlen=50):
    length = len(vectors)
    if length > maxlen:
        return vectors[0:maxlen], [1] * maxlen, [0] * maxlen
    input_ = vectors + [0] * (maxlen - length)
    mask = [1] * length + [0] * (maxlen - length)
    segment = [0] * maxlen
    return input_, mask, segment


def convert_to_uni(text):
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


def get_nlp_data(dir_name, vocab_file, num):
    """
    Return raw data of aclImdb dataset.

    Args:
        dir_name (str): String of aclImdb dataset's path.
        vocab_file (str): String of dictionary's path.
        num (int): Number of sample.

    Returns:
        List
    """
    if not os.path.isdir(dir_name):
        raise IOError("Directory {} not exists".format(dir_name))
    for root, _, files in os.walk(dir_name):
        for index, file_name_extension in enumerate(files):
            if index < num:
                file_path = os.path.join(root, file_name_extension)
                file_name, _ = file_name_extension.split('.', 1)
                id_, rating = file_name.split('_', 1)
                with open(file_path, 'r') as f:
                    raw_content = f.read()

                dictionary = load_vocab(vocab_file)
                vectors = [dictionary.get('[CLS]')]
                vectors += [dictionary.get(i) if i in dictionary
                            else dictionary.get('[UNK]')
                            for i in re.findall(r"[\w']+|[{}]"
                                                .format(string.punctuation),
                                                raw_content)]
                vectors += [dictionary.get('[SEP]')]
                input_, mask, segment = inputs(vectors)
                input_ids = np.reshape(np.array(input_), [-1])
                input_mask = np.reshape(np.array(mask), [1, -1])
                segment_ids = np.reshape(np.array(segment), [2, -1])
                data = {
                    "label": 1,
                    "id": id_,
                    "rating": float(rating),
                    "input_ids": input_ids,
                    "input_mask": input_mask,
                    "segment_ids": segment_ids
                }
                yield data


@pytest.fixture
def add_and_remove_cv_file():
    """add/remove cv file"""
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    paths = ["{}{}".format(file_name, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    try:
        for x in paths:
            if os.path.exists("{}".format(x)):
                os.remove("{}".format(x))
            if os.path.exists("{}.db".format(x)):
                os.remove("{}.db".format(x))
        writer = FileWriter(file_name, FILES_NUM)
        data = get_data(CV_DIR_NAME)
        cv_schema_json = {"id": {"type": "int32"},
                          "file_name": {"type": "string"},
                          "label": {"type": "int32"},
                          "data": {"type": "bytes"}}
        writer.add_schema(cv_schema_json, "img_schema")
        writer.add_index(["file_name", "label"])
        writer.write_raw_data(data)
        writer.commit()
        yield "yield_cv_data"
    except Exception as error:
        for x in paths:
            os.remove("{}".format(x))
            os.remove("{}.db".format(x))
        raise error
    else:
        for x in paths:
            os.remove("{}".format(x))
            os.remove("{}.db".format(x))


@pytest.fixture
def add_and_remove_file():
    """add/remove file"""
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    paths = ["{}{}".format(file_name + "_cv", str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    paths += ["{}{}".format(file_name + "_nlp", str(x).rjust(1, '0'))
              for x in range(FILES_NUM)]
    try:
        for x in paths:
            if os.path.exists("{}".format(x)):
                os.remove("{}".format(x))
            if os.path.exists("{}.db".format(x)):
                os.remove("{}.db".format(x))
        writer = FileWriter(file_name + "_nlp", FILES_NUM)
        data = list(get_nlp_data("../data/mindrecord/testAclImdbData/pos",
                                 "../data/mindrecord/testAclImdbData/vocab.txt",
                                 10))
        nlp_schema_json = {"id": {"type": "string"}, "label": {"type": "int32"},
                           "rating": {"type": "float32"},
                           "input_ids": {"type": "int64",
                                         "shape": [1, -1]},
                           "input_mask": {"type": "int64",
                                          "shape": [1, -1]},
                           "segment_ids": {"type": "int64",
                                           "shape": [1, -1]}
                           }
        writer.add_schema(nlp_schema_json, "nlp_schema")
        writer.add_index(["id", "rating"])
        writer.write_raw_data(data)
        writer.commit()

        writer = FileWriter(file_name + "_cv", FILES_NUM)
        data = get_data(CV_DIR_NAME)
        cv_schema_json = {"id": {"type": "int32"},
                          "file_name": {"type": "string"},
                          "label": {"type": "int32"},
                          "data": {"type": "bytes"}}
        writer.add_schema(cv_schema_json, "img_schema")
        writer.add_index(["file_name", "label"])
        writer.write_raw_data(data)
        writer.commit()

        yield "yield_data"
    except Exception as error:
        for x in paths:
            os.remove("{}".format(x))
            os.remove("{}.db".format(x))
        raise error
    else:
        for x in paths:
            os.remove("{}".format(x))
            os.remove("{}.db".format(x))
