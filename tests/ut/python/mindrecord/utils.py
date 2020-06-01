# Copyright 2019 Huawei Technologies Co., Ltd
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
"""utils for test"""

import os
import re
import string
import collections
import json
import numpy as np

from mindspore import log as logger


def get_data(dir_name):
    """
    Return raw data of imagenet dataset.

    Args:
        dir_name (str): String of imagenet dataset's path.

    Returns:
        List
    """
    if not os.path.isdir(dir_name):
        raise IOError("Directory {} not exists".format(dir_name))
    img_dir = os.path.join(dir_name, "images")
    ann_file = os.path.join(dir_name, "annotation.txt")
    with open(ann_file, "r") as file_reader:
        lines = file_reader.readlines()

    data_list = []
    for line in lines:
        try:
            filename, label = line.split(",")
            label = label.strip("\n")
            with open(os.path.join(img_dir, filename), "rb") as file_reader:
                img = file_reader.read()
            data_json = {"file_name": filename,
                         "data": img,
                         "label": int(label)}
            data_list.append(data_json)
        except FileNotFoundError:
            continue
    return data_list


def get_two_bytes_data(file_name):
    """
    Return raw data of two-bytes dataset.

    Args:
        file_name (str): String of two-bytes dataset's path.

    Returns:
        List
    """
    if not os.path.exists(file_name):
        raise IOError("map file {} not exists".format(file_name))
    dir_name = os.path.dirname(file_name)
    with open(file_name, "r") as file_reader:
        lines = file_reader.readlines()
    data_list = []
    row_num = 0
    for line in lines:
        try:
            img, label = line.strip('\n').split(" ")
            with open(os.path.join(dir_name, img), "rb") as file_reader:
                img_data = file_reader.read()
            with open(os.path.join(dir_name, label), "rb") as file_reader:
                label_data = file_reader.read()
            data_json = {"file_name": img,
                         "img_data": img_data,
                         "label_name": label,
                         "label_data": label_data,
                         "id": row_num
                         }
            row_num += 1
            data_list.append(data_json)
        except FileNotFoundError:
            continue
    return data_list


def get_multi_bytes_data(file_name, bytes_num=3):
    """
    Return raw data of multi-bytes dataset.

    Args:
        file_name (str): String of multi-bytes dataset's path.
        bytes_num (int): Number of bytes fields.

    Returns:
       List
    """
    if not os.path.exists(file_name):
        raise IOError("map file {} not exists".format(file_name))
    dir_name = os.path.dirname(file_name)
    with open(file_name, "r") as file_reader:
        lines = file_reader.readlines()
    data_list = []
    row_num = 0
    for line in lines:
        try:
            img10_path = line.strip('\n').split(" ")
            img5 = []
            for path in img10_path[:bytes_num]:
                with open(os.path.join(dir_name, path), "rb") as file_reader:
                    img5 += [file_reader.read()]
            data_json = {"image_{}".format(i): img5[i]
                         for i in range(len(img5))}
            data_json.update({"id": row_num})
            row_num += 1
            data_list.append(data_json)
        except FileNotFoundError:
            continue
    return data_list


def get_mkv_data(dir_name):
    """
    Return raw data of Vehicle_and_Person dataset.

    Args:
        dir_name (str): String of Vehicle_and_Person dataset's path.

    Returns:
        List
    """
    if not os.path.isdir(dir_name):
        raise IOError("Directory {} not exists".format(dir_name))
    img_dir = os.path.join(dir_name, "Image")
    label_dir = os.path.join(dir_name, "prelabel")

    data_list = []
    file_list = os.listdir(label_dir)

    index = 1
    for file in file_list:
        if os.path.splitext(file)[1] == '.json':
            file_path = os.path.join(label_dir, file)

            image_name = ''.join([os.path.splitext(file)[0], ".jpg"])
            image_path = os.path.join(img_dir, image_name)

            with open(file_path, "r") as load_f:
                load_dict = json.load(load_f)

            if os.path.exists(image_path):
                with open(image_path, "rb") as file_reader:
                    img = file_reader.read()
                data_json = {"file_name": image_name,
                             "prelabel": str(load_dict),
                             "data": img,
                             "id": index}
                data_list.append(data_json)
            index += 1
    logger.info('{} images are missing'.format(len(file_list) - len(data_list)))
    return data_list


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
                input_ids = np.reshape(np.array(input_), [1, -1])
                input_mask = np.reshape(np.array(mask), [1, -1])
                segment_ids = np.reshape(np.array(segment), [1, -1])
                data = {
                    "label": 1,
                    "id": id_,
                    "rating": float(rating),
                    "input_ids": input_ids,
                    "input_mask": input_mask,
                    "segment_ids": segment_ids
                }
                yield data


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


def inputs(vectors, maxlen=50):
    length = len(vectors)
    if length > maxlen:
        return vectors[0:maxlen], [1] * maxlen, [0] * maxlen
    input_ = vectors + [0] * (maxlen - length)
    mask = [1] * length + [0] * (maxlen - length)
    segment = [0] * maxlen
    return input_, mask, segment
