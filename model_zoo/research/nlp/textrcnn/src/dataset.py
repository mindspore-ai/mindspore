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
"""dataset api"""
import os
from itertools import chain
import numpy as np
import gensim

from mindspore.mindrecord import FileWriter
import mindspore.dataset as ds


# preprocess part
def encode_samples(tokenized_samples, word_to_idx):
    """ encode word to index """
    features = []
    for sample in tokenized_samples:
        feature = []
        for token in sample:
            if token in word_to_idx:
                feature.append(word_to_idx[token])
            else:
                feature.append(0)
        features.append(feature)
    return features


def pad_samples(features, maxlen=50, pad=0):
    """ pad all features to the same length """
    padded_features = []
    for feature in features:
        if len(feature) >= maxlen:
            padded_feature = feature[:maxlen]
        else:
            padded_feature = feature
            while len(padded_feature) < maxlen:
                padded_feature.append(pad)
        padded_features.append(padded_feature)
    return padded_features


def read_imdb(path, seg='train'):
    """ read imdb dataset """
    pos_or_neg = ['pos', 'neg']
    data = []
    for label in pos_or_neg:

        f = os.path.join(path, seg, label)
        rf = open(f, 'r')
        for line in rf:
            line = line.strip()
            if label == 'pos':
                data.append([line, 1])
            elif label == 'neg':
                data.append([line, 0])

    return data


def tokenizer(text):
    return [tok.lower() for tok in text.split(' ')]


def collect_weight(glove_path, vocab, word_to_idx, embed_size):
    """ collect weight """
    vocab_size = len(vocab)
    wvmodel = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(glove_path,
                                                                           'GoogleNews-vectors-negative300.bin'),
                                                              binary=True)
    weight_np = np.zeros((vocab_size + 1, embed_size)).astype(np.float32)

    idx_to_word = {i + 1: word for i, word in enumerate(vocab)}
    idx_to_word[0] = '<unk>'

    for i in range(len(wvmodel.index2word)):
        try:
            index = word_to_idx[wvmodel.index2word[i]]
        except KeyError:
            continue
        weight_np[index, :] = wvmodel.get_vector(
            idx_to_word[word_to_idx[wvmodel.index2word[i]]])
    return weight_np


def preprocess(data_path, glove_path, embed_size):
    """ preprocess the train and test data """
    train_data = read_imdb(data_path, 'train')
    test_data = read_imdb(data_path, 'test')

    train_tokenized = []
    test_tokenized = []
    for review, _ in train_data:
        train_tokenized.append(tokenizer(review))
    for review, _ in test_data:
        test_tokenized.append(tokenizer(review))

    vocab = set(chain(*train_tokenized))
    vocab_size = len(vocab)
    print("vocab_size: ", vocab_size)

    word_to_idx = {word: i + 1 for i, word in enumerate(vocab)}
    word_to_idx['<unk>'] = 0

    train_features = np.array(pad_samples(encode_samples(train_tokenized, word_to_idx))).astype(np.int32)
    train_labels = np.array([score for _, score in train_data]).astype(np.int32)
    test_features = np.array(pad_samples(encode_samples(test_tokenized, word_to_idx))).astype(np.int32)
    test_labels = np.array([score for _, score in test_data]).astype(np.int32)

    weight_np = collect_weight(glove_path, vocab, word_to_idx, embed_size)
    return train_features, train_labels, test_features, test_labels, weight_np, vocab_size


def get_imdb_data(labels_data, features_data):
    data_list = []
    for i, (label, feature) in enumerate(zip(labels_data, features_data)):
        data_json = {"id": i,
                     "label": int(label),
                     "feature": feature.reshape(-1)}
        data_list.append(data_json)
    return data_list


def convert_to_mindrecord(embed_size, data_path, proprocess_path, glove_path):
    """ convert imdb dataset to mindrecord """

    num_shard = 4
    train_features, train_labels, test_features, test_labels, weight_np, _ = \
        preprocess(data_path, glove_path, embed_size)
    np.savetxt(os.path.join(proprocess_path, 'weight.txt'), weight_np)

    print("train_features.shape:", train_features.shape, "train_labels.shape:", train_labels.shape, "weight_np.shape:",
          weight_np.shape, "type:", train_labels.dtype)
    # write mindrecord
    schema_json = {"id": {"type": "int32"},
                   "label": {"type": "int32"},
                   "feature": {"type": "int32", "shape": [-1]}}

    writer = FileWriter(os.path.join(proprocess_path, 'aclImdb_train.mindrecord'), num_shard)
    data = get_imdb_data(train_labels, train_features)
    writer.add_schema(schema_json, "nlp_schema")
    writer.add_index(["id", "label"])
    writer.write_raw_data(data)
    writer.commit()

    writer = FileWriter(os.path.join(proprocess_path, 'aclImdb_test.mindrecord'), num_shard)
    data = get_imdb_data(test_labels, test_features)
    writer.add_schema(schema_json, "nlp_schema")
    writer.add_index(["id", "label"])
    writer.write_raw_data(data)
    writer.commit()


def create_dataset(base_path, batch_size, is_train):
    """Create dataset for training."""
    columns_list = ["feature", "label"]
    num_consumer = 4

    if is_train:
        path = os.path.join(base_path, 'aclImdb_train.mindrecord0')
    else:
        path = os.path.join(base_path, 'aclImdb_test.mindrecord0')

    data_set = ds.MindDataset(path, columns_list, num_consumer)
    ds.config.set_seed(0)
    data_set = data_set.shuffle(buffer_size=data_set.get_dataset_size())
    data_set = data_set.batch(batch_size, drop_remainder=True)
    return data_set
