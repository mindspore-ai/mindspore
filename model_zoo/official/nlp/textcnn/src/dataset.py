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
Data operations, will be used in train.py and eval.py
"""
import os
import math
import random
import codecs
from pathlib import Path
import numpy as np
import pandas as pd
import mindspore.dataset as ds



class Generator():
    def __init__(self, input_list):
        self.input_list = input_list

    def __getitem__(self, item):
        return np.array(self.input_list[item][0], dtype=np.int32), np.array(self.input_list[item][1], dtype=np.int32)

    def __len__(self):
        return len(self.input_list)


class DataProcessor:
    """
    preprocess dataset
    """
    def get_dict_len(self):
        """
        get number of different words in the whole dataset
        """
        if self.doConvert:
            return len(self.Vocab)
        return -1

    def collect_weight(self, glove_path, embed_size):
        """ collect weight """
        vocab_size = self.get_dict_len()
        embedding_index = {}
        f = open(glove_path)
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.array(values[1:], dtype='float32')
            embedding_index[word] = vec
        weight_np = np.zeros((vocab_size, embed_size)).astype(np.float32)

        for word, vec in embedding_index.items():
            try:
                index = self.Vocab[word]
            except KeyError:
                continue
            weight_np[index, :] = vec
        return weight_np

    def create_train_dataset(self, epoch_size, batch_size, collect_weight=False, glove_path='', embed_size=50):
        if collect_weight:
            weight_np = self.collect_weight(glove_path, embed_size)
            np.savetxt('./weight.txt', weight_np)
        dataset = ds.GeneratorDataset(source=Generator(input_list=self.train),
                                      column_names=["data", "label"], shuffle=False)
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
        return dataset

    def create_test_dataset(self, batch_size):
        dataset = ds.GeneratorDataset(source=Generator(input_list=self.test),
                                      column_names=["data", "label"], shuffle=False)
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
        return dataset

class MovieReview(DataProcessor):
    """
    preprocess MovieReview dataset
    """
    def __init__(self, root_dir, maxlen, split):
        """
        input:
            root_dir: the root directory path of the MR dataset
            maxlen: set the max length of the sentence
            split: set the ratio of training set to testing set
            rank: the logic order of the worker
            size: the worker num
        """
        self.path = root_dir
        self.feelMap = {
            'neg': 0,
            'pos': 1
        }
        self.files = []
        self.doConvert = False
        mypath = Path(self.path)

        if not mypath.exists() or not mypath.is_dir():
            print("please check the root_dir!")
            raise ValueError

        # walk through the root_dir
        for root, _, filename in os.walk(self.path):
            for each in filename:
                self.files.append(os.path.join(root, each))
            break

        # check whether get two files
        if len(self.files) != 2:
            print("There are {} files in the root_dir".format(len(self.files)))
            raise ValueError

        # begin to read data
        self.word_num = 0
        self.maxlen = 0
        self.minlen = float("inf")
        self.maxlen = float("-inf")
        self.Pos = []
        self.Neg = []
        for filename in self.files:
            f = codecs.open(filename, 'r')
            ff = f.read()
            file_object = codecs.open(filename, 'w', 'utf-8')
            file_object.write(ff)
            self.read_data(filename)
        self.PosNeg = self.Pos + self.Neg
        self.text2vec(maxlen=maxlen)
        self.split_dataset(split=split)

    def read_data(self, filePath):
        """
        read text into memory

        input:
            filePath: the path where the data is stored in
        """
        with open(filePath, 'r') as f:
            for sentence in f.readlines():
                sentence = sentence.replace('\n', '')\
                                    .replace('"', '')\
                                    .replace('\'', '')\
                                    .replace('.', '')\
                                    .replace(',', '')\
                                    .replace('[', '')\
                                    .replace(']', '')\
                                    .replace('(', '')\
                                    .replace(')', '')\
                                    .replace(':', '')\
                                    .replace('--', '')\
                                    .replace('-', '')\
                                    .replace('\\', '')\
                                    .replace('0', '')\
                                    .replace('1', '')\
                                    .replace('2', '')\
                                    .replace('3', '')\
                                    .replace('4', '')\
                                    .replace('5', '')\
                                    .replace('6', '')\
                                    .replace('7', '')\
                                    .replace('8', '')\
                                    .replace('9', '')\
                                    .replace('`', '')\
                                    .replace('=', '')\
                                    .replace('$', '')\
                                    .replace('/', '')\
                                    .replace('*', '')\
                                    .replace(';', '')\
                                    .replace('<b>', '')\
                                    .replace('%', '')
                sentence = sentence.split(' ')
                sentence = list(filter(lambda x: x, sentence))
                if sentence:
                    self.word_num += len(sentence)
                    self.maxlen = self.maxlen if self.maxlen >= len(sentence) else len(sentence)
                    self.minlen = self.minlen if self.minlen <= len(sentence) else len(sentence)
                    if 'pos' in filePath:
                        self.Pos.append([sentence, self.feelMap['pos']])
                    else:
                        self.Neg.append([sentence, self.feelMap['neg']])

    def text2vec(self, maxlen):
        """
        convert the sentence into a vector in an int type

        input:
            maxlen: max length of the sentence
        """
        # Vocab = {word : index}
        self.Vocab = dict()

        for SentenceLabel in self.Pos+self.Neg:
            vector = [0]*maxlen
            for index, word in enumerate(SentenceLabel[0]):
                if index >= maxlen:
                    break
                if word not in self.Vocab.keys():
                    self.Vocab[word] = len(self.Vocab)
                    vector[index] = len(self.Vocab) - 1
                else:
                    vector[index] = self.Vocab[word]
            SentenceLabel[0] = vector
        self.doConvert = True

    def split_dataset(self, split):
        """
        split the dataset into training set and test set
        input:
            split: the ratio of training set to test set
            rank: logic order
            size: device num
        """
        trunk_pos_size = math.ceil((1-split)*len(self.Pos))
        trunk_neg_size = math.ceil((1-split)*len(self.Neg))
        trunk_num = int(1/(1-split))
        pos_temp = list()
        neg_temp = list()
        for index in range(trunk_num):
            pos_temp.append(self.Pos[index*trunk_pos_size:(index+1)*trunk_pos_size])
            neg_temp.append(self.Neg[index*trunk_neg_size:(index+1)*trunk_neg_size])
        self.test = pos_temp.pop(2)+neg_temp.pop(2)
        self.train = [i for item in pos_temp+neg_temp for i in item]

        random.shuffle(self.train)

class Subjectivity(DataProcessor):
    """
    preprocess Subjectivity dataset
    """
    def __init__(self, root_dir, maxlen, split):
        self.path = root_dir
        self.feelMap = {
            'neg': 0,
            'pos': 1
        }
        self.files = []
        self.doConvert = False
        mypath = Path(self.path)

        if not mypath.exists() or not mypath.is_dir():
            print("please check the root_dir!")
            raise ValueError

        # walk through the root_dir
        for root, _, filename in os.walk(self.path):
            for each in filename:
                self.files.append(os.path.join(root, each))
            break

        # begin to read data
        self.word_num = 0
        self.maxlen = 0
        self.minlen = float("inf")
        self.maxlen = float("-inf")
        self.Pos = []
        self.Neg = []
        for filename in self.files:
            self.read_data(filename)
        self.PosNeg = self.Pos + self.Neg
        self.text2vec(maxlen=maxlen)
        self.split_dataset(split=split)

    def read_data(self, filePath):
        """
        read text into memory

        input:
            filePath: the path where the data is stored in
        """
        with open(filePath, 'r', encoding="ISO-8859-1") as f:
            for sentence in f.readlines():
                sentence = sentence.replace('\n', '')\
                                    .replace('"', '')\
                                    .replace('\'', '')\
                                    .replace('.', '')\
                                    .replace(',', '')\
                                    .replace('[', '')\
                                    .replace(']', '')\
                                    .replace('(', '')\
                                    .replace(')', '')\
                                    .replace(':', '')\
                                    .replace('--', '')\
                                    .replace('-', '')\
                                    .replace('\\', '')\
                                    .replace('0', '')\
                                    .replace('1', '')\
                                    .replace('2', '')\
                                    .replace('3', '')\
                                    .replace('4', '')\
                                    .replace('5', '')\
                                    .replace('6', '')\
                                    .replace('7', '')\
                                    .replace('8', '')\
                                    .replace('9', '')\
                                    .replace('`', '')\
                                    .replace('=', '')\
                                    .replace('$', '')\
                                    .replace('/', '')\
                                    .replace('*', '')\
                                    .replace(';', '')\
                                    .replace('<b>', '')\
                                    .replace('%', '')
                sentence = sentence.split(' ')
                sentence = list(filter(lambda x: x, sentence))
                if sentence:
                    self.word_num += len(sentence)
                    self.maxlen = self.maxlen if self.maxlen >= len(sentence) else len(sentence)
                    self.minlen = self.minlen if self.minlen <= len(sentence) else len(sentence)
                    if 'quote' in filePath:
                        self.Pos.append([sentence, self.feelMap['pos']])
                    elif 'plot' in filePath:
                        self.Neg.append([sentence, self.feelMap['neg']])

    def text2vec(self, maxlen):
        """
        convert the sentence into a vector in an int type

        input:
            maxlen: max length of the sentence
        """
        # Vocab = {word : index}
        self.Vocab = dict()

        for SentenceLabel in self.Pos+self.Neg:
            vector = [0]*maxlen
            for index, word in enumerate(SentenceLabel[0]):
                if index >= maxlen:
                    break
                if word not in self.Vocab.keys():
                    self.Vocab[word] = len(self.Vocab)
                    vector[index] = len(self.Vocab) - 1
                else:
                    vector[index] = self.Vocab[word]
            SentenceLabel[0] = vector
        self.doConvert = True

    def split_dataset(self, split):
        """
        split the dataset into training set and test set
        input:
            split: the ratio of training set to test set
            rank: logic order
            size: device num
        """
        trunk_pos_size = math.ceil((1-split)*len(self.Pos))
        trunk_neg_size = math.ceil((1-split)*len(self.Neg))
        trunk_num = int(1/(1-split))
        pos_temp = list()
        neg_temp = list()
        for index in range(trunk_num):
            pos_temp.append(self.Pos[index*trunk_pos_size:(index+1)*trunk_pos_size])
            neg_temp.append(self.Neg[index*trunk_neg_size:(index+1)*trunk_neg_size])
        self.test = pos_temp.pop(2)+neg_temp.pop(2)
        self.train = [i for item in pos_temp+neg_temp for i in item]

        random.shuffle(self.train)

class SST2(DataProcessor):
    """
    preprocess SST2 dataset
    """
    def __init__(self, root_dir, maxlen, split):
        self.path = root_dir
        self.files = []
        self.train = []
        self.test = []
        self.doConvert = False
        mypath = Path(self.path)

        if not mypath.exists() or not mypath.is_dir():
            print("please check the root_dir!")
            raise ValueError

        # walk through the root_dir
        for root, _, filename in os.walk(self.path):
            for each in filename:
                self.files.append(os.path.join(root, each))
            break

        # begin to read data
        self.word_num = 0
        self.maxlen = 0
        self.minlen = float("inf")
        self.maxlen = float("-inf")
        for filename in self.files:
            if 'train' in filename or 'dev' in filename:
                f = codecs.open(filename, 'r')
                ff = f.read()
                file_object = codecs.open(filename, 'w', 'utf-8')
                file_object.write(ff)
                self.read_data(filename)
        self.text2vec(maxlen=maxlen)
        self.split_dataset(split=split)

    def read_data(self, filePath):
        """
        read text into memory

        input:
            filePath: the path where the data is stored in
        """
        df = pd.read_csv(filePath, delimiter='\t')
        for sentence, label in zip(df['sentence'], df['label']):
            sentence = sentence.replace('\n', '')\
                                .replace('"', '')\
                                .replace('\'', '')\
                                .replace('.', '')\
                                .replace(',', '')\
                                .replace('[', '')\
                                .replace(']', '')\
                                .replace('(', '')\
                                .replace(')', '')\
                                .replace(':', '')\
                                .replace('--', '')\
                                .replace('-', '')\
                                .replace('\\', '')\
                                .replace('0', '')\
                                .replace('1', '')\
                                .replace('2', '')\
                                .replace('3', '')\
                                .replace('4', '')\
                                .replace('5', '')\
                                .replace('6', '')\
                                .replace('7', '')\
                                .replace('8', '')\
                                .replace('9', '')\
                                .replace('`', '')\
                                .replace('=', '')\
                                .replace('$', '')\
                                .replace('/', '')\
                                .replace('*', '')\
                                .replace(';', '')\
                                .replace('<b>', '')\
                                .replace('%', '')
            sentence = sentence.split(' ')
            sentence = list(filter(lambda x: x, sentence))
            if sentence:
                self.word_num += len(sentence)
                self.maxlen = self.maxlen if self.maxlen >= len(sentence) else len(sentence)
                self.minlen = self.minlen if self.minlen <= len(sentence) else len(sentence)
                if 'train' in filePath:
                    self.train.append([sentence, label])
                elif 'dev' in filePath:
                    self.test.append([sentence, label])

    def text2vec(self, maxlen):
        """
        convert the sentence into a vector in an int type

        input:
            maxlen: max length of the sentence
        """
        # Vocab = {word : index}
        self.Vocab = dict()

        for SentenceLabel in self.train+self.test:
            vector = [0]*maxlen
            for index, word in enumerate(SentenceLabel[0]):
                if index >= maxlen:
                    break
                if word not in self.Vocab.keys():
                    self.Vocab[word] = len(self.Vocab)
                    vector[index] = len(self.Vocab) - 1
                else:
                    vector[index] = self.Vocab[word]
            SentenceLabel[0] = vector
        self.doConvert = True

    def split_dataset(self, split):
        """
        split the dataset into training set and test set
        input:
            split: the ratio of training set to test set
            rank: logic order
            size: device num
        """
        random.shuffle(self.train)
