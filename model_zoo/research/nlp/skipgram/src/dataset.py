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
"""
Produce the dataset
"""

import collections
import math
import os
import random
import re

import numpy as np

import mindspore.dataset as ds
from mindspore.mindrecord import FileWriter


def get_corpus(data_dir):
    """Get list of words in the text.

    Args:
        data_dir: data directory.
    Returns:
        list of str words.
    """
    corpus = []
    files = os.listdir(data_dir)
    for filename in files:
        data_path = os.path.join(data_dir, filename)
        if not os.path.isfile(data_path):
            continue
        with open(data_path, 'r') as f:
            text = f.read().strip('\n')
        corpus.extend(del_useless_char(text))
    return corpus


def del_useless_char(corpus):
    """Remove useless character in corpus.

    Args:
        corpus: str.
    Returns:
        list of str words.
    """
    corpus = corpus.strip().lower()
    rule = re.compile("[^a-z^A-Z]")  # English words.
    corpus = rule.sub(' ', corpus)
    corpus = corpus.split()
    return corpus


def get_count_freq(corpus):
    """Construct dictionaries about words and count.

    Args:
        corpus: list of str words.
    Returns:
        list of (word, count) pairs, dictionary that maps word to its frequency.
    """
    word_count = collections.Counter(corpus)
    word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    word_sum = sum([item[1] for item in word_count])
    word2freq = dict()
    for w, c in word_count:
        word2freq[w] = c / word_sum
    return word_count, word2freq


def build_dict(word_count, min_count):
    """Construct dictionaries about words' id and frequency.

    Args:
        word_count: list, with element like (word, count)
        min_count: int, minimum number of word occurrences
    Returns:
        2 dictionary that map word to id, id to word, respectively.
    """
    word2id = dict()
    id2word = dict()
    wid = 0  # word id
    for word, cnt in word_count:
        if cnt < min_count:  # delete word which is seldom used
            continue
        word2id[word] = wid
        id2word[wid] = word
        wid += 1
    return word2id, id2word


def convert_words_to_id(corpus, word2id):
    """Convert list of words to list of corresponding id.

    Args:
        corpus: a list of str words.
        word2id: a dictionary that maps word to id.
    Returns:
        converted corpus, i.e., list of word ids.
    """
    new_corpus = [-1] * len(corpus)
    for i, word in enumerate(corpus):
        if word in word2id:
            new_corpus[i] = word2id[word]
    return new_corpus


def get_sample_table(word_count, word2id, table_size):
    """Get a list about the probability of each word being sampled.

    Args:
        word_count: a list of (word, count) tuples.
        word2id: a dictionary that maps word to id.
        table_size: sample table size
    Returns:
        sample table
    """
    count = [ele[1] for ele in word_count]
    pow_freq = np.array(count)**0.75
    ratios = pow_freq/sum(pow_freq)
    sample_count = np.round(ratios * table_size)
    sample_table = []
    for i, ele in enumerate(word_count):
        w = ele[0]
        if w in word2id:
            wid = word2id[w]
            sample_table += [wid] * int(sample_count[i])
    return sample_table


def subsampling(corpus, word2freq):
    """Randomly delete words with high frequency to reduce redundant information.
    Args:
        corpus: a list of word ids.
        word2freq: a dictionary that maps word id to word frequency.
    Returns:
        a list of words.
    """
    def reserve(w, t=1e-3):
        freq = word2freq[w]
        p = (math.sqrt(freq / t) + 1) * t / freq
        return random.uniform(0, 1) < p
    new_corpus = [word for word in corpus if reserve(word)]
    return new_corpus


def preprocess_data(data_dir, min_count):
    """encapsulated preprocess works"""
    corpus = get_corpus(data_dir)  # list of str words
    word_count, word2freq = get_count_freq(corpus)  # list of (str, int) pairs
    corpus = subsampling(corpus, word2freq)   # remove some most frequent word, such as 'a', 'the'
    word2id, id2word = build_dict(word_count, min_count)  # dictionary doesn't contain seldom used words
    sample_table = get_sample_table(word_count, word2id, int(1e8))
    corpus = convert_words_to_id(corpus, word2id)   # set ID of deleted word as -1
    return corpus, word_count, word2id, id2word, sample_table


def load_eval_data(data_dir):
    """load questions-words.txt
    """
    samples = dict()
    files = os.listdir(data_dir)
    for filename in files:
        data_path = os.path.join(data_dir, filename)
        if not os.path.isfile(data_path):
            continue
        with open(data_path, 'r') as f:
            k = "capital-common-countries"
            samples[k] = list()
            for line in f:
                if ':' in line:
                    strs = line.strip().split(' ')
                    k = strs[1]
                    samples[k] = list()
                else:
                    samples[k].append(line.strip().lower().split(' '))
    return samples


class DataController:
    """encapsulated data operations
    """
    def __init__(self, train_data_dir, ms_dir, min_count, window_size,
                 neg_sample_num, epoch_num, batch_size, rank_size=1, rank_id=0):
        super(DataController, self).__init__()
        self.corpus, self.word_count, self.word2id, self.id2word, self.sample_table = \
            preprocess_data(train_data_dir, min_count)

        self.ms_dir = ms_dir

        self.window_size = window_size
        self.neg_sample_num = neg_sample_num
        self.epoch_num = epoch_num
        self.batch_size = batch_size

        self.rank_size = rank_size
        self.rank_id = rank_id

        self.samples, self.dataset = [], []
        self.cnt, self.sample_id = 0, 0

    def prepare_mindrecord(self):
        """
        prepare mindrecord
        :return: None
        """
        for _ in range(self.epoch_num):
            for i in range(len(self.corpus)):
                center_word = self.corpus[i]
                if center_word == -1:
                    continue
                cur_window_size = random.randint(1, self.window_size)
                pos_range = (max(0, i - cur_window_size), min(len(self.corpus) - 1, i + cur_window_size))
                pos_words = [self.corpus[j] for j in range(pos_range[0], pos_range[1]+1) if j != i]
                for pos_word in pos_words:
                    if pos_word == -1:
                        continue
                    neg_words = self.get_neg_words(center_word, pos_word)
                    self.samples.append([center_word, pos_word, neg_words])

                if len(self.samples) >= int(1e6) * self.batch_size:  # save as mindrecord
                    filename = self.samples_to_dataset_aux()
                    self.convert_to_mindrecord(os.path.join(self.ms_dir, filename), self.dataset)
                    self.cnt += 1
                    self.dataset, self.samples = [], []

        if self.samples:  # save as mindrecord
            filename = self.samples_to_dataset_aux()
            self.convert_to_mindrecord(os.path.join(self.ms_dir, filename), self.dataset)
            self.dataset, self.samples = [], []

    def get_neg_words(self, center_word, pos_word):
        """Get list of negative word ids.
        Args:
            center_word: center word.
            pos_word: positive word.
        """
        neg_words = random.sample(self.sample_table, self.neg_sample_num)
        while center_word in neg_words or pos_word in neg_words or -1 in neg_words:
            neg_words = random.sample(self.sample_table, self.neg_sample_num)
        return np.array(neg_words)

    def samples_to_dataset_aux(self):
        """
        an auxiliary function, which helps convert samples to mindrecord dataset.
        """
        random.shuffle(self.samples)
        for start in range(0, len(self.samples) - self.batch_size, self.batch_size):
            c_words = [self.samples[start + step][0] for step in range(self.batch_size)]
            p_words = [self.samples[start + step][1] for step in range(self.batch_size)]
            n_words = [self.samples[start + step][2] for step in range(self.batch_size)]
            data_json = {"id": self.sample_id,
                         "c_words": np.array(c_words, dtype=np.int32),
                         "p_words": np.array(p_words, dtype=np.int32),
                         "n_words": np.array(n_words, dtype=np.int32)}
            self.dataset.append(data_json)
            self.sample_id += 1
        filename = 'text' + str(self.cnt) + '.mindrecord'
        return filename

    def convert_to_mindrecord(self, ms_data_path, data, shard_num=1):
        schema_json = {"id": {"type": "int64"},
                       "c_words": {"type": "int32", "shape": [-1]},
                       "p_words": {"type": "int32", "shape": [-1]},
                       "n_words": {"type": "int32", "shape": [self.batch_size, self.neg_sample_num]}}
        writer = FileWriter(ms_data_path, shard_num)
        writer.add_schema(schema_json, "w2v_schema")
        writer.add_index(["id"])  # select index fields from schema to accelerate reading.
        writer.write_raw_data(data)
        writer.commit()

    def get_mindrecord_dataset(self, col_list, repeat_count=1):
        """Create dataset (from mindrecord) for training."""
        ms_data_path = []
        files = os.listdir(self.ms_dir)
        for file in files:
            filepath = os.path.join(self.ms_dir, file)
            if os.path.isfile(filepath) and '.db' not in file:
                ms_data_path.append(filepath)
        dataset = ds.MindDataset(ms_data_path, col_list, num_shards=self.rank_size,
                                 shard_id=self.rank_id, num_parallel_workers=4, shuffle=True)
        return dataset

    def get_corpus_len(self):
        return len(self.corpus)

    def get_vocabs_size(self):
        return len(self.word2id)
