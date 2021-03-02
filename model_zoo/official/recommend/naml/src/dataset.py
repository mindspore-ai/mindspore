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
"""Dataset loading, creation and processing"""
import re
import os
import random
import math
import pickle

import numpy as np
import mindspore.dataset as ds

ds.config.set_prefetch_size(8)

class MINDPreprocess:
    """
    MIND dataset Preprocess class.
    when training, neg_sample=4, when test, neg_sample=-1
    """
    def __init__(self, config, dataset_path=""):
        self.config = config
        self.dataset_dir = dataset_path
        self.word_dict_path = config['word_dict_path']
        self.category_dict_path = config['category_dict_path']
        self.subcategory_dict_path = config['subcategory_dict_path']
        self.uid2index_path = config['uid2index_path']

        # behaviros config
        self.neg_sample = config['neg_sample']
        self.n_words_title = config['n_words_title']
        self.n_words_abstract = config['n_words_abstract']
        self.n_browsed_news = config['n_browsed_news']

        # news config
        self.n_words_title = config['n_words_title']
        self.n_words_abstract = config['n_words_abstract']
        self._tokenize = 're'

        self._is_init_data = False
        self._init_data()

        self._index = 0
        self._sample_store = []

        self._diff = 0

    def _load_pickle(self, file_path):
        with open(file_path, 'rb') as fp:
            return pickle.load(fp)

    def _init_news(self, news_path):
        """News info initialization."""
        print(f"Start to init news, news path: {news_path}")

        category_dict = self._load_pickle(file_path=self.category_dict_path)
        word_dict = self._load_pickle(file_path=self.word_dict_path)
        subcategory_dict = self._load_pickle(
            file_path=self.subcategory_dict_path)

        self.nid_map_index = {}
        title_list = []
        category_list = []
        subcategory_list = []
        abstract_list = []

        with open(news_path) as file_handler:
            for line in file_handler:
                nid, category, subcategory, title, abstract, _ = line.strip("\n").split('\t')[:6]

                if nid in self.nid_map_index:
                    continue

                self.nid_map_index[nid] = len(self.nid_map_index)
                title_list.append(self._word_tokenize(title))
                category_list.append(category)
                subcategory_list.append(subcategory)
                abstract_list.append(self._word_tokenize(abstract))

        news_len = len(title_list)
        self.news_title_index = np.zeros((news_len, self.n_words_title), dtype=np.int32)
        self.news_abstract_index = np.zeros((news_len, self.n_words_abstract), dtype=np.int32)
        self.news_category_index = np.zeros((news_len, 1), dtype=np.int32)
        self.news_subcategory_index = np.zeros((news_len, 1), dtype=np.int32)
        self.news_ids = np.zeros((news_len, 1), dtype=np.int32)

        for news_index in range(news_len):
            title = title_list[news_index]
            title_index_list = [word_dict.get(word.lower(), 0) for word in title[:self.n_words_title]]
            self.news_title_index[news_index, list(range(len(title_index_list)))] = title_index_list

            abstract = abstract_list[news_index]
            abstract_index_list = [word_dict.get(word.lower(), 0) for word in abstract[:self.n_words_abstract]]
            self.news_abstract_index[news_index, list(range(len(abstract_index_list)))] = abstract_index_list

            category = category_list[news_index]
            self.news_category_index[news_index, 0] = category_dict.get(category, 0)

            subcategory = subcategory_list[news_index]
            self.news_subcategory_index[news_index, 0] = subcategory_dict.get(subcategory, 0)

            self.news_ids[news_index, 0] = news_index

    def _init_behaviors(self, behaviors_path):
        """Behaviors info initialization."""
        print(f"Start to init behaviors, path: {behaviors_path}")

        self.history_list = []
        self.impression_list = []
        self.label_list = []
        self.impression_index_list = []
        self.uid_list = []
        self.poss = []
        self.negs = []
        self.index_map = {}

        self.total_count = 0
        uid2index = self._load_pickle(self.uid2index_path)

        with open(behaviors_path) as file_handler:
            for index, line in enumerate(file_handler):
                uid, _, history, impressions = line.strip("\n").split('\t')[-4:]
                negs = []
                history = [self.nid_map_index[i] for i in history.split()]
                random.shuffle(history)
                history = [0] * (self.n_browsed_news - len(history)) + history[:self.n_browsed_news]
                user_id = uid2index.get(uid, 0)

                if self.neg_sample > 0:
                    for item in impressions.split():
                        nid, label = item.split('-')
                        nid = self.nid_map_index[nid]
                        if label == '1':
                            self.poss.append(nid)
                            self.index_map[self.total_count] = index
                            self.total_count += 1
                        else:
                            negs.append(nid)
                else:
                    nids = []
                    labels = []
                    for item in impressions.split():
                        nid, label = item.split('-')
                        nids.append(self.nid_map_index[nid])
                        labels.append(int(label))
                    self.impression_list.append((np.array(nids, dtype=np.int32), np.array(labels, dtype=np.int32)))
                    self.total_count += 1

                self.history_list.append(history)
                self.negs.append(negs)
                self.uid_list.append(user_id)

    def _init_data(self):
        news_path = os.path.join(self.dataset_dir, 'news.tsv')
        behavior_path = os.path.join(self.dataset_dir, 'behaviors.tsv')
        if not self._is_init_data:
            self._init_news(news_path)
            self._init_behaviors(behavior_path)
            self._is_init_data = True
            print(f'init data end, count: {self.total_count}')

    def _word_tokenize(self, sent):
        """
        Split sentence into word list using regex.
        Args:
            sent (str): Input sentence

        Return:
            list: word list
        """
        pat = re.compile(r"[\w]+|[.,!?;|]")
        if isinstance(sent, str):
            return pat.findall(sent.lower())
        return []

    def __getitem__(self, index):
        uid_index = self.index_map[index]
        if self.neg_sample >= 0:
            negs = self.negs[uid_index]
            nid = self.poss[index]
            random.shuffle(negs)
            neg_samples = (negs + [0] * (self.neg_sample - len(negs))) if self.neg_sample > len(negs) \
                else random.sample(negs, self.neg_sample)
            candidate_samples = [nid] + neg_samples
            labels = [1] + [0] * self.neg_sample

        else:
            candidate_samples, labels = self.preprocess.impression_list[index]
        browsed_samples = self.history_list[uid_index]
        browsed_category = np.array(self.news_category_index[browsed_samples], dtype=np.int32)
        browsed_subcategory = np.array(self.news_subcategory_index[browsed_samples], dtype=np.int32)
        browsed_title = np.array(self.news_title_index[browsed_samples], dtype=np.int32)
        browsed_abstract = np.array(self.news_abstract_index[browsed_samples], dtype=np.int32)
        candidate_category = np.array(self.news_category_index[candidate_samples], dtype=np.int32)
        candidate_subcategory = np.array(self.news_subcategory_index[candidate_samples], dtype=np.int32)
        candidate_title = np.array(self.news_title_index[candidate_samples], dtype=np.int32)
        candidate_abstract = np.array(self.news_abstract_index[candidate_samples], dtype=np.int32)
        labels = np.array(labels, dtype=np.int32)
        return browsed_category, browsed_subcategory, browsed_title, browsed_abstract,\
               candidate_category, candidate_subcategory, candidate_title, candidate_abstract, labels

    @property
    def column_names(self):
        news_column_names = ['category', 'subcategory', 'title', 'abstract']
        column_names = ['browsed_' + item for item in news_column_names]
        column_names += ['candidate_' + item for item in news_column_names]
        column_names += ['labels']
        return column_names

    def __len__(self):
        return self.total_count


class EvalDatasetBase:
    """Base evaluation Datase class."""
    def __init__(self, preprocess: MINDPreprocess):
        self.preprocess = preprocess


class EvalNews(EvalDatasetBase):
    """Generator dataset for all news."""
    def __len__(self):
        return len(self.preprocess.news_title_index)

    def __getitem__(self, index):
        news_id = self.preprocess.news_ids[index]
        title = self.preprocess.news_title_index[index]
        category = self.preprocess.news_category_index[index]
        subcategory = self.preprocess.news_subcategory_index[index]
        abstract = self.preprocess.news_abstract_index[index]
        return news_id.reshape(-1), category.reshape(-1), subcategory.reshape(-1), title.reshape(-1),\
               abstract.reshape(-1)

    @property
    def column_names(self):
        return ['news_id', 'category', 'subcategory', 'title', 'abstract']


class EvalUsers(EvalDatasetBase):
    """Generator dataset for all user."""
    def __len__(self):
        return len(self.preprocess.uid_list)

    def __getitem__(self, index):
        uid = np.array(self.preprocess.uid_list[index], dtype=np.int32)
        history = np.array(self.preprocess.history_list[index], dtype=np.int32)
        return uid, history.reshape(50, 1)

    @property
    def column_names(self):
        return ['uid', 'history']


class EvalCandidateNews(EvalDatasetBase):
    """Generator dataset for all candidate news."""
    @property
    def column_names(self):
        return ['uid', 'candidate_nid', 'labels']

    def __len__(self):
        return self.preprocess.total_count

    def __getitem__(self, index):
        uid = np.array(self.preprocess.uid_list[index], dtype=np.int32)
        nid, label = self.preprocess.impression_list[index]
        return uid, nid, label

class DistributedSampler():
    """
    sampling the dataset.

    Args:
    Returns:
        num_samples, number of samples.
    """
    def __init__(self, preprocess: MINDPreprocess, rank, group_size, shuffle=True, seed=0):
        self.preprocess = preprocess
        self.rank = rank
        self.group_size = group_size
        self.dataset_length = preprocess.total_count
        self.num_samples = int(math.ceil(self.dataset_length * 1.0 / self.group_size))
        self.total_size = self.num_samples * self.group_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            self.seed = (self.seed + 1) & 0xffffffff
            np.random.seed(self.seed)
            indices = np.random.permutation(self.dataset_length).tolist()
        else:
            indices = list(range(len(self.dataset_length)))

        indices += indices[:(self.total_size - len(indices))]
        indices = indices[self.rank::self.group_size]
        return iter(indices)

    def __len__(self):
        return self.num_samples

def create_dataset(mindpreprocess, batch_size=64, rank=0, group_size=1):
    """Get generator dataset when training."""
    sampler = DistributedSampler(mindpreprocess, rank, group_size, shuffle=True)
    dataset = ds.GeneratorDataset(mindpreprocess, mindpreprocess.column_names, sampler=sampler)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


def create_eval_dataset(mindpreprocess, eval_cls, batch_size=64):
    """Get generator dataset when evaluation."""
    eval_instance = eval_cls(mindpreprocess)
    dataset = ds.GeneratorDataset(eval_instance, eval_instance.column_names, shuffle=False)
    if not isinstance(eval_instance, EvalCandidateNews):
        dataset = dataset.batch(batch_size)
    return dataset
