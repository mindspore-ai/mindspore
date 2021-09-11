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
"""Utility."""
import random
import os
import time
import multiprocessing as mp
import numpy as np
import cv2
from src.config import config2

def resize_short(img, target_size):
    """ resize_short """
    percent = float(target_size) / min(img.shape[0], img.shape[1])
    resized_width = int(round(img.shape[1] * percent))
    resized_height = int(round(img.shape[0] * percent))
    resized = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_LANCZOS4)
    return resized

def functtt(param):
    """ fun """
    sharedlist, s, e = param
    fea, a, b = sharedlist
    ab = np.dot(fea[s:e], fea.T)
    d = a[s:e] + b - 2 * ab
    for i in range(e - s):
        d[i][s + i] += 1e8
    sorted_index = np.argsort(d, 1)[:, :10]
    return sorted_index

def recall_topk_parallel(fea, lab, k):
    """ recall_topk_parallel """
    fea = np.array(fea)
    fea = fea.reshape(fea.shape[0], -1)
    n = np.sqrt(np.sum(fea**2, 1)).reshape(-1, 1)
    fea = fea / n
    a = np.sum(fea**2, 1).reshape(-1, 1)
    b = a.T
    sharedlist = mp.Manager().list()
    sharedlist.append(fea)
    sharedlist.append(a)
    sharedlist.append(b)
    N = 100
    L = fea.shape[0] / N
    params = []
    for i in range(N):
        if i == N - 1:
            s, e = int(i * L), int(fea.shape[0])
        else:
            s, e = int(i * L), int((i + 1) * L)
        params.append([sharedlist, s, e])
    pool = mp.Pool(processes=4)
    sorted_index_list = pool.map(functtt, params)
    pool.close()
    pool.join()
    sorted_index = np.vstack(sorted_index_list)
    res = 0
    for i in range(len(fea)):
        for j in range(k):
            pred = lab[sorted_index[i][j]]
            if lab[i] == pred:
                res += 1.0
                break
    res = res / len(fea)
    return res

class GetDatasetGenerator_eval():
    """ GetDatasetGenerator_eval"""
    def __init__(self, data_dir, train_list):
        self.DATA_DIR = data_dir
        self.TRAIN_LIST = train_list
        train_image_list = []
        TRAIN_LISTS = open(self.TRAIN_LIST, "r").readlines()
        for _, item in enumerate(TRAIN_LISTS):
            items = item.strip().split()
            if items[0] == 'image_id':
                continue
            path = items[0]
            label = int(items[1]) - 1
            train_image_list.append((path, label))
        r = random.random
        random.seed(int(time.time()))
        random.shuffle(train_image_list, random=r)
        self.__data = [i[0] for i in train_image_list]
        self.__label = [i[1] for i in train_image_list]
    def __getitem__(self, index):
        self.__img = cv2.imread(os.path.join(self.DATA_DIR, self.__data[index]))
        self.__img = resize_short(self.__img, 224)
        item = (self.__img, self.__label[index])
        return item
    def __len__(self):
        return len(self.__data)

class GetDatasetGenerator_softmax():
    """ GetDatasetGenerator_softmax """
    def __init__(self, data_dir, train_list):
        self.DATA_DIR = data_dir
        self.TRAIN_LIST = train_list
        train_image_list = []
        TRAIN_LISTS = open(self.TRAIN_LIST, "r").readlines()
        for _, item in enumerate(TRAIN_LISTS):
            items = item.strip().split()
            if items[0] == 'image_id':
                continue
            path = items[0]
            label = int(items[1]) - 1
            train_image_list.append((path, label))
        r = random.random
        random.seed(int(time.time()))
        random.shuffle(train_image_list, random=r)
        self.__data = [i[0] for i in train_image_list]
        self.__label = [i[1] for i in train_image_list]
    def __getitem__(self, index):
        self.__img = cv2.imread(os.path.join(self.DATA_DIR, self.__data[index]))
        item = (self.__img, self.__label[index])
        return item
    def __len__(self):
        return len(self.__data)

class GetDatasetGenerator_triplet():
    """ GetDatasetGenerator_triplet """
    def __init__(self, data_dir, train_list):
        self.DATA_DIR = data_dir
        self.TRAIN_LIST = train_list
        train_data = {}
        train_image_list_tiplet = []
        TRAIN_LISTS = open(self.TRAIN_LIST, "r").readlines()
        count = 0
        for _, item in enumerate(TRAIN_LISTS):
            items = item.strip().split()
            if items[0] == 'image_id':
                continue
            path = items[0]
            label = int(items[1]) - 1
            if label not in train_data:
                train_data[label] = []
            train_data[label].append(path)
        #shuffle
        r = random.random
        random.seed(int(time.time()))
        #data generates
        labs = list(train_data.keys())
        lab_num = len(labs)
        ind = list(range(0, lab_num))
        total_count = len(TRAIN_LISTS)
        while True:
            random.shuffle(ind, random=r)
            ind_pos, ind_neg = ind[:2]
            lab_pos = labs[ind_pos]
            pos_data_list = train_data[lab_pos]
            data_ind = list(range(0, len(pos_data_list)))
            random.shuffle(data_ind, random=r)
            anchor_ind, pos_ind = data_ind[:2]
            lab_neg = labs[ind_neg]
            neg_data_list = train_data[lab_neg]
            neg_ind = random.randint(0, len(neg_data_list) - 1)
            anchor_path = self.DATA_DIR + pos_data_list[anchor_ind]
            train_image_list_tiplet.append((anchor_path, lab_pos))
            pos_path = self.DATA_DIR + pos_data_list[pos_ind]
            train_image_list_tiplet.append((pos_path, lab_pos))
            neg_path = self.DATA_DIR + neg_data_list[neg_ind]
            train_image_list_tiplet.append((neg_path, lab_neg))
            count += 3
            if count >= total_count:
                break
        self.__data = [i[0] for i in train_image_list_tiplet]
        self.__label = [i[1] for i in train_image_list_tiplet]

    def __getitem__(self, index):
        img = cv2.imread(self.__data[index])
        item = (img, self.__label[index])
        return item
    def __len__(self):
        return len(self.__data)

class GetDatasetGenerator_quadruplet():
    """GetDatasetGenerator_quadruplet."""
    def __init__(self, data_dir, train_list):
        self.DATA_DIR = data_dir
        self.TRAIN_LIST = train_list
        self.batch_size = config2.batch_size
        samples_each_class = 2
        assert self.batch_size % samples_each_class == 0
        class_num = self.batch_size // samples_each_class
        train_data = {}
        train_image_list_quadruplet = []
        TRAIN_LISTS = open(self.TRAIN_LIST, "r").readlines()
        count = 0
        for _, item in enumerate(TRAIN_LISTS):
            items = item.strip().split()
            if items[0] == 'image_id':
                continue
            path = items[0]
            label = int(items[1]) - 1
            if label not in train_data:
                train_data[label] = []
            train_data[label].append(path)
        #shuffle
        r = random.random
        random.seed(int(time.time()))
        #data generates
        labs = list(train_data.keys())
        lab_num = len(labs)
        ind = list(range(0, lab_num))
        total_count = len(TRAIN_LISTS)
        while True:
            random.shuffle(ind, random=r)
            ind_sample = ind[:class_num]
            for ind_i in ind_sample:
                lab = labs[ind_i]
                data_list = train_data[lab]
                data_ind = list(range(0, len(data_list)))
                random.shuffle(data_ind, random=r)
                anchor_ind = data_ind[:samples_each_class]
                for anchor_ind_i in anchor_ind:
                    anchor_path = self.DATA_DIR + data_list[anchor_ind_i]
                    train_image_list_quadruplet.append((anchor_path, lab))
                    count += 1
            if count >= total_count:
                break

        self.__data = [i[0] for i in train_image_list_quadruplet]
        self.__label = [i[1] for i in train_image_list_quadruplet]

    def __getitem__(self, index):
        img = cv2.imread(self.__data[index])
        item = (img, self.__label[index])
        return item
    def __len__(self):
        return len(self.__data)
