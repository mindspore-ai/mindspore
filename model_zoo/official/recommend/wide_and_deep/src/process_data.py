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
Recommendation dataset process
"""

import os
import pickle
import collections

import numpy as np
import pandas as pd

from .model_utils.config import config

TRAIN_LINE_COUNT = 45840617
TEST_LINE_COUNT = 6042135

class RecommendationDatasetStatsDict():
    """create data dict"""
    def __init__(self):
        self.field_size = 39 # value_1-13; cat_1-26;
        self.val_cols = ["val_{}".format(i+1) for i in range(13)]
        self.cat_cols = ["cat_{}".format(i+1) for i in range(26)]

        self.val_min_dict = {col: 0 for col in self.val_cols}
        self.val_max_dict = {col: 0 for col in self.val_cols}
        self.cat_count_dict = {col: collections.defaultdict(int) for col in self.cat_cols}

        self.oov_prefix = "OOV_"

        self.cat2id_dict = {}
        self.cat2id_dict.update({col: i for i, col in enumerate(self.val_cols)})
        self.cat2id_dict.update({self.oov_prefix + col: i + len(self.val_cols) for i, col in enumerate(self.cat_cols)})

    def stats_vals(self, val_list):
        """vals status"""
        assert len(val_list) == len(self.val_cols)
        def map_max_min(i, val):
            key = self.val_cols[i]
            if val != "":
                if float(val) > self.val_max_dict[key]:
                    self.val_max_dict[key] = float(val)
                if float(val) < self.val_min_dict[key]:
                    self.val_min_dict[key] = float(val)

        for i, val in enumerate(val_list):
            map_max_min(i, val)

    def stats_cats(self, cat_list):
        assert len(cat_list) == len(self.cat_cols)
        def map_cat_count(i, cat):
            key = self.cat_cols[i]
            self.cat_count_dict[key][cat] += 1

        for i, cat in enumerate(cat_list):
            map_cat_count(i, cat)

    def save_dict(self, output_path, prefix=""):
        with open(os.path.join(output_path, "{}val_max_dict.pkl".format(prefix)), "wb") as file_wrt:
            pickle.dump(self.val_max_dict, file_wrt)
        with open(os.path.join(output_path, "{}val_min_dict.pkl".format(prefix)), "wb") as file_wrt:
            pickle.dump(self.val_min_dict, file_wrt)
        with open(os.path.join(output_path, "{}cat_count_dict.pkl".format(prefix)), "wb") as file_wrt:
            pickle.dump(self.cat_count_dict, file_wrt)

    def load_dict(self, dict_path, prefix=""):
        with open(os.path.join(dict_path, "{}val_max_dict.pkl".format(prefix)), "rb") as file_wrt:
            self.val_max_dict = pickle.load(file_wrt)
        with open(os.path.join(dict_path, "{}val_min_dict.pkl".format(prefix)), "rb") as file_wrt:
            self.val_min_dict = pickle.load(file_wrt)
        with open(os.path.join(dict_path, "{}cat_count_dict.pkl".format(prefix)), "rb") as file_wrt:
            self.cat_count_dict = pickle.load(file_wrt)
        print("val_max_dict.items()[:50]: {}".format(list(self.val_max_dict.items())))
        print("val_min_dict.items()[:50]: {}".format(list(self.val_min_dict.items())))

    def get_cat2id(self, threshold=100):
        """get cat to id"""
        for key, cat_count_d in self.cat_count_dict.items():
            new_cat_count_d = dict(filter(lambda x: x[1] > threshold, cat_count_d.items()))
            for cat_str, _ in new_cat_count_d.items():
                self.cat2id_dict[key + "_" + cat_str] = len(self.cat2id_dict)
        print("cat2id_dict.size: {}".format(len(self.cat2id_dict)))
        print("cat2id_dict.items()[:50]: {}".format(self.cat2id_dict.items()[:50]))

    def map_cat2id(self, values, cats):
        """map cat to id"""
        def minmax_sclae_value(i, val):
            max_v = float(self.val_max_dict["val_{}".format(i + 1)])
            return float(val) * 1.0 / max_v

        id_list = []
        weight_list = []
        for i, val in enumerate(values):
            if val == "":
                id_list.append(i)
                weight_list.append(0)
            else:
                key = "val_{}".format(i + 1)
                id_list.append(self.cat2id_dict[key])
                weight_list.append(minmax_sclae_value(i, float(val)))

        for i, cat_str in enumerate(cats):
            key = "cat_{}".format(i + 1) + "_" + cat_str
            if key in self.cat2id_dict:
                id_list.append(self.cat2id_dict[key])
            else:
                id_list.append(self.cat2id_dict[self.oov_prefix + "cat_{}".format(i + 1)])
            weight_list.append(1.0)
        return id_list, weight_list


def mkdir_path(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def statsdata(data_file_path, output_path, recommendation_dataset_stats):
    """data status"""
    with open(data_file_path, encoding="utf-8") as file_in:
        errorline_list = []
        count = 0
        for line in file_in:
            count += 1
            line = line.strip("\n")
            items = line.strip("\t")
            if len(items) != 40:
                errorline_list.append(count)
                print("line: {}".format(line))
                continue
            if count % 1000000 == 0:
                print("Have handle {}w lines.".format(count//10000))

            values = items[1:14]
            cats = items[14:]
            assert len(values) == 13, "value.size: {}".format(len(values))
            assert len(cats) == 26, "cat.size: {}".format(len(cats))
            recommendation_dataset_stats.stats_vals(values)
            recommendation_dataset_stats.stats_cats(cats)
    recommendation_dataset_stats.save_dict(output_path)


def add_write(file_path, wr_str):
    with open(file_path, "a", encoding="utf-8") as file_out:
        file_out.write(wr_str + "\n")


def random_split_trans2h5(in_file_path, output_path, recommendation_dataset_stats,
                          part_rows=2000000, test_size=0.1, seed=2020):
    """random split trans2h5"""
    test_size = int(TRAIN_LINE_COUNT * test_size)
    all_indices = [i for i in range(TRAIN_LINE_COUNT)]
    np.random.seed(seed)
    np.random.shuffle(all_indices)
    print("all_indices.size: {}".format(len(all_indices)))
    test_indices_set = set(all_indices[:test_size])
    print("test_indices_set.size: {}".format(len(test_indices_set)))
    print("------" * 10 + "\n" * 2)

    train_feature_file_name = os.path.join(output_path, "train_input_part_{}.h5")
    train_label_file_name = os.path.join(output_path, "train_output_part_{}.h5")
    test_feature_file_name = os.path.join(output_path, "test_input_part_{}.h5")
    test_label_file_name = os.path.join(output_path, "test_input_part_{}.h5")
    train_feature_list = []
    train_label_list = []
    test_feature_list = []
    test_label_list = []
    with open(in_file_path, encoding="utf-8") as file_in:
        count = 0
        train_part_number = 0
        test_part_number = 0
        for i, line in enumerate(file_in):
            count += 1
            if count % 1000000 == 0:
                print("Have handle {}w lines.".format(count // 10000))
            line = line.strip("\n")
            items = line.split("\t")
            if len(items) != 40:
                continue
            label = float(items[0])
            values = items[1:14]
            cats = items[14:]
            assert len(values) == 13, "value.size: {}".format(len(values))
            assert len(cats) == 26, "cat.size: {}".format(len(cats))
            ids, wts = recommendation_dataset_stats.map_cat2id(values, cats)
            if i not in test_indices_set:
                train_feature_list.append(ids + wts)
                train_label_list.append(label)
            else:
                test_feature_list.append(ids + wts)
                test_label_list.append(label)
            if train_label_list and (len(train_label_list) % part_rows == 0):
                pd.DataFrame(np.asarray(train_feature_list)).to_hdf(train_feature_file_name.format(train_part_number),
                                                                    key="fixed")
                pd.DataFrame(np.asarray(train_label_list)).to_hdf(train_label_file_name.format(train_part_number),
                                                                  key="fixed")
                train_feature_list = []
                train_label_list = []
                train_part_number += 1
            if test_label_list and (len(test_label_list) % part_rows == 0):
                pd.DataFrame(np.asarray(test_feature_list)).to_hdf(test_feature_file_name.format(test_part_number),
                                                                   key="fixed")
                pd.DataFrame(np.asarray(test_label_list)).to_hdf(test_label_file_name.format(test_part_number),
                                                                 key="fixed")
                test_feature_list = []
                test_label_list = []
                test_part_number += 1

        if train_label_list:
            pd.DataFrame(np.asarray(train_feature_list)).to_hdf(train_feature_file_name.format(train_part_number),
                                                                key="fixed")
            pd.DataFrame(np.asarray(train_label_list)).to_hdf(train_label_file_name.format(train_part_number),
                                                              key="fixed")
        if test_label_list:
            pd.DataFrame(np.asarray(test_feature_list)).to_hdf(test_feature_file_name.format(test_part_number),
                                                               key="fixed")
            pd.DataFrame(np.asarray(test_label_list)).to_hdf(test_label_file_name.format(test_part_number),
                                                             key="fixed")



if __name__ == "__main__":
    base_path = config.raw_data_path
    recommendation_dataset_stat = RecommendationDatasetStatsDict()
    # step 1, stats the vocab and normalize value
    datafile_path = base_path + "train_small.txt"
    stats_out_path = base_path + "stats_dict/"
    mkdir_path(stats_out_path)
    statsdata(datafile_path, stats_out_path, recommendation_dataset_stat)
    print("------" * 10)
    recommendation_dataset_stat.load_dict(dict_path=stats_out_path, prefix="")
    recommendation_dataset_stat.get_cat2id(threshold=100)
    # step 2, transform data trans2h5; version 2: np.random.shuffle
    infile_path = base_path + "train_small.txt"
    mkdir_path(config.output_path)
    random_split_trans2h5(infile_path, config.output_path, recommendation_dataset_stat,
                          part_rows=2000000, test_size=0.1, seed=2020)
