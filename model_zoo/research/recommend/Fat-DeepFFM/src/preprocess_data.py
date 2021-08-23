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
# ===========================================================================
"""Download raw data and preprocessed data."""
import argparse
import collections
import os
import pickle

import numpy as np
from mindspore.dataset import context
from mindspore.mindrecord import FileWriter


class StatsDict:
    """preprocessed data"""

    def __init__(self, field_size, dense_dim, slot_dim, skip_id_convert):
        self.field_size = field_size  # 40
        self.dense_dim = dense_dim  # 13
        self.slot_dim = slot_dim  # 26
        self.skip_id_convert = bool(skip_id_convert)

        self.val_cols = ["val_{}".format(i + 1) for i in range(self.dense_dim)]
        self.cat_cols = ["cat_{}".format(i + 1) for i in range(self.slot_dim)]

        self.val_min_dict = {col: 0 for col in self.val_cols}
        self.val_max_dict = {col: 0 for col in self.val_cols}

        self.cat_count_dict = {col: collections.defaultdict(int) for col in self.cat_cols}

        self.oov_prefix = "OOV"

        self.cat2id_dict = {}
        self.cat2id_dict.update({col: i for i, col in enumerate(self.val_cols)})
        self.cat2id_dict.update(
            {self.oov_prefix + col: i + len(self.val_cols) for i, col in enumerate(self.cat_cols)})

    def stats_vals(self, val_list):
        """Handling weights column"""
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
        """Handling cats column"""

        assert len(cat_list) == len(self.cat_cols)

        def map_cat_count(i, cat):
            key = self.cat_cols[i]
            self.cat_count_dict[key][cat] += 1

        for i, cat in enumerate(cat_list):
            map_cat_count(i, cat)

    def save_dict(self, dict_path, prefix=""):
        with open(os.path.join(dict_path, "{}val_max_dict.pkl".format(prefix)), "wb") as file_wrt:
            pickle.dump(self.val_max_dict, file_wrt)
        with open(os.path.join(dict_path, "{}val_min_dict.pkl".format(prefix)), "wb") as file_wrt:
            pickle.dump(self.val_min_dict, file_wrt)
        with open(os.path.join(dict_path, "{}cat_count_dict.pkl".format(prefix)), "wb") as file_wrt:
            pickle.dump(self.cat_count_dict, file_wrt)

    def load_dict(self, dict_path, prefix=""):
        with open(os.path.join(dict_path, "{}val_max_dict.pkl".format(prefix)), "rb") as file_wrt:
            self.val_max_dict = pickle.load(file_wrt)
        with open(os.path.join(dict_path, "{}val_min_dict.pkl".format(prefix)), "rb") as file_wrt:
            self.val_min_dict = pickle.load(file_wrt)
        with open(os.path.join(dict_path, "{}cat_count_dict.pkl".format(prefix)), "rb") as file_wrt:
            self.cat_count_dict = pickle.load(file_wrt)
        print("val_max_dict.items()[:50]:{}".format(list(self.val_max_dict.items())))
        print("val_min_dict.items()[:50]:{}".format(list(self.val_min_dict.items())))

    def get_cat2id(self, threshold=100):
        for key, cat_count_d in self.cat_count_dict.items():
            new_cat_count_d = dict(filter(lambda x: x[1] > threshold, cat_count_d.items()))
            for cat_str, _ in new_cat_count_d.items():
                self.cat2id_dict[key + "_" + cat_str] = len(self.cat2id_dict)
        print("cat2id_dict.size:{}".format(len(self.cat2id_dict)))
        print("cat2id.dict.items()[:50]:{}".format(list(self.cat2id_dict.items())[:50]))

    def map_cat2id(self, values, cats):
        """Cat to id"""

        def minmax_scale_value(i, val):
            max_v = float(self.val_max_dict["val_{}".format(i + 1)])
            return float(val) * 1.0 / max_v

        dense_list = []
        spare_list = []
        for i, val in enumerate(values):
            if val == "":
                dense_list.append(0)
            else:
                dense_list.append(minmax_scale_value(i, float(val)))

        for i, cat_str in enumerate(cats):
            key = "cat_{}".format(i + 1) + "_" + cat_str
            if key in self.cat2id_dict:
                spare_list.append(self.cat2id_dict[key])
            else:
                spare_list.append(self.cat2id_dict[self.oov_prefix + "cat_{}".format(i + 1)])
        return dense_list, spare_list


def mkdir_path(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def statsdata(file_path, dict_output_path, recommendation_dataset_stats_dict, dense_dim=13, slot_dim=26):
    """Preprocess data and save data"""
    with open(file_path, encoding="utf-8") as file_in:
        errorline_list = []
        count = 0
        for line in file_in:
            count += 1
            line = line.strip("\n")
            items = line.split("\t")
            if len(items) != (dense_dim + slot_dim + 1):
                errorline_list.append(count)
                print("Found line length: {}, suppose to be {}, the line is {}".format(len(items),
                                                                                       dense_dim + slot_dim + 1, line))
                continue
            if count % 1000000 == 0:
                print("Have handled {}w lines.".format(count // 10000))
            values = items[1: dense_dim + 1]
            cats = items[dense_dim + 1:]

            assert len(values) == dense_dim, "values.size: {}".format(len(values))
            assert len(cats) == slot_dim, "cats.size: {}".format(len(cats))
            recommendation_dataset_stats_dict.stats_vals(values)
            recommendation_dataset_stats_dict.stats_cats(cats)
    recommendation_dataset_stats_dict.save_dict(dict_output_path)


def random_split_trans2mindrecord(input_file_path, output_file_path, recommendation_dataset_stats_dict,
                                  part_rows=100000, line_per_sample=1000, train_line_count=None,
                                  test_size=0.1, seed=2020, dense_dim=13, slot_dim=26):
    """Random split data and save mindrecord"""
    if train_line_count is None:
        raise ValueError("Please provide training file line count")
    test_size = int(train_line_count * test_size)
    all_indices = [i for i in range(train_line_count)]
    np.random.seed(seed)
    np.random.shuffle(all_indices)
    print("all_indices.size:{}".format(len(all_indices)))
    test_indices_set = set(all_indices[:test_size])
    print("test_indices_set.size:{}".format(len(test_indices_set)))
    print("-----------------------" * 10 + "\n" * 2)

    train_data_list = []
    test_data_list = []
    cats_list = []
    dense_list = []
    label_list = []

    writer_train = FileWriter(os.path.join(output_file_path, "train_input_part.mindrecord"), 21)
    writer_test = FileWriter(os.path.join(output_file_path, "test_input_part.mindrecord"), 3)

    schema = {"label": {"type": "float32", "shape": [-1]}, "num_vals": {"type": "float32", "shape": [-1]},
              "cats_vals": {"type": "int32", "shape": [-1]}}
    writer_train.add_schema(schema, "CRITEO_TRAIN")
    writer_test.add_schema(schema, "CRITEO_TEST")

    with open(input_file_path, encoding="utf-8") as file_in:
        items_error_size_lineCount = []
        count = 0
        train_part_number = 0
        test_part_number = 0
        for i, line in enumerate(file_in):
            count += 1
            if count % 1000000 == 0:
                print("Have handle {}w lines.".format(count // 10000))
            line = line.strip("\n")
            items = line.split("\t")
            if len(items) != (1 + dense_dim + slot_dim):
                items_error_size_lineCount.append(i)
                continue
            label = float(items[0])
            values = items[1:1 + dense_dim]
            cats = items[1 + dense_dim:]

            assert len(values) == dense_dim, "values.size: {}".format(len(values))
            assert len(cats) == slot_dim, "cats.size: {}".format(len(cats))

            dense, cats = recommendation_dataset_stats_dict.map_cat2id(values, cats)

            dense_list.extend(dense)
            cats_list.extend(cats)
            label_list.append(label)

            if count % line_per_sample == 0:
                if i not in test_indices_set:
                    train_data_list.append({"cats_vals": np.array(cats_list, dtype=np.int32),
                                            "num_vals": np.array(dense_list, dtype=np.float32),
                                            "label": np.array(label_list, dtype=np.float32)
                                            })
                else:
                    test_data_list.append({"cats_vals": np.array(cats_list, dtype=np.int32),
                                           "num_vals": np.array(dense_list, dtype=np.float32),
                                           "label": np.array(label_list, dtype=np.float32)
                                           })
                if train_data_list and len(train_data_list) % part_rows == 0:
                    writer_train.write_raw_data(train_data_list)
                    train_data_list.clear()
                    train_part_number += 1

                if test_data_list and len(test_data_list) % part_rows == 0:
                    writer_test.write_raw_data(test_data_list)
                    test_data_list.clear()
                    test_part_number += 1

                cats_list.clear()
                dense_list.clear()
                label_list.clear()

        if train_data_list:
            writer_train.write_raw_data(train_data_list)
        if test_data_list:
            writer_test.write_raw_data(test_data_list)
    writer_train.commit()
    writer_test.commit()

    print("-------------" * 10)
    print("items_error_size_lineCount.size(): {}.".format(len(items_error_size_lineCount)))
    print("-------------" * 10)
    np.save("items_error_size_lineCount.npy", items_error_size_lineCount)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Recommendation dataset")
    parser.add_argument("--data_path", type=str, default="./data/",
                        help='The path of the data file')
    parser.add_argument("--dense_dim", type=int, default=13, help='The number of your continues fields')
    parser.add_argument("--slot_dim", type=int, default=26,
                        help='The number of your sparse fields, it can also be called catelogy features.')
    parser.add_argument("--threshold", type=int, default=100,
                        help='Word frequency below this will be regarded as OOV. It aims to reduce the vocab size')
    parser.add_argument("--train_line_count", type=int, default=45840617, help='The number of examples in your dataset')
    parser.add_argument("--skip_id_convert", type=int, default=0, choices=[0, 1],
                        help='Skip the id convert, regarding the original id as the final id.')

    args, _ = parser.parse_known_args()
    data_path = args.data_path
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend",
                        device_id=device_id)
    target_field_size = args.dense_dim + args.slot_dim
    stats = StatsDict(field_size=target_field_size, dense_dim=args.dense_dim, slot_dim=args.slot_dim,
                      skip_id_convert=args.skip_id_convert)
    data_file_path = data_path + "train.txt"
    stats_output_path = data_path + "stats_dict/"
    mkdir_path(stats_output_path)
    statsdata(data_file_path, stats_output_path, stats, dense_dim=args.dense_dim, slot_dim=args.slot_dim)

    stats.load_dict(dict_path=stats_output_path, prefix="")
    stats.get_cat2id(threshold=args.threshold)

    output_path = data_path + "mindrecord/"
    mkdir_path(output_path)
    random_split_trans2mindrecord(data_file_path, output_path, stats, part_rows=100000,
                                  train_line_count=45840617, line_per_sample=1000,
                                  test_size=0.1, seed=2020, dense_dim=13, slot_dim=26)
