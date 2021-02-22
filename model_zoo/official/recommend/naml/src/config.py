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
#===================================================
"""
network config setting, will be used in train.py and eval.py
The parameter is usually a multiple of 16 in order to adapt to Ascend.
"""

class MINDlarge:
    """MIND large config."""
    n_categories = 19
    n_sub_categories = 286
    n_words = 74308
    epochs = 1
    lr = 0.001
    print_times = 1000
    embedding_file = "{}/MINDlarge_utils/embedding_all.npy"
    word_dict_path = "{}/MINDlarge_utils/word_dict_all.pkl"
    category_dict_path = "{}/MINDlarge_utils/vert_dict.pkl"
    subcategory_dict_path = "{}/MINDlarge_utils/subvert_dict.pkl"
    uid2index_path = "{}/MINDlarge_utils/uid2index.pkl"
    train_dataset_path = "{}/MINDlarge_train"
    eval_dataset_path = "{}/MINDlarge_dev"

class MINDsmall:
    """MIND small config."""
    n_categories = 19
    n_sub_categories = 271
    n_words = 60993
    epochs = 3
    lr = 0.0005
    print_times = 500
    embedding_file = "{}/MINDsmall_utils/embedding_all.npy"
    word_dict_path = "{}/MINDsmall_utils/word_dict_all.pkl"
    category_dict_path = "{}/MINDsmall_utils/vert_dict.pkl"
    subcategory_dict_path = "{}/MINDsmall_utils/subvert_dict.pkl"
    uid2index_path = "{}/MINDsmall_utils/uid2index.pkl"
    train_dataset_path = "{}/MINDsmall_train"
    eval_dataset_path = "{}/MINDsmall_dev"

class MINDdemo:
    """MIND small config."""
    n_categories = 18
    n_sub_categories = 237
    n_words = 41059
    epochs = 10
    lr = 0.0005
    print_times = 100
    embedding_file = "{}/MINDdemo_utils/embedding_all.npy"
    word_dict_path = "{}/MINDdemo_utils/word_dict_all.pkl"
    category_dict_path = "{}/MINDdemo_utils/vert_dict.pkl"
    subcategory_dict_path = "{}/MINDdemo_utils/subvert_dict.pkl"
    uid2index_path = "{}/MINDdemo_utils/uid2index.pkl"
    train_dataset_path = "{}/MINDdemo_train"
    eval_dataset_path = "{}/MINDdemo_dev"

def get_dataset_config(dataset):
    if dataset == "large":
        return MINDlarge
    if dataset == "small":
        return MINDsmall
    if dataset == "demo":
        return MINDdemo
    raise ValueError(f"Only support MINDlarge, MINDsmall and MINDdemo")
