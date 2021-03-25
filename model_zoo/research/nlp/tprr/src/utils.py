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
Retriever Utils.

"""

import json
import unicodedata
import pickle as pkl


def normalize(text):
    """normalize text"""
    text = unicodedata.normalize('NFD', text)
    return text[0].capitalize() + text[1:]


def read_query(config, device_id):
    """get query data"""
    with open(config.q_path + str(device_id), 'rb') as f:
        queries = pkl.load(f, encoding='gbk')
    return queries


def split_queries(config, queries):
    batch_size = config.batch_size
    batch_queries = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
    return batch_queries


def save_json(obj, path, name):
    with open(path + name, "w") as f:
        return json.dump(obj, f)

def get_new_title(title):
    """get new title"""
    if title[-2:] == "_0":
        return normalize(title[:-2]) + "_0"
    return normalize(title) + "_0"


def get_raw_title(title):
    """get raw title"""
    if title[-2:] == "_0":
        return title[:-2]
    return title
