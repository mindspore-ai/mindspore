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
# ==============================================================================
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as data_trans
from mindspore import log as logger

DATA_FILE = "../data/dataset/testManifestData/test.manifest"


def test_manifest_dataset_train():
    data = ds.ManifestDataset(DATA_FILE, decode=True)
    count = 0
    cat_count = 0
    dog_count = 0
    for item in data.create_dict_iterator():
        logger.info("item[image] is {}".format(item["image"]))
        count = count + 1
        if item["label"].size == 1 and item["label"] == 0:
            cat_count = cat_count + 1
        elif item["label"].size == 1 and item["label"] == 1:
            dog_count = dog_count + 1
    assert cat_count == 2
    assert dog_count == 1
    assert count == 4


def test_manifest_dataset_eval():
    data = ds.ManifestDataset(DATA_FILE, "eval", decode=True)
    count = 0
    for item in data.create_dict_iterator():
        logger.info("item[image] is {}".format(item["image"]))
        count = count + 1
        if item["label"] != 0 and item["label"] != 1:
            assert 0
    assert count == 2


def test_manifest_dataset_class_index():
    class_indexing = {"dog": 11}
    data = ds.ManifestDataset(DATA_FILE, decode=True, class_indexing=class_indexing)
    out_class_indexing = data.get_class_indexing()
    assert out_class_indexing == {"dog": 11}
    count = 0
    for item in data.create_dict_iterator():
        logger.info("item[image] is {}".format(item["image"]))
        count = count + 1
        if item["label"] != 11:
            assert 0
    assert count == 1


def test_manifest_dataset_get_class_index():
    data = ds.ManifestDataset(DATA_FILE, decode=True)
    class_indexing = data.get_class_indexing()
    assert class_indexing == {'cat': 0, 'dog': 1, 'flower': 2}
    data = data.shuffle(4)
    class_indexing = data.get_class_indexing()
    assert class_indexing == {'cat': 0, 'dog': 1, 'flower': 2}
    count = 0
    for item in data.create_dict_iterator():
        logger.info("item[image] is {}".format(item["image"]))
        count = count + 1
    assert count == 4


def test_manifest_dataset_multi_label():
    data = ds.ManifestDataset(DATA_FILE, decode=True, shuffle=False)
    count = 0
    expect_label = [1, 0, 0, [0, 2]]
    for item in data.create_dict_iterator():
        assert item["label"].tolist() == expect_label[count]
        logger.info("item[image] is {}".format(item["image"]))
        count = count + 1
    assert count == 4


def multi_label_hot(x):
    result = np.zeros(x.size // x.ndim, dtype=int)
    if x.ndim > 1:
        for i in range(x.ndim):
            result = np.add(result, x[i])
    else:
        result = np.add(result, x)

    return result


def test_manifest_dataset_multi_label_onehot():
    data = ds.ManifestDataset(DATA_FILE, decode=True, shuffle=False)
    expect_label = [[[0, 1, 0], [1, 0, 0]], [[1, 0, 0], [1, 0, 1]]]
    one_hot_encode = data_trans.OneHot(3)
    data = data.map(input_columns=["label"], operations=one_hot_encode)
    data = data.map(input_columns=["label"], operations=multi_label_hot)
    data = data.batch(2)
    count = 0
    for item in data.create_dict_iterator():
        assert item["label"].tolist() == expect_label[count]
        logger.info("item[image] is {}".format(item["image"]))
        count = count + 1


if __name__ == '__main__':
    test_manifest_dataset_train()
    test_manifest_dataset_eval()
    test_manifest_dataset_class_index()
    test_manifest_dataset_get_class_index()
    test_manifest_dataset_multi_label()
    test_manifest_dataset_multi_label_onehot()
