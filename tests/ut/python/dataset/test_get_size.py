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
from mindspore import log as logger

FILES = ["../data/dataset/testTFTestAllTypes/test.data"]
DATASET_ROOT = "../data/dataset/testTFTestAllTypes/"
SCHEMA_FILE = "../data/dataset/testTFTestAllTypes/datasetSchema.json"
GENERATE_GOLDEN = False


def test_case1():
    data = ds.TFRecordDataset(FILES, SCHEMA_FILE)
    assert data.get_dataset_size() == 12
    assert data.get_batch_size() == 1
    assert data.get_repeat_count() == 1
    data = data.shuffle(100)
    assert data.get_dataset_size() == 12
    assert data.get_batch_size() == 1
    assert data.get_repeat_count() == 1
    data = data.batch(2)
    assert data.get_dataset_size() == 6
    assert data.get_batch_size() == 2
    assert data.get_repeat_count() == 1
    data = data.rename("col_sint64", "new_column")
    assert data.get_dataset_size() == 6
    assert data.get_batch_size() == 2
    assert data.get_repeat_count() == 1
    data = data.repeat(10)
    assert data.get_dataset_size() == 6
    assert data.get_batch_size() == 2
    assert data.get_repeat_count() == 10
    data = data.project(["new_column"])
    assert data.get_dataset_size() == 6
    assert data.get_batch_size() == 2
    assert data.get_repeat_count() == 10

    data2 = ds.TFRecordDataset(FILES, SCHEMA_FILE).batch(2).repeat(10)

    data1 = data.zip(data2)
    assert data1.get_dataset_size() == 6


def test_case2():
    data = ds.TFRecordDataset(FILES, num_samples=6)
    assert data.get_dataset_size() == 6
    data = data.shuffle(100)
    assert data.get_dataset_size() == 6
    data = data.batch(2)
    assert data.get_dataset_size() == 3
    data = data.rename("col_sint64", "new_column")
    assert data.get_dataset_size() == 3
    data = data.repeat(10)
    assert data.get_dataset_size() == 3
    data = data.project(["new_column"])
    assert data.get_dataset_size() == 3

    data2 = ds.TFRecordDataset(FILES, num_samples=6).batch(2).repeat(10)

    data1 = data.zip(data2)
    assert data1.get_dataset_size() == 3


def test_case3():
    data1 = ds.TFRecordDataset(FILES, SCHEMA_FILE).batch(2).repeat(10)
    data2 = ds.TFRecordDataset(FILES, SCHEMA_FILE).batch(2).repeat(5)
    data3 = ds.TFRecordDataset(FILES, SCHEMA_FILE).batch(2)

    data4 = ds.zip((data1, data2, data3))

    assert data4.get_dataset_size() == 6


def test_case4():
    data1 = ds.TFRecordDataset(FILES, SCHEMA_FILE).batch(2).repeat(10)
    data2 = ds.TFRecordDataset(FILES)
    assert data2.get_dataset_size() == 12
    data2 = data2.batch(2)
    assert data2.get_dataset_size() == 6
    data2 = data2.shuffle(100)
    assert data2.get_dataset_size() == 6
    data2 = data2.repeat(3)
    assert data2.get_dataset_size() == 6

    data3 = ds.zip((data1, data2))

    assert data3.get_dataset_size() == 6


def test_case5():
    data = ds.TFRecordDataset(FILES, num_samples=10).batch(3, drop_remainder=True)
    assert data.get_dataset_size() == 3
    data = ds.TFRecordDataset(FILES, num_samples=10).batch(3, drop_remainder=False)
    assert data.get_dataset_size() == 4


def test_cifar():
    data = ds.Cifar10Dataset("../data/dataset/testCifar10Data")
    assert data.get_dataset_size() == 10000

    data = ds.Cifar10Dataset("../data/dataset/testCifar10Data", num_samples=10)
    assert data.get_dataset_size() == 10

    data = ds.Cifar10Dataset("../data/dataset/testCifar10Data", num_samples=90000)
    assert data.get_dataset_size() == 10000

    data = ds.Cifar100Dataset("../data/dataset/testCifar100Data")
    assert data.get_dataset_size() == 10000

    data = ds.Cifar100Dataset("../data/dataset/testCifar100Data", num_samples=10)
    assert data.get_dataset_size() == 10

    data = ds.Cifar100Dataset("../data/dataset/testCifar100Data", num_samples=20000)
    assert data.get_dataset_size() == 10000


def test_mnist():
    data = ds.MnistDataset("../data/dataset/testMnistData")
    logger.info("dataset.size: {}".format(data.get_dataset_size()))
    assert data.get_dataset_size() == 10000

    data = ds.MnistDataset("../data/dataset/testMnistData", num_samples=10)
    assert data.get_dataset_size() == 10

    data = ds.MnistDataset("../data/dataset/testMnistData", num_samples=90000)
    assert data.get_dataset_size() == 10000


def test_manifest():
    data = ds.ManifestDataset("../data/dataset/testManifestData/test.manifest")
    assert data.get_dataset_size() == 4
    assert data.num_classes() == 3

    data = data.shuffle(100)
    assert data.num_classes() == 3


def test_imagefolder():
    data = ds.ImageFolderDatasetV2("../data/dataset/testPK/data/")
    assert data.get_dataset_size() == 44
    assert data.num_classes() == 4
    data = data.shuffle(100)
    assert data.num_classes() == 4

    data = ds.ImageFolderDatasetV2("../data/dataset/testPK/data/", num_samples=10)
    assert data.get_dataset_size() == 10
    assert data.num_classes() == 4


def test_generator():
    def generator():
        for i in range(64):
            yield (np.array([i]),)

    data1 = ds.GeneratorDataset(generator, ["data"])
    data1.set_dataset_size(10)
    assert data1.get_dataset_size() == 10
    data1.output_shapes()
    assert data1.get_dataset_size() == 10


if __name__ == '__main__':
    # test_compare_v1_and_2()
    # test_imagefolder()
    # test_manifest()
    test_case1()
    # test_case2()
    # test_case3()
    # test_case4()
    # test_case5()
