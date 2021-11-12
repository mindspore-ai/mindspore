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
# ==============================================================================
import mindspore.dataset as ds

DATA_DIR = '../data/dataset/testDBpedia/'


def test_dbpedia_dataset_basic():
    """
    Feature: DBpediaDataset.
    Description: read data from train file.
    Expectation: the data is processed successfully.
    """
    buffer = []
    data = ds.DBpediaDataset(DATA_DIR, usage="train", shuffle=False)
    data = data.repeat(2)
    data = data.skip(3)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append(d)
    assert len(buffer) == 3


def test_dbpedia_dataset_quoted():
    """
    Feature: DBpediaDataset.
    Description: read the data and compare it to expectations.
    Expectation: the data is processed successfully.
    """
    data = ds.DBpediaDataset(DATA_DIR, usage="test", shuffle=False)
    buffer = []
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.extend([d['class'].item().decode("utf8"),
                       d['title'].item().decode("utf8"),
                       d['content'].item().decode("utf8")])
    assert buffer == ["5", "My Bedroom", "Look at this room. It's my bedroom.",
                      "8", "My English teacher", "She has two big eyes and a small mouth.",
                      "6", "My Holiday", "I have a lot of fun every day."]


def test_dbpedia_dataset_usage():
    """
    Feature: DBpediaDataset.
    Description: read all files with usage all.
    Expectation: the data is processed successfully.
    """
    buffer = []
    data = ds.DBpediaDataset(DATA_DIR, usage="all", shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append(d)
    assert len(buffer) == 6


def test_dbpedia_dataset_get_datasetsize():
    """
    Feature: DBpediaDataset.
    Description: test get_dataset_size function.
    Expectation: the data is processed successfully.
    """
    data = ds.DBpediaDataset(DATA_DIR, usage="test", shuffle=False)
    size = data.get_dataset_size()
    assert size == 3


def test_dbpedia_dataset_distribution():
    """
    Feature: DBpediaDataset.
    Description: test in a distributed state.
    Expectation: the data is processed successfully.
    """
    data = ds.DBpediaDataset(DATA_DIR, usage="test", shuffle=False, num_shards=2, shard_id=0)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 2


def test_dbpedia_dataset_num_samples():
    """
    Feature: DBpediaDataset.
    Description: test num_samples parameter.
    Expectation: the data is processed successfully.
    """
    data = ds.DBpediaDataset(DATA_DIR, usage="test", shuffle=False, num_samples=2)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 2


def test_dbpedia_dataset_exception():
    """
    Feature: DBpediaDataset.
    Description: test the wrong input.
    Expectation: Unable to read data properly.
    """
    def exception_func(item):
        raise Exception("Error occur!")
    try:
        data = ds.DBpediaDataset(DATA_DIR, usage="test", shuffle=False)
        data = data.map(operations=exception_func, input_columns=["class"], num_parallel_workers=1)
        for _ in data.create_dict_iterator():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)
    try:
        data = ds.DBpediaDataset(DATA_DIR, usage="test", shuffle=False)
        data = data.map(operations=exception_func, input_columns=["content"], num_parallel_workers=1)
        for _ in data.create_dict_iterator():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)


if __name__ == "__main__":
    test_dbpedia_dataset_basic()
    test_dbpedia_dataset_quoted()
    test_dbpedia_dataset_usage()
    test_dbpedia_dataset_get_datasetsize()
    test_dbpedia_dataset_distribution()
    test_dbpedia_dataset_num_samples()
    test_dbpedia_dataset_exception()
