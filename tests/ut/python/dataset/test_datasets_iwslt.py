# Copyright 2021-2022 Huawei Technologies Co., Ltd
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

DATA_IWSLT2016_DIR = '../data/dataset/testIWSLT/IWSLT2016'
DATA_IWSLT2017_DIR = '../data/dataset/testIWSLT/IWSLT2017'


def test_iwslt2016_dataset_basic():
    """
    Feature: Test IWSLT2016 Dataset.
    Description: Read data from a single file.
    Expectation: The data is processed successfully.
    """
    buffer = []
    data = ds.IWSLT2016Dataset(DATA_IWSLT2016_DIR, usage='train', language_pair=["de", "en"], shuffle=False)
    data = data.repeat(2)
    data = data.skip(2)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append(d)
    assert len(buffer) == 2


def test_iwslt2016_dataset_quoted():
    """
    Feature: Test get the IWSLT2016 Dataset.
    Description: Read IWSLT2016 data and get data.
    Expectation: The data is processed successfully.
    """
    data = ds.IWSLT2016Dataset(DATA_IWSLT2016_DIR, usage='train', language_pair=["de", "en"], shuffle=False)
    buffer = []
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.extend([d['text'],
                       d['translation']])
    assert buffer == ["Code schreiben macht Freude.",
                      "Writing code is a joy.",
                      "Ich hoffe in Zukunft weniger Überstunden machen zu können.",
                      "I hope to work less overtime in the future."]


def test_iwslt2016_dataset_usage_all():
    """
    Feature: Test IWSLT2016 Dataset (usage=all).
    Description: Read train data and test data.
    Expectation: The data is processed successfully.
    """
    buffer = []
    data = ds.IWSLT2016Dataset(DATA_IWSLT2016_DIR, usage='all', language_pair=["de", "en"], valid_set='tst2013',
                               test_set='tst2014', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append(d)
    assert len(buffer) == 6


def test_iwslt2016_dataset_get_datasetsize():
    """
    Feature: Test Getters.
    Description: Test get_dataset_size of IWSLT2016 dataset.
    Expectation: The data is processed successfully.
    """
    data = ds.IWSLT2016Dataset(DATA_IWSLT2016_DIR, usage='train', language_pair=["de", "en"], shuffle=False)
    size = data.get_dataset_size()
    assert size == 2


def test_iwslt2016_dataset_distribution():
    """
    Feature: Test IWSLT2016Dataset in distribution.
    Description: Test in a distributed state.
    Expectation: The data is processed successfully.
    """
    data = ds.IWSLT2016Dataset(DATA_IWSLT2016_DIR, usage='train', language_pair=["de", "en"], shuffle=False,
                               num_shards=2, shard_id=0)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 1


def test_iwslt2016_dataset_num_samples():
    """
    Feature: Test IWSLT2016 Dataset (num_samples=2).
    Description: Test get num_samples.
    Expectation: The data is processed successfully.
    """
    data = ds.IWSLT2016Dataset(DATA_IWSLT2016_DIR, usage='train', language_pair=["de", "en"], shuffle=False,
                               num_samples=2)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 2


def test_iwslt2016_dataset_exception():
    """
    Feature: Error Test.
    Description: Test the wrong input.
    Expectation: Unable to read in data.
    """
    def exception_func(item):
        raise Exception("Error occur!")

    try:
        data = ds.IWSLT2016Dataset(DATA_IWSLT2016_DIR, usage='train', language_pair=["de", "en"], shuffle=False)
        data = data.map(operations=exception_func, input_columns=["text"], num_parallel_workers=1)
        for _ in data.create_dict_iterator():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    try:
        data = ds.IWSLT2016Dataset(DATA_IWSLT2016_DIR, usage='train', language_pair=["de", "en"], shuffle=False)
        data = data.map(operations=exception_func, input_columns=["translation"], num_parallel_workers=1)
        for _ in data.create_dict_iterator():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)


def test_iwslt2017_dataset_basic():
    """
    Feature: Test IWSLT2017 Dataset.
    Description: Read data from a single file.
    Expectation: The data is processed successfully.
    """
    buffer = []
    data = ds.IWSLT2017Dataset(DATA_IWSLT2017_DIR, usage='train', language_pair=["de", "en"], shuffle=False)
    data = data.repeat(2)
    data = data.skip(2)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append(d)
    assert len(buffer) == 2


def test_iwslt2017_dataset_quoted():
    """
    Feature: Test get the IWSLT2017 Dataset.
    Description: Read IWSLT2017 data and get data.
    Expectation: The data is processed successfully.
    """
    data = ds.IWSLT2017Dataset(DATA_IWSLT2017_DIR, usage='train', language_pair=["de", "en"], shuffle=False)
    buffer = []
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.extend([d['text'],
                       d['translation']])
    assert buffer == ["Schönes Wetter heute.",
                      "The weather is nice today.",
                      "Ich bin heute gut gelaunt.",
                      "I am in a good mood today."]


def test_iwslt2017_dataset_usage_all():
    """
    Feature: Test IWSLT2017 Dataset(usage=all).
    Description: Read train data and test data.
    Expectation: The data is processed successfully.
    """
    buffer = []
    data = ds.IWSLT2017Dataset(DATA_IWSLT2017_DIR, usage='all', language_pair=["de", "en"], shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append(d)
    assert len(buffer) == 6


def test_iwslt2017_dataset_get_datasetsize():
    """
    Feature: Test Getters.
    Description: Test get_dataset_size of IWSLT2017 dataset.
    Expectation: The data is processed successfully.
    """
    data = ds.IWSLT2017Dataset(DATA_IWSLT2017_DIR, usage='train', language_pair=["de", "en"], shuffle=False)
    size = data.get_dataset_size()
    assert size == 2


def test_iwslt2017_dataset_distribution():
    """
    Feature: Test IWSLT2017Dataset in distribution.
    Description: Test in a distributed state.
    Expectation: The data is processed successfully.
    """
    data = ds.IWSLT2017Dataset(DATA_IWSLT2017_DIR, usage='train', language_pair=["de", "en"], shuffle=False,
                               num_shards=2, shard_id=0)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 1


def test_iwslt2017_dataset_num_samples():
    """
    Feature: Test IWSLT2017 Dataset (num_samples=2).
    Description: Test get num_samples.
    Expectation: The data is processed successfully.
    """
    data = ds.IWSLT2017Dataset(DATA_IWSLT2017_DIR, usage='train', language_pair=["de", "en"], shuffle=False,
                               num_samples=2)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 2


def test_iwslt2017_dataset_exception():
    """
    Feature: Error Test.
    Description: Test the wrong input.
    Expectation: Unable to read in data.
    """
    def exception_func(item):
        raise Exception("Error occur!")

    try:
        data = ds.IWSLT2017Dataset(DATA_IWSLT2017_DIR, usage='train', language_pair=["de", "en"], shuffle=False)
        data = data.map(operations=exception_func, input_columns=["text"], num_parallel_workers=1)
        for _ in data.create_dict_iterator():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    try:
        data = ds.IWSLT2017Dataset(DATA_IWSLT2017_DIR, usage='train', language_pair=["de", "en"], shuffle=False)
        data = data.map(operations=exception_func, input_columns=["translation"], num_parallel_workers=1)
        for _ in data.create_dict_iterator():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)


if __name__ == "__main__":
    test_iwslt2016_dataset_basic()
    test_iwslt2016_dataset_quoted()
    test_iwslt2016_dataset_usage_all()
    test_iwslt2016_dataset_get_datasetsize()
    test_iwslt2016_dataset_distribution()
    test_iwslt2016_dataset_num_samples()
    test_iwslt2016_dataset_exception()

    test_iwslt2017_dataset_basic()
    test_iwslt2017_dataset_quoted()
    test_iwslt2017_dataset_usage_all()
    test_iwslt2017_dataset_get_datasetsize()
    test_iwslt2017_dataset_distribution()
    test_iwslt2017_dataset_num_samples()
    test_iwslt2017_dataset_exception()
