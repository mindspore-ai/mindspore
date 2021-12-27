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
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.text.transforms as a_c_trans

POLARITY_DIR = '../data/dataset/testAmazonReview/polarity'
FULL_DIR = '../data/dataset/testAmazonReview/full'


def count_unequal_element(data_expected, data_me):
    assert data_expected.shape == data_me.shape
    assert data_expected == data_me


def test_amazon_review_polarity_dataset_basic():
    """
    Feature: Test AmazonReviewPolarity Dataset.
    Description: read data from a single file.
    Expectation: the data is processed successfully.
    """
    buffer = []
    data = ds.AmazonReviewDataset(POLARITY_DIR, usage='test', shuffle=False)
    data = data.repeat(2)
    data = data.skip(2)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append(d)
    assert len(buffer) == 2


def test_amazon_review_full_dataset_basic():
    """
    Feature: Test AmazonReviewFull Dataset.
    Description: read data from a single file.
    Expectation: the data is processed successfully.
    """
    buffer = []
    data = ds.AmazonReviewDataset(FULL_DIR, usage='test', shuffle=False)
    data = data.repeat(2)
    data = data.skip(2)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append(d)
    assert len(buffer) == 4


def test_amazon_review_dataset_quoted():
    """
    Feature: Test get the AmazonReview Dataset.
    Description: read AmazonReviewPolarityDataset data and get data.
    Expectation: the data is processed successfully.
    """
    data = ds.AmazonReviewDataset(FULL_DIR, usage='test', shuffle=False)
    buffer = []
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.extend([d['label'].item().decode("utf8"),
                       d['title'].item().decode("utf8"),
                       d['content'].item().decode("utf8")])
    assert buffer == ["1", "amazing", "unlimited buyback!",
                      "4", "delightful", "a funny book!",
                      "3", "Small", "It is a small ball!"]


def test_amazon_review_full_dataset_usage_all():
    """
    Feature: Test AmazonReviewPolarity Dataset(usage=all).
    Description: read train data and test data.
    Expectation: the data is processed successfully.
    """
    buffer = []
    data = ds.AmazonReviewDataset(FULL_DIR, usage='all', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.extend([d['label'].item().decode("utf8"),
                       d['title'].item().decode("utf8"),
                       d['content'].item().decode("utf8")])
    assert buffer == ["1", "amazing", "unlimited buyback!",
                      "3", "Satisfied", "good quality.",
                      "4", "delightful", "a funny book!",
                      "5", "good", "This is an very good product.",
                      "3", "Small", "It is a small ball!",
                      "1", "bad", "work badly."]


def test_amazon_review_polarity_dataset_usage_all():
    """
    Feature: Test AmazonReviewPolarityPolarity Dataset(usage=all).
    Description: read train data and test data.
    Expectation: the data is processed successfully.
    """
    buffer = []
    data = ds.AmazonReviewDataset(POLARITY_DIR, usage='all', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.extend([d['label'].item().decode("utf8"),
                       d['title'].item().decode("utf8"),
                       d['content'].item().decode("utf8")])
    assert buffer == ["1", "DVD", "It is very good!",
                      "2", "Great Read", "I thought this book was excellent!",
                      "2", "Book", "I would read it again lol.",
                      "1", "Oh dear", "It is so bad!",
                      "2", "Delicious", "A funny product."]


def test_amazon_review_dataset_get_datasetsize():
    """
    Feature: Test Getters.
    Description: test get_dataset_size of AmazonReview dataset.
    Expectation: the data is processed successfully.
    """
    data = ds.AmazonReviewDataset(FULL_DIR, usage='test', shuffle=False)
    size = data.get_dataset_size()
    assert size == 3


def test_amazon_review_dataset_distribution():
    """
    Feature: Test AmazonReviewDataset in distribution.
    Description: test in a distributed state.
    Expectation: the data is processed successfully.
    """
    data = ds.AmazonReviewDataset(FULL_DIR, usage='test', shuffle=False, num_shards=2, shard_id=0)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 2


def test_amazon_review_dataset_num_samples():
    """
    Feature: Test AmazonReview Dataset(num_samples = 2).
    Description: test get num_samples.
    Expectation: the data is processed successfully.
    """
    data = ds.AmazonReviewDataset(FULL_DIR, usage='test', shuffle=False, num_samples=2)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 2


def test_amazon_review_dataset_exception():
    """
    Feature: Error Test.
    Description: test the wrong input.
    Expectation: unable to read in data.
    """
    def exception_func(item):
        raise Exception("Error occur!")

    try:
        data = ds.AmazonReviewDataset(FULL_DIR, usage='test', shuffle=False)
        data = data.map(operations=exception_func, input_columns=["label"], num_parallel_workers=1)
        for _ in data.create_dict_iterator():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data = ds.AmazonReviewDataset(FULL_DIR, usage='test', shuffle=False)
        data = data.map(operations=exception_func, input_columns=["title"], num_parallel_workers=1)
        for _ in data.create_dict_iterator():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data = ds.AmazonReviewDataset(FULL_DIR, usage='test', shuffle=False)
        data = data.map(operations=exception_func, input_columns=["content"], num_parallel_workers=1)
        for _ in data.create_dict_iterator():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)


def test_amazon_review_dataset_pipeline():
    """
    Feature: AmazonReviewDataset
    Description: test AmazonReviewDataset in pipeline mode
    Expectation: the data is processed successfully
    """
    expected_columns1 = np.array(["3", "5", "1"], dtype=np.string_)
    dataset = ds.AmazonReviewDataset(FULL_DIR, 'train', shuffle=False)
    filter_wikipedia_xml_op = a_c_trans.CaseFold()
    dataset = dataset.map(input_columns=["label"], operations=filter_wikipedia_xml_op, num_parallel_workers=1)
    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        count_unequal_element(np.array(expected_columns1[i]), data['label'])
        i += 1
    assert i == 3

    expected_columns2 = np.array(["satisfied", "good", "bad"], dtype=np.string_)
    dataset = ds.AmazonReviewDataset(FULL_DIR, 'train', shuffle=False)
    filter_wikipedia_xml_op = a_c_trans.CaseFold()
    dataset = dataset.map(input_columns=["title"], operations=filter_wikipedia_xml_op, num_parallel_workers=1)
    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        count_unequal_element(np.array(expected_columns2[i]), data['title'])
        i += 1
    assert i == 3

    expected_columns3 = np.array(["good quality.",
                                  "this is an very good product.",
                                  "work badly."], dtype=np.string_)
    dataset = ds.AmazonReviewDataset(FULL_DIR, 'train', shuffle=False)
    filter_wikipedia_xml_op = a_c_trans.CaseFold()
    dataset = dataset.map(input_columns=["content"], operations=filter_wikipedia_xml_op, num_parallel_workers=1)
    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        count_unequal_element(np.array(expected_columns3[i]), data['content'])
        i += 1
    assert i == 3


if __name__ == "__main__":
    test_amazon_review_polarity_dataset_basic()
    test_amazon_review_full_dataset_basic()
    test_amazon_review_dataset_quoted()
    test_amazon_review_full_dataset_usage_all()
    test_amazon_review_polarity_dataset_usage_all()
    test_amazon_review_dataset_get_datasetsize()
    test_amazon_review_dataset_distribution()
    test_amazon_review_dataset_num_samples()
    test_amazon_review_dataset_exception()
    test_amazon_review_dataset_pipeline()
