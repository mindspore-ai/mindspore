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

DATA_POLARITY_DIR = '../data/dataset/testYelpReview/polarity'
DATA_FULL_DIR = '../data/dataset/testYelpReview/full'


def test_yelp_review_polarity_dataset_basic():
    """
    Feature: Test YelpReviewPolarity Dataset.
    Description: Read data from a single file.
    Expectation: The data is processed successfully.
    """
    buffer = []
    data = ds.YelpReviewDataset(DATA_POLARITY_DIR, usage='test', shuffle=False)
    data = data.repeat(2)
    data = data.skip(2)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append(d)
    assert len(buffer) == 2


def test_yelp_review_full_dataset_basic():
    """
    Feature: Test YelpReviewFull Dataset.
    Description: Read data from a single file.
    Expectation: The data is processed successfully.
    """
    buffer = []
    data = ds.YelpReviewDataset(DATA_FULL_DIR, usage='test', shuffle=False)
    data = data.repeat(2)
    data = data.skip(2)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append(d)
    assert len(buffer) == 2


def test_yelp_review_dataset_quoted():
    """
    Feature: Test get the YelpReview Dataset.
    Description: Read YelpReviewPolarityDataset data and get data.
    Expectation: The data is processed successfully.
    """
    data = ds.YelpReviewDataset(DATA_POLARITY_DIR, usage='test', shuffle=False)
    buffer = []
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.extend([d['label'],
                       d['text']])
    assert buffer == ["2", "\\\"Yelp\\\" service was very good.\\n",
                      "1", "\\\"Yelp\\\" service was very bad.\\n"]


def test_yelp_review_dataset_usage_all():
    """
    Feature: Test YelpReviewPolarity Dataset(usage=all).
    Description: Read train data and test data.
    Expectation: The data is processed successfully.
    """
    buffer = []
    data = ds.YelpReviewDataset(DATA_POLARITY_DIR, usage='all', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append(d)
    assert len(buffer) == 5


def test_yelp_review_dataset_get_datasetsize():
    """
    Feature: Test Getters.
    Description: Test get_dataset_size of YelpReview dataset.
    Expectation: The data is processed successfully.
    """
    data = ds.YelpReviewDataset(DATA_POLARITY_DIR, usage='test', shuffle=False)
    size = data.get_dataset_size()
    assert size == 2


def test_yelp_review_dataset_distribution():
    """
    Feature: Test YelpReviewDataset in distribution.
    Description: Test in a distributed state.
    Expectation: The data is processed successfully.
    """
    data = ds.YelpReviewDataset(DATA_POLARITY_DIR, usage='test', shuffle=False, num_shards=2, shard_id=0)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 1


def test_yelp_review_dataset_num_samples():
    """
    Feature: Test YelpReview Dataset(num_samples = 2).
    Description: Test get num_samples.
    Expectation: The data is processed successfully.
    """
    data = ds.YelpReviewDataset(DATA_POLARITY_DIR, usage='test', shuffle=False, num_samples=2)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 2


def test_yelp_review_dataset_exception():
    """
    Feature: Error Test.
    Description: Test the wrong input.
    Expectation: Unable to read in data.
    """
    def exception_func(item):
        raise Exception("Error occur!")

    try:
        data = ds.YelpReviewDataset(DATA_POLARITY_DIR, usage='test', shuffle=False)
        data = data.map(operations=exception_func, input_columns=["label"], num_parallel_workers=1)
        for _ in data.create_dict_iterator():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    try:
        data = ds.YelpReviewDataset(DATA_POLARITY_DIR, usage='test', shuffle=False)
        data = data.map(operations=exception_func, input_columns=["text"], num_parallel_workers=1)
        for _ in data.create_dict_iterator():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)


if __name__ == "__main__":
    test_yelp_review_polarity_dataset_basic()
    test_yelp_review_full_dataset_basic()
    test_yelp_review_dataset_quoted()
    test_yelp_review_dataset_usage_all()
    test_yelp_review_dataset_get_datasetsize()
    test_yelp_review_dataset_distribution()
    test_yelp_review_dataset_num_samples()
    test_yelp_review_dataset_exception()
