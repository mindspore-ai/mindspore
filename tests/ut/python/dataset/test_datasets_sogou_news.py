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

DATA_SOGOU_NEWS_DIR = '../data/dataset/testSogouNews/'


def test_sogou_news_dataset_basic():
    """
    Feature: Test SogouNews Dataset.
    Description: Read data from a test.csv file.
    Expectation: The data is processed successfully.
    """
    buffer = []
    data = ds.SogouNewsDataset(DATA_SOGOU_NEWS_DIR, usage='test', shuffle=False)
    data = data.repeat(2)
    data = data.skip(2)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append(d)
    assert len(buffer) == 4


def test_sogou_news_dataset_all():
    """
    Feature: Test SogouNews Dataset.
    Description: Read data from a test.csv and train.csv file.
    Expectation: The data is processed successfully.
    """
    data = ds.SogouNewsDataset(DATA_SOGOU_NEWS_DIR, usage='all', shuffle=False)
    buffer = []
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.extend([d['index'],
                       d['title'],
                       d['content']])
    assert buffer == ["1", "Jefferson commented on thick eyebrow: he has the top five talents in the league, but he "
                      "is not the top five", "They say he has the talent of the top five in the league. The talent "
                      "of the top five in the league is one of the most disrespectful statements. I say he has the "
                      "talent of the top five in the league, but he is not the top five players because the top five "
                      "players play every night.",
                      "1", "Make history", "Su Bingtian's 100m breakthrough\\n 9.83",
                      "3", "Group pictures: Liu Shishi's temperament in early autumn released a large piece of micro "
                      "curly long hair, elegant, lazy, gentle and capable", "Liu Shishi's latest group of cover "
                      "magazine blockbusters are released. In the photos, Liu Shishi's long hair is slightly curly, "
                      "or camel colored belted woolen coat, or plaid suit, which is gentle and elegant and beautiful "
                      "to a new height.",
                      "4", "Tesla price", "Tesla reduced its price by 70000 yuan",
                      "3", "Ni Ni deduces elegant retro style in different styles", "Ni Ni's latest group of magazine "
                      "cover blockbusters released that wearing gift hats is cool, retro, unique and full of fashion "
                      "expression.",
                      "1", "Opening ceremony of the 14th National Games", "On the evening of September 15, Beijing "
                      "time, the 14th games of the people's Republic of China opened in Xi'an Olympic Sports Center "
                      "Stadium, Shaanxi Province. Yang Qian, the first gold medalist of the Chinese delegation in "
                      "the Tokyo Olympic Games and a Post-00 shooter, lit the main torch platform. From then on, "
                      "to September 27, the 14th National Games flame will burn here for 12 days."]


def test_sogou_news_dataset_quoted():
    """
    Feature: Test get the SogouNews Dataset.
    Description: Read SogouNewsDataset data and get data.
    Expectation: The data is processed successfully.
    """
    data = ds.SogouNewsDataset(DATA_SOGOU_NEWS_DIR, usage='test', shuffle=False)
    buffer = []
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.extend([d['index'],
                       d['title'],
                       d['content']])
    assert buffer == ["1", "Make history", "Su Bingtian's 100m breakthrough\\n 9.83",
                      "4", "Tesla price", "Tesla reduced its price by 70000 yuan",
                      "1", "Opening ceremony of the 14th National Games", "On the evening of September 15, Beijing time"
                      ", the 14th games of the people's Republic of China opened in Xi'an Olympic Sports Center "
                      "Stadium, Shaanxi Province. Yang Qian, the first gold medalist of the Chinese delegation in the"
                      " Tokyo Olympic Games and a Post-00 shooter, lit the main torch platform. From then on, to "
                      "September 27, the 14th National Games flame will burn here for 12 days."]


def test_sogou_news_dataset_usage_all():
    """
    Feature: Test SogouNews Dataset(usage=all).
    Description: Read train data and test data.
    Expectation: The data is processed successfully.
    """
    buffer = []
    data = ds.SogouNewsDataset(DATA_SOGOU_NEWS_DIR, usage='all', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append(d)
    assert len(buffer) == 6


def test_sogou_news_dataset_get_datasetsize():
    """
    Feature: Test Getters.
    Description: Test get_dataset_size of SogouNews dataset.
    Expectation: The data is processed successfully.
    """
    data = ds.SogouNewsDataset(DATA_SOGOU_NEWS_DIR, usage='test', shuffle=False)
    size = data.get_dataset_size()
    assert size == 3


def test_sogou_news_dataset_distribution():
    """
    Feature: Test SogouNewsDataset in distribution.
    Description: Test in a distributed state.
    Expectation: The data is processed successfully.
    """
    data = ds.SogouNewsDataset(DATA_SOGOU_NEWS_DIR, usage='test', shuffle=False, num_shards=2, shard_id=0)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 2


def test_sogou_news_dataset_num_samples():
    """
    Feature: Test SogouNews Dataset(num_samples = 2).
    Description: Test get num_samples.
    Expectation: The data is processed successfully.
    """
    data = ds.SogouNewsDataset(DATA_SOGOU_NEWS_DIR, usage='test', shuffle=False, num_samples=2)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 2


def test_sogou_news_dataset_exception():
    """
    Feature: Error Test.
    Description: Test the wrong input.
    Expectation: Unable to read in data.
    """
    def exception_func(item):
        raise Exception("Error occur!")

    try:
        data = ds.SogouNewsDataset(DATA_SOGOU_NEWS_DIR, usage='test', shuffle=False)
        data = data.map(operations=exception_func, input_columns=["index"], num_parallel_workers=1)
        for _ in data.create_dict_iterator():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    try:
        data = ds.SogouNewsDataset(DATA_SOGOU_NEWS_DIR, usage='test', shuffle=False)
        data = data.map(operations=exception_func, input_columns=["title"], num_parallel_workers=1)
        for _ in data.create_dict_iterator():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    try:
        data = ds.SogouNewsDataset(DATA_SOGOU_NEWS_DIR, usage='test', shuffle=False)
        data = data.map(operations=exception_func, input_columns=["content"], num_parallel_workers=1)
        for _ in data.create_dict_iterator():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)


if __name__ == "__main__":
    test_sogou_news_dataset_basic()
    test_sogou_news_dataset_all()
    test_sogou_news_dataset_quoted()
    test_sogou_news_dataset_usage_all()
    test_sogou_news_dataset_get_datasetsize()
    test_sogou_news_dataset_distribution()
    test_sogou_news_dataset_num_samples()
    test_sogou_news_dataset_exception()
