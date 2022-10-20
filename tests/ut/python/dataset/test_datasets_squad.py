# Copyright 2022 Huawei Technologies Co., Ltd
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
import pytest

import mindspore.dataset as ds

DATASET_DIR_V1 = '../data/dataset/testSQuAD/SQuAD1'
DATASET_DIR_V2 = '../data/dataset/testSQuAD/SQuAD2'


def test_squad_basic():
    """
    Feature: SQuADDataset.
    Description: Test SQuADDataset with repeat, skip and so on.
    Expectation: The data is processed successfully.
    """
    data = ds.SQuADDataset(DATASET_DIR_V1, usage='train', shuffle=False)

    data = data.repeat(2)
    data = data.skip(3)
    expected_result = ["Who is \"The Father of Modern Computers\"?",
                       "When was John von Neumann's birth date?",
                       "Where is John von Neumann's birthplace?"]
    count = 0
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert d['question'] == expected_result[count]
        count += 1
    assert count == 3


def test_squad_num_shards():
    """
    Feature: SQuADDataset.
    Description: Test num_shards param of SQuAD dataset.
    Expectation: The data is processed successfully.
    """
    data = ds.SQuADDataset(DATASET_DIR_V1, usage='train',
                           num_shards=3, shard_id=2)
    expected_result = ["Where is John von Neumann's birthplace?"]
    count = 0
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert d['question'] == expected_result[count]
        count += 1
    assert count == 1


def test_squad_num_samples():
    """
    Feature: SQuADDataset.
    Description: Test num_samples param of SQuAD dataset.
    Expectation: The data is processed successfully.
    """
    data = ds.SQuADDataset(DATASET_DIR_V1, usage='train', num_samples=2)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 2


def test_squad_dataset_get_datasetsize():
    """
    Feature: SQuADDataset.
    Description: Test get_dataset_size of SQuAD dataset.
    Expectation: The data is processed successfully.
    """
    data = ds.SQuADDataset(DATASET_DIR_V1, usage='train')
    size = data.get_dataset_size()
    assert size == 3


def test_squad_version1():
    """
    Feature: SQuADDataset.
    Description: Test SQuAD 1.1 for train, dev and all.
    Expectation: The data is processed successfully.
    """
    # train
    data = ds.SQuADDataset(DATASET_DIR_V1, usage='train', shuffle=False)
    expected_result = ["Who is \"The Father of Modern Computers\"?",
                       "When was John von Neumann's birth date?",
                       "Where is John von Neumann's birthplace?"]
    count = 0
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert d['question'] == expected_result[count]
        count += 1
    assert count == 3

    # dev
    data = ds.SQuADDataset(DATASET_DIR_V1, usage='dev', shuffle=False)
    expected_result = ["\"The Mathematical Principles of Natural Philosophy\" is a philosophical philosophy " +
                       "of physics created by British Cognitive Isaac Newton. It was first published in 1687.",
                       "\"The Mathematical Principles of Natural Philosophy\" is a philosophical philosophy " +
                       "of physics created by British Cognitive Isaac Newton. It was first published in 1687."]
    count = 0
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert d['context'] == expected_result[count]
        count += 1
    assert count == 2

    # all
    data = ds.SQuADDataset(DATASET_DIR_V1, usage='all', shuffle=False)
    expected_result = [[0], [122, 122, 122], [18], [162, 162, 162], [55]]
    count = 0
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert [i.item() for i in d['answer_start']] == expected_result[count]
        count += 1
    assert count == 5


def test_squad_version2():
    """
    Feature: SQuADDataset.
    Description: Test SQuAD2.0 for train, dev and all.
    Expectation: The data is processed successfully.
    """

    # train
    data = ds.SQuADDataset(DATASET_DIR_V2, usage='train', shuffle=False)
    expected_result = ["Stephen William Hawking, born on January 8, 1942 in Oxford, England, " +
                       "is one of the greatest modern physicists.",
                       "Stephen William Hawking, born on January 8, 1942 in Oxford, England, " +
                       "is one of the greatest modern physicists."]
    count = 0
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert d['context'] == expected_result[count]
        count += 1
    assert count == 2

    # dev
    data = ds.SQuADDataset(DATASET_DIR_V2, usage='dev', shuffle=False)
    expected_result = ["What is the lifestyle of dolphins?",
                       "Who ate the squid?"]
    count = 0
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert d['question'] == expected_result[count]
        count += 1
    assert count == 2

    # all
    data = ds.SQuADDataset(DATASET_DIR_V2, usage='all', shuffle=False)
    expected_result = [["Oxford, England"],
                       ["live in groups", "live in groups",
                        "live in groups", "live in groups"],
                       ["January 8, 1942"], [""]]
    count = 0
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        result = [i for i in d['text']]
        assert result == expected_result[count]
        count += 1
    assert count == 4


def test_squad_device_que():
    """
    Feature: SQuADDataset.
    Description: Test SQuAD with device_que.
    Expectation: The data is processed successfully.
    """
    data = ds.SQuADDataset(DATASET_DIR_V1, usage='train', shuffle=False)
    data = data.device_que()
    data.send()


def test_squad_invalid_dir():
    """
    Feature: SQuADDataset.
    Description: Test SQuAD with invalid dir.
    Expectation: Throw correct error and message.
    """
    invalid_dataset_dir = '../data/dataset/invalid_dir'
    with pytest.raises(ValueError) as info:
        _ = ds.SQuADDataset(invalid_dataset_dir, usage='train', shuffle=False)
    assert "The folder " + invalid_dataset_dir + " does not exist or is not a directory or permission denied!" \
        in str(info.value)
    assert invalid_dataset_dir in str(info.value)


def test_squad_exception():
    """
    Feature: SQuADDataset.
    Description: Test file info in err msg when exception occur of SQuAD dataset.
    Expectation: Unable to read in data.
    """

    def exception_func(item):
        raise Exception("Error occur!")

    try:
        data = ds.SQuADDataset(DATASET_DIR_V1, usage='train')
        data = data.map(operations=exception_func, input_columns=["context"],
                        num_parallel_workers=1)
        for _ in data.create_dict_iterator():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" \
            in str(e)

    try:
        data = ds.SQuADDataset(DATASET_DIR_V1, usage='train')
        data = data.map(operations=exception_func, input_columns=["question"],
                        num_parallel_workers=1)
        for _ in data.create_dict_iterator():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" \
            in str(e)

    try:
        data = ds.SQuADDataset(DATASET_DIR_V1, usage='train')
        data = data.map(operations=exception_func, input_columns=["answer_start"],
                        num_parallel_workers=1)
        for _ in data.create_dict_iterator():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" \
            in str(e)

    try:
        data = ds.SQuADDataset(DATASET_DIR_V1, usage='train')
        data = data.map(operations=exception_func, input_columns=["text"],
                        num_parallel_workers=1)
        for _ in data.create_dict_iterator():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" \
            in str(e)


if __name__ == "__main__":
    test_squad_basic()
    test_squad_num_shards()
    test_squad_num_samples()
    test_squad_dataset_get_datasetsize()
    test_squad_version1()
    test_squad_version2()
    test_squad_device_que()
    test_squad_invalid_dir()
    test_squad_exception()
