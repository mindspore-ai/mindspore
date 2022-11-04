# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
import os
import pytest
import mindspore.dataset as ds


def test_clue():
    """
    Feature: CLUEDataset
    Description: Test CLUEDataset with repeat, skip, and so on
    Expectation: The dataset is processed as expected
    """
    train_file = '../data/dataset/testCLUE/afqmc/train.json'

    buffer = []
    data = ds.CLUEDataset(train_file, task='AFQMC', usage='train', shuffle=False)
    data = data.repeat(2)
    data = data.skip(3)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'label': d['label'],
            'sentence1': d['sentence1'],
            'sentence2': d['sentence2']
        })
    assert len(buffer) == 3


def test_clue_num_shards():
    """
    Feature: CLUEDataset
    Description: Test num_shards as parameter of CLUEDataset
    Expectation: The dataset is processed as expected
    """
    train_file = '../data/dataset/testCLUE/afqmc/train.json'

    buffer = []
    data = ds.CLUEDataset(train_file, task='AFQMC', usage='train', num_shards=3, shard_id=1)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'label': d['label'],
            'sentence1': d['sentence1'],
            'sentence2': d['sentence2']
        })
    assert len(buffer) == 1


def test_clue_num_samples():
    """
    Feature: CLUEDataset
    Description: Test num_samples as parameter of CLUEDataset
    Expectation: The dataset is processed as expected
    """
    train_file = '../data/dataset/testCLUE/afqmc/train.json'

    data = ds.CLUEDataset(train_file, task='AFQMC', usage='train', num_samples=2)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 2


def test_textline_dataset_get_datasetsize():
    """
    Feature: CLUEDataset
    Description: Test get_dataset_size of CLUEDataset
    Expectation: The dataset is processed as expected
    """
    train_file = '../data/dataset/testCLUE/afqmc/train.json'

    data = ds.TextFileDataset(train_file)
    size = data.get_dataset_size()
    assert size == 3


def test_clue_afqmc():
    """
    Feature: CLUEDataset
    Description: Test AFQMC for train, test, and evaluation
    Expectation: The dataset is processed as expected
    """
    train_file = '../data/dataset/testCLUE/afqmc/train.json'
    test_file = '../data/dataset/testCLUE/afqmc/test.json'
    eval_file = '../data/dataset/testCLUE/afqmc/dev.json'

    # train
    buffer = []
    data = ds.CLUEDataset(train_file, task='AFQMC', usage='train', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'label': d['label'],
            'sentence1': d['sentence1'],
            'sentence2': d['sentence2']
        })
    assert len(buffer) == 3

    # test
    buffer = []
    data = ds.CLUEDataset(test_file, task='AFQMC', usage='test', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'id': d['id'],
            'sentence1': d['sentence1'],
            'sentence2': d['sentence2']
        })
    assert len(buffer) == 3

    # evaluation
    buffer = []
    data = ds.CLUEDataset(eval_file, task='AFQMC', usage='eval', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'label': d['label'],
            'sentence1': d['sentence1'],
            'sentence2': d['sentence2']
        })
    assert len(buffer) == 3


def test_clue_cmnli():
    """
    Feature: CLUEDataset
    Description: Test CMNLI for train, test, and evaluation
    Expectation: The dataset is processed as expected
    """
    train_file = '../data/dataset/testCLUE/cmnli/train.json'
    test_file = '../data/dataset/testCLUE/cmnli/test.json'
    eval_file = '../data/dataset/testCLUE/cmnli/dev.json'

    # train
    buffer = []
    data = ds.CLUEDataset(train_file, task='CMNLI', usage='train', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'label': d['label'],
            'sentence1': d['sentence1'],
            'sentence2': d['sentence2']
        })
    assert len(buffer) == 3

    # test
    buffer = []
    data = ds.CLUEDataset(test_file, task='CMNLI', usage='test', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'id': d['id'],
            'sentence1': d['sentence1'],
            'sentence2': d['sentence2']
        })
    assert len(buffer) == 3

    # eval
    buffer = []
    data = ds.CLUEDataset(eval_file, task='CMNLI', usage='eval', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'label': d['label'],
            'sentence1': d['sentence1'],
            'sentence2': d['sentence2']
        })
    assert len(buffer) == 3


def test_clue_csl():
    """
    Feature: CLUEDataset
    Description: Test CSL for train, test, and evaluation
    Expectation: The dataset is processed as expected
    """
    train_file = '../data/dataset/testCLUE/csl/train.json'
    test_file = '../data/dataset/testCLUE/csl/test.json'
    eval_file = '../data/dataset/testCLUE/csl/dev.json'

    # train
    buffer = []
    data = ds.CLUEDataset(train_file, task='CSL', usage='train', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'id': d['id'],
            'abst': d['abst'],
            'keyword': [i for i in d['keyword']],
            'label': d['label']
        })
    assert len(buffer) == 3

    # test
    buffer = []
    data = ds.CLUEDataset(test_file, task='CSL', usage='test', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'id': d['id'],
            'abst': d['abst'],
            'keyword': [i for i in d['keyword']],
        })
    assert len(buffer) == 3

    # eval
    buffer = []
    data = ds.CLUEDataset(eval_file, task='CSL', usage='eval', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'id': d['id'],
            'abst': d['abst'],
            'keyword': [i for i in d['keyword']],
            'label': d['label']
        })
    assert len(buffer) == 3


def test_clue_iflytek():
    """
    Feature: CLUEDataset
    Description: Test IFLYTEK for train, test, and evaluation
    Expectation: The dataset is processed as expected
    """
    train_file = '../data/dataset/testCLUE/iflytek/train.json'
    test_file = '../data/dataset/testCLUE/iflytek/test.json'
    eval_file = '../data/dataset/testCLUE/iflytek/dev.json'

    # train
    buffer = []
    data = ds.CLUEDataset(train_file, task='IFLYTEK', usage='train', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'label': d['label'],
            'label_des': d['label_des'],
            'sentence': d['sentence'],
        })
    assert len(buffer) == 3

    # test
    buffer = []
    data = ds.CLUEDataset(test_file, task='IFLYTEK', usage='test', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'id': d['id'],
            'sentence': d['sentence']
        })
    assert len(buffer) == 3

    # eval
    buffer = []
    data = ds.CLUEDataset(eval_file, task='IFLYTEK', usage='eval', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'label': d['label'],
            'label_des': d['label_des'],
            'sentence': d['sentence']
        })
    assert len(buffer) == 3


def test_clue_tnews():
    """
    Feature: CLUEDataset
    Description: Test TNEWS for train, test, and evaluation
    Expectation: The dataset is processed as expected
    """
    train_file = '../data/dataset/testCLUE/tnews/train.json'
    test_file = '../data/dataset/testCLUE/tnews/test.json'
    eval_file = '../data/dataset/testCLUE/tnews/dev.json'

    # train
    buffer = []
    data = ds.CLUEDataset(train_file, task='TNEWS', usage='train', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'label': d['label'],
            'label_desc': d['label_desc'],
            'sentence': d['sentence'],
            'keywords':
                d['keywords'] if d['keywords'].size > 0 else d['keywords']
        })
    assert len(buffer) == 3

    # test
    buffer = []
    data = ds.CLUEDataset(test_file, task='TNEWS', usage='test', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'id': d['id'],
            'sentence': d['sentence'],
            'keywords':
                d['keywords'] if d['keywords'].size > 0 else d['keywords']
        })
    assert len(buffer) == 3

    # eval
    buffer = []
    data = ds.CLUEDataset(eval_file, task='TNEWS', usage='eval', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'label': d['label'],
            'label_desc': d['label_desc'],
            'sentence': d['sentence'],
            'keywords':
                d['keywords'] if d['keywords'].size > 0 else d['keywords']
        })
    assert len(buffer) == 3


def test_clue_wsc():
    """
    Feature: CLUEDataset
    Description: Test WSC for train, test, and evaluation
    Expectation: The dataset is processed as expected
    """
    train_file = '../data/dataset/testCLUE/wsc/train.json'
    test_file = '../data/dataset/testCLUE/wsc/test.json'
    eval_file = '../data/dataset/testCLUE/wsc/dev.json'

    # train
    buffer = []
    data = ds.CLUEDataset(train_file, task='WSC', usage='train')
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'span1_index': d['span1_index'],
            'span2_index': d['span2_index'],
            'span1_text': d['span1_text'],
            'span2_text': d['span2_text'],
            'idx': d['idx'],
            'label': d['label'],
            'text': d['text']
        })
    assert len(buffer) == 3

    # test
    buffer = []
    data = ds.CLUEDataset(test_file, task='WSC', usage='test')
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'span1_index': d['span1_index'],
            'span2_index': d['span2_index'],
            'span1_text': d['span1_text'],
            'span2_text': d['span2_text'],
            'idx': d['idx'],
            'text': d['text']
        })
    assert len(buffer) == 3

    # eval
    buffer = []
    data = ds.CLUEDataset(eval_file, task='WSC', usage='eval')
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'span1_index': d['span1_index'],
            'span2_index': d['span2_index'],
            'span1_text': d['span1_text'],
            'span2_text': d['span2_text'],
            'idx': d['idx'],
            'label': d['label'],
            'text': d['text']
        })
    assert len(buffer) == 3


def test_clue_device_que():
    """
    Feature: CLUEDataset
    Description: Test CLUEDataset with device_que
    Expectation: The dataset is processed as expected
    """
    train_file = '../data/dataset/testCLUE/afqmc/train.json'
    data = ds.CLUEDataset(train_file, task='AFQMC', usage='train', shuffle=False)
    data = data.device_que()
    data.send()


def test_clue_invalid_files():
    """
    Feature: CLUEDataset
    Description: Test CLUE with invalid files
    Expectation: Error is raised as expected
    """
    afqmc_dir = '../data/dataset/testCLUE/afqmc'
    afqmc_train_json = os.path.join(afqmc_dir)
    with pytest.raises(ValueError) as info:
        _ = ds.CLUEDataset(afqmc_train_json, task='AFQMC', usage='train', shuffle=False)
    assert "The following patterns did not match any files" in str(info.value)
    assert afqmc_dir in str(info.value)


def test_clue_exception_file_path():
    """
    Feature: CLUEDataset
    Description: Test file info in error message when exception occurred for CLUEDataset
    Expectation: Throw correct error as expected
    """
    train_file = '../data/dataset/testCLUE/afqmc/train.json'
    def exception_func(item):
        raise Exception("Error occur!")

    try:
        data = ds.CLUEDataset(train_file, task='AFQMC', usage='train')
        data = data.map(operations=exception_func, input_columns=["label"], num_parallel_workers=1)
        for _ in data.create_dict_iterator(num_epochs=1):
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    try:
        data = ds.CLUEDataset(train_file, task='AFQMC', usage='train')
        data = data.map(operations=exception_func, input_columns=["sentence1"], num_parallel_workers=1)
        for _ in data.create_dict_iterator(num_epochs=1):
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    try:
        data = ds.CLUEDataset(train_file, task='AFQMC', usage='train')
        data = data.map(operations=exception_func, input_columns=["sentence2"], num_parallel_workers=1)
        for _ in data.create_dict_iterator(num_epochs=1):
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)


if __name__ == "__main__":
    test_clue()
    test_clue_num_shards()
    test_clue_num_samples()
    test_textline_dataset_get_datasetsize()
    test_clue_afqmc()
    test_clue_cmnli()
    test_clue_csl()
    test_clue_iflytek()
    test_clue_tnews()
    test_clue_wsc()
    test_clue_device_que()
    test_clue_invalid_files()
    test_clue_exception_file_path()
