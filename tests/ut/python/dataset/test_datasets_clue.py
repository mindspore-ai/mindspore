# Copyright 2020 Huawei Technologies Co., Ltd
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
    Test CLUE with repeat, skip and so on
    """
    TRAIN_FILE = '../data/dataset/testCLUE/afqmc/train.json'

    buffer = []
    data = ds.CLUEDataset(TRAIN_FILE, task='AFQMC', usage='train', shuffle=False)
    data = data.repeat(2)
    data = data.skip(3)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'label': d['label'].item().decode("utf8"),
            'sentence1': d['sentence1'].item().decode("utf8"),
            'sentence2': d['sentence2'].item().decode("utf8")
        })
    assert len(buffer) == 3


def test_clue_num_shards():
    """
    Test num_shards param of CLUE dataset
    """
    TRAIN_FILE = '../data/dataset/testCLUE/afqmc/train.json'

    buffer = []
    data = ds.CLUEDataset(TRAIN_FILE, task='AFQMC', usage='train', num_shards=3, shard_id=1)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'label': d['label'].item().decode("utf8"),
            'sentence1': d['sentence1'].item().decode("utf8"),
            'sentence2': d['sentence2'].item().decode("utf8")
        })
    assert len(buffer) == 1


def test_clue_num_samples():
    """
    Test num_samples param of CLUE dataset
    """
    TRAIN_FILE = '../data/dataset/testCLUE/afqmc/train.json'

    data = ds.CLUEDataset(TRAIN_FILE, task='AFQMC', usage='train', num_samples=2)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 2


def test_textline_dataset_get_datasetsize():
    """
    Test get_dataset_size of CLUE dataset
    """
    TRAIN_FILE = '../data/dataset/testCLUE/afqmc/train.json'

    data = ds.TextFileDataset(TRAIN_FILE)
    size = data.get_dataset_size()
    assert size == 3


def test_clue_afqmc():
    """
    Test AFQMC for train, test and evaluation
    """
    TRAIN_FILE = '../data/dataset/testCLUE/afqmc/train.json'
    TEST_FILE = '../data/dataset/testCLUE/afqmc/test.json'
    EVAL_FILE = '../data/dataset/testCLUE/afqmc/dev.json'

    # train
    buffer = []
    data = ds.CLUEDataset(TRAIN_FILE, task='AFQMC', usage='train', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'label': d['label'].item().decode("utf8"),
            'sentence1': d['sentence1'].item().decode("utf8"),
            'sentence2': d['sentence2'].item().decode("utf8")
        })
    assert len(buffer) == 3

    # test
    buffer = []
    data = ds.CLUEDataset(TEST_FILE, task='AFQMC', usage='test', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'id': d['id'],
            'sentence1': d['sentence1'].item().decode("utf8"),
            'sentence2': d['sentence2'].item().decode("utf8")
        })
    assert len(buffer) == 3

    # evaluation
    buffer = []
    data = ds.CLUEDataset(EVAL_FILE, task='AFQMC', usage='eval', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'label': d['label'].item().decode("utf8"),
            'sentence1': d['sentence1'].item().decode("utf8"),
            'sentence2': d['sentence2'].item().decode("utf8")
        })
    assert len(buffer) == 3


def test_clue_cmnli():
    """
    Test CMNLI for train, test and evaluation
    """
    TRAIN_FILE = '../data/dataset/testCLUE/cmnli/train.json'
    TEST_FILE = '../data/dataset/testCLUE/cmnli/test.json'
    EVAL_FILE = '../data/dataset/testCLUE/cmnli/dev.json'

    # train
    buffer = []
    data = ds.CLUEDataset(TRAIN_FILE, task='CMNLI', usage='train', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'label': d['label'].item().decode("utf8"),
            'sentence1': d['sentence1'].item().decode("utf8"),
            'sentence2': d['sentence2'].item().decode("utf8")
        })
    assert len(buffer) == 3

    # test
    buffer = []
    data = ds.CLUEDataset(TEST_FILE, task='CMNLI', usage='test', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'id': d['id'],
            'sentence1': d['sentence1'],
            'sentence2': d['sentence2']
        })
    assert len(buffer) == 3

    # eval
    buffer = []
    data = ds.CLUEDataset(EVAL_FILE, task='CMNLI', usage='eval', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'label': d['label'],
            'sentence1': d['sentence1'],
            'sentence2': d['sentence2']
        })
    assert len(buffer) == 3


def test_clue_csl():
    """
    Test CSL for train, test and evaluation
    """
    TRAIN_FILE = '../data/dataset/testCLUE/csl/train.json'
    TEST_FILE = '../data/dataset/testCLUE/csl/test.json'
    EVAL_FILE = '../data/dataset/testCLUE/csl/dev.json'

    # train
    buffer = []
    data = ds.CLUEDataset(TRAIN_FILE, task='CSL', usage='train', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'id': d['id'],
            'abst': d['abst'].item().decode("utf8"),
            'keyword': [i.item().decode("utf8") for i in d['keyword']],
            'label': d['label'].item().decode("utf8")
        })
    assert len(buffer) == 3

    # test
    buffer = []
    data = ds.CLUEDataset(TEST_FILE, task='CSL', usage='test', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'id': d['id'],
            'abst': d['abst'].item().decode("utf8"),
            'keyword': [i.item().decode("utf8") for i in d['keyword']],
        })
    assert len(buffer) == 3

    # eval
    buffer = []
    data = ds.CLUEDataset(EVAL_FILE, task='CSL', usage='eval', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'id': d['id'],
            'abst': d['abst'].item().decode("utf8"),
            'keyword': [i.item().decode("utf8") for i in d['keyword']],
            'label': d['label'].item().decode("utf8")
        })
    assert len(buffer) == 3


def test_clue_iflytek():
    """
    Test IFLYTEK for train, test and evaluation
    """
    TRAIN_FILE = '../data/dataset/testCLUE/iflytek/train.json'
    TEST_FILE = '../data/dataset/testCLUE/iflytek/test.json'
    EVAL_FILE = '../data/dataset/testCLUE/iflytek/dev.json'

    # train
    buffer = []
    data = ds.CLUEDataset(TRAIN_FILE, task='IFLYTEK', usage='train', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'label': d['label'].item().decode("utf8"),
            'label_des': d['label_des'].item().decode("utf8"),
            'sentence': d['sentence'].item().decode("utf8"),
        })
    assert len(buffer) == 3

    # test
    buffer = []
    data = ds.CLUEDataset(TEST_FILE, task='IFLYTEK', usage='test', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'id': d['id'],
            'sentence': d['sentence'].item().decode("utf8")
        })
    assert len(buffer) == 3

    # eval
    buffer = []
    data = ds.CLUEDataset(EVAL_FILE, task='IFLYTEK', usage='eval', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'label': d['label'].item().decode("utf8"),
            'label_des': d['label_des'].item().decode("utf8"),
            'sentence': d['sentence'].item().decode("utf8")
        })
    assert len(buffer) == 3


def test_clue_tnews():
    """
    Test TNEWS for train, test and evaluation
    """
    TRAIN_FILE = '../data/dataset/testCLUE/tnews/train.json'
    TEST_FILE = '../data/dataset/testCLUE/tnews/test.json'
    EVAL_FILE = '../data/dataset/testCLUE/tnews/dev.json'

    # train
    buffer = []
    data = ds.CLUEDataset(TRAIN_FILE, task='TNEWS', usage='train', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'label': d['label'].item().decode("utf8"),
            'label_desc': d['label_desc'].item().decode("utf8"),
            'sentence': d['sentence'].item().decode("utf8"),
            'keywords':
                d['keywords'].item().decode("utf8") if d['keywords'].size > 0 else d['keywords']
        })
    assert len(buffer) == 3

    # test
    buffer = []
    data = ds.CLUEDataset(TEST_FILE, task='TNEWS', usage='test', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'id': d['id'],
            'sentence': d['sentence'].item().decode("utf8"),
            'keywords':
                d['keywords'].item().decode("utf8") if d['keywords'].size > 0 else d['keywords']
        })
    assert len(buffer) == 3

    # eval
    buffer = []
    data = ds.CLUEDataset(EVAL_FILE, task='TNEWS', usage='eval', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'label': d['label'].item().decode("utf8"),
            'label_desc': d['label_desc'].item().decode("utf8"),
            'sentence': d['sentence'].item().decode("utf8"),
            'keywords':
                d['keywords'].item().decode("utf8") if d['keywords'].size > 0 else d['keywords']
        })
    assert len(buffer) == 3


def test_clue_wsc():
    """
    Test WSC for train, test and evaluation
    """
    TRAIN_FILE = '../data/dataset/testCLUE/wsc/train.json'
    TEST_FILE = '../data/dataset/testCLUE/wsc/test.json'
    EVAL_FILE = '../data/dataset/testCLUE/wsc/dev.json'

    # train
    buffer = []
    data = ds.CLUEDataset(TRAIN_FILE, task='WSC', usage='train')
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'span1_index': d['span1_index'],
            'span2_index': d['span2_index'],
            'span1_text': d['span1_text'].item().decode("utf8"),
            'span2_text': d['span2_text'].item().decode("utf8"),
            'idx': d['idx'],
            'label': d['label'].item().decode("utf8"),
            'text': d['text'].item().decode("utf8")
        })
    assert len(buffer) == 3

    # test
    buffer = []
    data = ds.CLUEDataset(TEST_FILE, task='WSC', usage='test')
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'span1_index': d['span1_index'],
            'span2_index': d['span2_index'],
            'span1_text': d['span1_text'].item().decode("utf8"),
            'span2_text': d['span2_text'].item().decode("utf8"),
            'idx': d['idx'],
            'text': d['text'].item().decode("utf8")
        })
    assert len(buffer) == 3

    # eval
    buffer = []
    data = ds.CLUEDataset(EVAL_FILE, task='WSC', usage='eval')
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append({
            'span1_index': d['span1_index'],
            'span2_index': d['span2_index'],
            'span1_text': d['span1_text'].item().decode("utf8"),
            'span2_text': d['span2_text'].item().decode("utf8"),
            'idx': d['idx'],
            'label': d['label'].item().decode("utf8"),
            'text': d['text'].item().decode("utf8")
        })
    assert len(buffer) == 3

def test_clue_to_device():
    """
    Test CLUE with to_device
    """
    TRAIN_FILE = '../data/dataset/testCLUE/afqmc/train.json'
    data = ds.CLUEDataset(TRAIN_FILE, task='AFQMC', usage='train', shuffle=False)
    data = data.to_device()
    data.send()


def test_clue_invalid_files():
    """
    Test CLUE with invalid files
    """
    AFQMC_DIR = '../data/dataset/testCLUE/afqmc'
    afqmc_train_json = os.path.join(AFQMC_DIR)
    with pytest.raises(ValueError) as info:
        _ = ds.CLUEDataset(afqmc_train_json, task='AFQMC', usage='train', shuffle=False)
    assert "The following patterns did not match any files" in str(info.value)
    assert AFQMC_DIR in str(info.value)


def test_clue_exception_file_path():
    """
    Test file info in err msg when exception occur of CLUE dataset
    """
    TRAIN_FILE = '../data/dataset/testCLUE/afqmc/train.json'
    def exception_func(item):
        raise Exception("Error occur!")

    try:
        data = ds.CLUEDataset(TRAIN_FILE, task='AFQMC', usage='train')
        data = data.map(operations=exception_func, input_columns=["label"], num_parallel_workers=1)
        for _ in data.create_dict_iterator():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data = ds.CLUEDataset(TRAIN_FILE, task='AFQMC', usage='train')
        data = data.map(operations=exception_func, input_columns=["sentence1"], num_parallel_workers=1)
        for _ in data.create_dict_iterator():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data = ds.CLUEDataset(TRAIN_FILE, task='AFQMC', usage='train')
        data = data.map(operations=exception_func, input_columns=["sentence2"], num_parallel_workers=1)
        for _ in data.create_dict_iterator():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)


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
    test_clue_to_device()
    test_clue_invalid_files()
    test_clue_exception_file_path()
