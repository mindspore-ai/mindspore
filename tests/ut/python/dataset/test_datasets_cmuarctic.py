"""
Test CmuArctic dataset operators
"""
import os
import pytest
import numpy as np
import matplotlib.pyplot as plt
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as vision
from mindspore import log as logger

DATA_DIR = "/home/user06/zjm/data/cmu_arctic/"

def test_cmuarctic_basic():
    """
    Validate CmuarcticDataset
    """
    logger.info("Test CmuArcticDataset Op")

    # case 1: test loading fault dataset
    data1 = ds.CmuArcticDataset(DATA_DIR)
    num_iter1 = 0
    for _ in data1.create_dict_iterator( output_numpy=True,num_epochs=1):
        num_iter1 += 1
    assert num_iter1 == 1132

    # case 2: test num_samples
    data2 = ds.CmuArcticDataset(DATA_DIR, num_samples=500)
    num_iter2 = 0
    for _ in data2.create_dict_iterator( output_numpy=True,num_epochs=1):
        num_iter2 += 1
    assert num_iter2 == 500

    # case 3: test repeat
    data3 = ds.CmuArcticDataset(DATA_DIR, num_samples=200)
    data3 = data3.repeat(5)
    num_iter3 = 0
    for _ in data3.create_dict_iterator( output_numpy=True,num_epochs=1):
        num_iter3 += 1
    assert num_iter3 == 1000

    # case 4: test batch with drop_remainder=False
    data4 = ds.CmuArcticDataset(DATA_DIR, num_samples=100)
    assert data4.get_dataset_size() == 100
    assert data4.get_batch_size() == 1
    data4 = data4.batch(batch_size=7)  # drop_remainder is default to be False
    assert data4.get_dataset_size() == 15
    assert data4.get_batch_size() == 7
    # num_iter4 = 0
    # for _ in data4.create_dict_iterator( output_numpy=True,num_epochs=1):
    #     num_iter4 += 1
    # assert num_iter4 == 15

    # case 5: test batch with drop_remainder=True
    data5 = ds.CmuArcticDataset(DATA_DIR, num_samples=100)
    assert data5.get_dataset_size() == 100
    assert data5.get_batch_size() == 1
    data5 = data5.batch(batch_size=7, drop_remainder=True)  # the rest of incomplete batch will be dropped
    assert data5.get_dataset_size() == 14
    assert data5.get_batch_size() == 7
    # num_iter5 = 0
    # for _ in data5.create_dict_iterator( output_numpy=True,num_epochs=1):
    #     num_iter5 += 1
    # assert num_iter5 == 14



def test_cmu_arctic_sequential_sampler():
    """
    Test CmuArcticDataset with SequentialSampler
    """
    logger.info("Test CmuArcticDataset Op with SequentialSampler")
    num_samples = 50
    sampler = ds.SequentialSampler(num_samples=num_samples)
    data1 = ds.CmuArcticDataset(DATA_DIR, sampler=sampler)
    data2 = ds.CmuArcticDataset(DATA_DIR, shuffle=False, num_samples=num_samples)
    label_list1, label_list2 = [], []
    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator( output_numpy=True,num_epochs=1), data2.create_dict_iterator( output_numpy=True,num_epochs=1)):
        label_list1.append(item1["utterance"])
        label_list2.append(item2["utterance"])
        num_iter += 1
    np.testing.assert_array_equal(label_list1, label_list2)
    assert num_iter == num_samples


def test_cmu_arctic_exception():
    """
    Test error cases for CmuArcticDataset
    """
    logger.info("Test error cases for CmuArcticDataset")
    error_msg_1 = "sampler and shuffle cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_1):
        ds.CmuArcticDataset(DATA_DIR, shuffle=False, sampler=ds.PKSampler(3))

    error_msg_2 = "sampler and sharding cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_2):
        ds.CmuArcticDataset(DATA_DIR, sampler=ds.PKSampler(3), num_shards=2, shard_id=0)

    error_msg_3 = "num_shards is specified and currently requires shard_id as well"
    with pytest.raises(RuntimeError, match=error_msg_3):
        ds.CmuArcticDataset(DATA_DIR, num_shards=10)

    error_msg_4 = "shard_id is specified but num_shards is not"
    with pytest.raises(RuntimeError, match=error_msg_4):
        ds.CmuArcticDataset(DATA_DIR, shard_id=0)

    error_msg_5 = "Input shard_id is not within the required interval"
    with pytest.raises(ValueError, match=error_msg_5):
        ds.CmuArcticDataset(DATA_DIR, num_shards=5, shard_id=-1)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.CmuArcticDataset(DATA_DIR, num_shards=5, shard_id=5)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.CmuArcticDataset(DATA_DIR, num_shards=2, shard_id=5)

    error_msg_6 = "num_parallel_workers exceeds"
    with pytest.raises(ValueError, match=error_msg_6):
        ds.CmuArcticDataset(DATA_DIR, shuffle=False, num_parallel_workers=0)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.CmuArcticDataset(DATA_DIR, shuffle=False, num_parallel_workers=256)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.CmuArcticDataset(DATA_DIR, shuffle=False, num_parallel_workers=-2)

    error_msg_7 = "Argument shard_id"
    with pytest.raises(TypeError, match=error_msg_7):
        ds.CmuArcticDataset(DATA_DIR, num_shards=2, shard_id="0")

    def exception_func(item):
        raise Exception("Error occur!")

    error_msg_8 = "The corresponding data files"
    with pytest.raises(RuntimeError, match=error_msg_8):
        data = ds.CmuArcticDataset(DATA_DIR)
        data = data.map(operations=exception_func, input_columns=["waveform"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
    with pytest.raises(RuntimeError, match=error_msg_8):
        data = ds.CmuArcticDataset(DATA_DIR)
        data = data.map(operations=vision.Decode(), input_columns=["waveform"], num_parallel_workers=1)
        data = data.map(operations=exception_func, input_columns=["waveform"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
    with pytest.raises(RuntimeError, match=error_msg_8):
        data = ds.CmuArcticDataset(DATA_DIR)
        data = data.map(operations=exception_func, input_columns=["waveform"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass


def test_cmu_arctic_visualize(plot=False):
    """
    Visualize CmuArcticDataset results
    """
    logger.info("Test CmuArcticDataset visualization")

    data1 = ds.CmuArcticDataset(DATA_DIR, num_samples=10, shuffle=False)
    num_iter = 0
    for item in data1.create_dict_iterator( num_epochs=1, output_numpy=True):
        audio = item["waveform"]
        sample_rate = item["sample_rate"]
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float64
        assert sample_rate.dtype == np.uint32
        num_iter += 1
    assert num_iter == 10


def test_cmu_arctic_usage():
    """
    Validate CmuArcticDataset audio readings
    """
    logger.info("Test CmuArcticDataset usage flag")

    def test_config(usage, cmu_arctic_path=None):
        cmu_arctic_path = DATA_DIR if cmu_arctic_path is None else cmu_arctic_path
        try:
            data = ds.CmuArcticDataset(cmu_arctic_path, usage=usage, shuffle=False)
            num_rows = 0
            for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                num_rows += 1
        except (ValueError, TypeError, RuntimeError) as e:
            return str(e)
        return num_rows

    assert test_config("aew") == 1132
    assert test_config("ahw") == 593
    assert "Input usage is not within the valid set of ['aew', 'ahw', 'aup', 'awb', 'axb', 'bdl', 'clb', 'eey', 'fem', 'gka', 'jmk', 'ksp', 'ljm', 'lnh', 'rms', 'rxr', 'slp', 'slt']." in test_config("invalid")
    assert "Argument usage with value ['list'] is not of type [<class 'str'>]" in test_config(["list"])

    all_files_path = None
    if all_files_path is not None:
        assert test_config("aew", all_files_path) == 1132
        assert test_config("ahw", all_files_path) == 593
        assert ds.cmu_arcticDataset(all_files_path, usage="aew").get_dataset_size() == 1132
        assert ds.cmu_arcticDataset(all_files_path, usage="ahw").get_dataset_size() == 593


if __name__ == '__main__':
    test_cmuarctic_basic()
    test_cmu_arctic_sequential_sampler()
    test_cmu_arctic_exception()
    test_cmu_arctic_visualize(plot=True)
    test_cmu_arctic_usage()
