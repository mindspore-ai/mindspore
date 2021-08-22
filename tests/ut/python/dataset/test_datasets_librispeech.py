"""
Test Librispeech dataset operators
"""
import pytest
import numpy as np
import matplotlib.pyplot as plt
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as vision
from mindspore import log as logger

DATA_DIR = "/home/user06/zjm/data/libri_speech/LibriSpeech/"


def test_librispeech_basic():
    """
    Validate LibriSpeechDataset
    """
    logger.info("Test LibriSpeechDataset Op")

    # case 1: test loading fault dataset
    data1 = ds.LibriSpeechDataset(DATA_DIR)
    num_iter1 = 0
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter1 += 1
    assert num_iter1 == 2939

    # case 2: test num_samples
    data2 = ds.LibriSpeechDataset(DATA_DIR, num_samples=500)
    num_iter2 = 0
    for _ in data2.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter2 += 1
    assert num_iter2 == 500

    # case 3: test repeat
    data3 = ds.LibriSpeechDataset(DATA_DIR, num_samples=200)
    data3 = data3.repeat(5)
    num_iter3 = 0
    for _ in data3.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter3 += 1
    assert num_iter3 == 1000

    # case 4: test batch with drop_remainder=False
    data4 = ds.LibriSpeechDataset(DATA_DIR, num_samples=100)
    assert data4.get_dataset_size() == 100
    assert data4.get_batch_size() == 1
    data4 = data4.batch(batch_size=7)  # drop_remainder is default to be False
    assert data4.get_dataset_size() == 15
    assert data4.get_batch_size() == 7
    # num_iter4 = 0
    # for _ in data4.create_dict_iterator(num_epochs=1,output_numpy=True):
    #     num_iter4 += 1
    # assert num_iter4 == 15

    # case 5: test batch with drop_remainder=True
    data5 = ds.LibriSpeechDataset(DATA_DIR, num_samples=100)
    assert data5.get_dataset_size() == 100
    assert data5.get_batch_size() == 1
    data5 = data5.batch(batch_size=7, drop_remainder=True)  # the rest of incomplete batch will be dropped
    assert data5.get_dataset_size() == 14
    assert data5.get_batch_size() == 7
    # num_iter5 = 0
    # for _ in data5.create_dict_iterator(num_epochs=1,output_numpy=True):
    #     num_iter5 += 1
    # assert num_iter5 == 14


def test_librispeech_sequential_sampler():
    """
    Test LibriSpeechDataset with SequentialSampler
    """
    logger.info("Test LibriSpeechDataset Op with SequentialSampler")
    num_samples = 50
    sampler = ds.SequentialSampler(num_samples=num_samples)
    data1 = ds.LibriSpeechDataset(DATA_DIR, sampler=sampler)
    data2 = ds.LibriSpeechDataset(DATA_DIR, shuffle=False, num_samples=num_samples)
    label_list1, label_list2 = [], []
    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        label_list1.append(item1["utterance"])
        label_list2.append(item2["utterance"])
        num_iter += 1
    np.testing.assert_array_equal(label_list1, label_list2)
    assert num_iter == num_samples


def test_librispeech_exception():
    """
    Test error cases for LibriSpeechDataset
    """
    logger.info("Test error cases for LibriSpeechDataset")
    error_msg_1 = "sampler and shuffle cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_1):
        ds.LibriSpeechDataset(DATA_DIR, shuffle=False, sampler=ds.PKSampler(3))

    error_msg_2 = "sampler and sharding cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_2):
        ds.LibriSpeechDataset(DATA_DIR, sampler=ds.PKSampler(3), num_shards=2, shard_id=0)

    error_msg_3 = "num_shards is specified and currently requires shard_id as well"
    with pytest.raises(RuntimeError, match=error_msg_3):
        ds.LibriSpeechDataset(DATA_DIR, num_shards=10)

    error_msg_4 = "shard_id is specified but num_shards is not"
    with pytest.raises(RuntimeError, match=error_msg_4):
        ds.LibriSpeechDataset(DATA_DIR, shard_id=0)

    error_msg_5 = "Input shard_id is not within the required interval"
    with pytest.raises(ValueError, match=error_msg_5):
        ds.LibriSpeechDataset(DATA_DIR, num_shards=5, shard_id=-1)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.LibriSpeechDataset(DATA_DIR, num_shards=5, shard_id=5)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.LibriSpeechDataset(DATA_DIR, num_shards=2, shard_id=5)

    error_msg_6 = "num_parallel_workers exceeds"
    with pytest.raises(ValueError, match=error_msg_6):
        ds.LibriSpeechDataset(DATA_DIR, shuffle=False, num_parallel_workers=0)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.LibriSpeechDataset(DATA_DIR, shuffle=False, num_parallel_workers=256)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.LibriSpeechDataset(DATA_DIR, shuffle=False, num_parallel_workers=-2)

    error_msg_7 = "Argument shard_id"
    with pytest.raises(TypeError, match=error_msg_7):
        ds.LibriSpeechDataset(DATA_DIR, num_shards=2, shard_id="0")

    def exception_func(item):
        raise Exception("Error occur!")

    error_msg_8 = "The corresponding data files"
    with pytest.raises(RuntimeError, match=error_msg_8):
        data = ds.LibriSpeechDataset(DATA_DIR)
        data = data.map(operations=exception_func, input_columns=["waveform"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
    with pytest.raises(RuntimeError, match=error_msg_8):
        data = ds.LibriSpeechDataset(DATA_DIR)
        data = data.map(operations=vision.Decode(), input_columns=["waveform"], num_parallel_workers=1)
        data = data.map(operations=exception_func, input_columns=["waveform"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
    with pytest.raises(RuntimeError, match=error_msg_8):
        data = ds.LibriSpeechDataset(DATA_DIR)
        data = data.map(operations=exception_func, input_columns=["waveform"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass


def test_librispeech_visualize(plot=False):
    """
    Visualize LibriSpeechDataset results
    """
    logger.info("Test LibriSpeechDataset visualization")

    data1 = ds.LibriSpeechDataset(DATA_DIR, num_samples=10, shuffle=False)
    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        audio = item["waveform"]
        sample_rate = item["sample_rate"]
        speaker_id = item["speaker_id"];
        chapter_id = item["chapter_id"];
        utterance_id = item["utterance_id"];
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float64
        assert sample_rate.dtype == np.uint32
        assert speaker_id.dtype == np.uint32
        assert chapter_id.dtype == np.uint32
        assert utterance_id.dtype == np.uint32
        num_iter += 1
    assert num_iter == 10


def test_librispeech_usage():
    """
    Validate LibriSpeechDataset audio readings
    """
    logger.info("Test LibriSpeechDataset usage flag")

    def test_config(usage, librispeech_path=None):
        librispeech_path = DATA_DIR if librispeech_path is None else librispeech_path
        try:
            data = ds.LibriSpeechDataset(librispeech_path, usage=usage, shuffle=False)
            num_rows = 0
            for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                num_rows += 1
        except (ValueError, TypeError, RuntimeError) as e:
            return str(e)
        return num_rows

    assert test_config("dev-clean") == 2703
    assert test_config("dev-other") == 2864
    assert "Input usage is not within the valid set of ['dev-clean', 'dev-other', 'test-clean', 'test-other', 'train-clean-100', 'train-clean-360', 'train-other-500']." in test_config("invalid")
    assert "Argument usage with value ['list'] is not of type [<class 'str'>]" in test_config(["list"])

    all_files_path = None
    if all_files_path is not None:
        assert test_config("dev-clean", all_files_path) == 2703
        assert test_config("dev-other", all_files_path) == 2864
        assert ds.LibriSpeechDataset(all_files_path, usage="dev-clean").get_dataset_size() == 2703
        assert ds.LibrispeechDataset(all_files_path, usage="dev-other").get_dataset_size() == 2864


if __name__ == '__main__':
    test_librispeech_basic()#pass
    test_librispeech_sequential_sampler()#pass
    test_librispeech_exception()#pass
    test_librispeech_visualize(plot=True)#pass
    test_librispeech_usage()#pass
