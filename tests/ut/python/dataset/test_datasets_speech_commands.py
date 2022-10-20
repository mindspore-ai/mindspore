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
"""
Test SpeechCommands dataset operations
"""
import pytest
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.audio as audio
from mindspore import log as logger

DATA_DIR = "../data/dataset/testSpeechCommandsData/"


def test_speech_commands_basic():
    """
    Feature: SpeechCommands Dataset
    Description: Read all files
    Expectation: Output the amount of files
    """
    logger.info("Test SpeechCommandsDataset Op.")

    # case 1: test loading whole dataset
    data1 = ds.SpeechCommandsDataset(DATA_DIR)
    num_iter1 = 0
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter1 += 1
    assert num_iter1 == 3

    # case 2: test num_samples
    data2 = ds.SpeechCommandsDataset(DATA_DIR, num_samples=3)
    num_iter2 = 0
    for _ in data2.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter2 += 1
    assert num_iter2 == 3

    # case 3: test repeat
    data3 = ds.SpeechCommandsDataset(DATA_DIR, num_samples=2)
    data3 = data3.repeat(5)
    num_iter3 = 0
    for _ in data3.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter3 += 1
    assert num_iter3 == 10


def test_speech_commands_sequential_sampler():
    """
    Feature: SpeechCommands Dataset
    Description: Use SequentialSampler to sample data.
    Expectation: The number of samplers returned by dict_iterator is equal to the requested number of samples.
    """
    logger.info("Test SpeechCommandsDataset with SequentialSampler.")
    num_samples = 2
    sampler = ds.SequentialSampler(num_samples=num_samples)
    data1 = ds.SpeechCommandsDataset(DATA_DIR, sampler=sampler)
    data2 = ds.SpeechCommandsDataset(DATA_DIR, shuffle=False, num_samples=num_samples)
    sample_rate_list1, sample_rate_list2 = [], []
    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        sample_rate_list1.append(item1["sample_rate"])
        sample_rate_list2.append(item2["sample_rate"])
        num_iter += 1
    np.testing.assert_array_equal(sample_rate_list1, sample_rate_list2)
    assert num_iter == num_samples


def test_speech_commands_exception():
    """
    Feature: SpeechCommands Dataset
    Description: Test error cases for SpeechCommandsDataset
    Expectation: Error message
    """
    logger.info("Test error cases for SpeechCommandsDataset.")
    error_msg_1 = "sampler and shuffle cannot be specified at the same time."
    with pytest.raises(RuntimeError, match=error_msg_1):
        ds.SpeechCommandsDataset(DATA_DIR, shuffle=False, sampler=ds.PKSampler(3))

    error_msg_2 = "sampler and sharding cannot be specified at the same time."
    with pytest.raises(RuntimeError, match=error_msg_2):
        ds.SpeechCommandsDataset(DATA_DIR, sampler=ds.PKSampler(3), num_shards=2, shard_id=0)

    error_msg_3 = "num_shards is specified and currently requires shard_id as well."
    with pytest.raises(RuntimeError, match=error_msg_3):
        ds.SpeechCommandsDataset(DATA_DIR, num_shards=10)

    error_msg_4 = "shard_id is specified but num_shards is not."
    with pytest.raises(RuntimeError, match=error_msg_4):
        ds.SpeechCommandsDataset(DATA_DIR, shard_id=0)

    error_msg_5 = "Input shard_id is not within the required interval."
    with pytest.raises(ValueError, match=error_msg_5):
        ds.SpeechCommandsDataset(DATA_DIR, num_shards=5, shard_id=-1)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.SpeechCommandsDataset(DATA_DIR, num_shards=5, shard_id=5)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.SpeechCommandsDataset(DATA_DIR, num_shards=2, shard_id=5)

    error_msg_6 = "num_parallel_workers exceeds."
    with pytest.raises(ValueError, match=error_msg_6):
        ds.SpeechCommandsDataset(DATA_DIR, shuffle=False, num_parallel_workers=0)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.SpeechCommandsDataset(DATA_DIR, shuffle=False, num_parallel_workers=256)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.SpeechCommandsDataset(DATA_DIR, shuffle=False, num_parallel_workers=-2)

    error_msg_7 = "Argument shard_id."
    with pytest.raises(TypeError, match=error_msg_7):
        ds.SpeechCommandsDataset(DATA_DIR, num_shards=2, shard_id="0")

    def exception_func(item):
        raise Exception("Error occur!")

    error_msg_8 = "The corresponding data file is."
    with pytest.raises(RuntimeError, match=error_msg_8):
        data = ds.SpeechCommandsDataset(DATA_DIR)
        data = data.map(operations=exception_func, input_columns=["waveform"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
    with pytest.raises(RuntimeError, match=error_msg_8):
        data = ds.SpeechCommandsDataset(DATA_DIR)
        data = data.map(operations=exception_func, input_columns=["sample_rate"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass


def test_speech_commands_usage():
    """
    Feature: SpeechCommands Dataset
    Description: Usage Test
    Expectation: Get the result of each function
    """
    logger.info("Test SpeechCommandsDataset usage flag.")

    def test_config(usage, speech_commands_path=DATA_DIR):
        try:
            data = ds.SpeechCommandsDataset(speech_commands_path, usage=usage)
            num_rows = 0
            for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                num_rows += 1
        except (ValueError, TypeError, RuntimeError) as e:
            return str(e)
        return num_rows

    # test the usage of SpeechCommands
    assert test_config("test") == 1
    assert test_config("train") == 1
    assert test_config("valid") == 1
    assert test_config("all") == 3
    assert "usage is not within the valid set of ['train', 'test', 'valid', 'all']." in test_config("invalid")

    # change this directory to the folder that contains all SpeechCommands files
    all_speech_commands = None
    if all_speech_commands is not None:
        assert test_config("test", all_speech_commands) == 11005
        assert test_config("valid", all_speech_commands) == 9981
        assert test_config("train", all_speech_commands) == 84843
        assert test_config("all", all_speech_commands) == 105829
        assert ds.SpeechCommandsDataset(all_speech_commands, usage="test").get_dataset_size() == 11005
        assert ds.SpeechCommandsDataset(all_speech_commands, usage="valid").get_dataset_size() == 9981
        assert ds.SpeechCommandsDataset(all_speech_commands, usage="train").get_dataset_size() == 84843
        assert ds.SpeechCommandsDataset(all_speech_commands, usage="all").get_dataset_size() == 105829


def test_speech_commands_pipeline():
    """
    Feature: Pipeline test
    Description: Read a sample
    Expectation: Test BandBiquad by pipeline
    """
    dataset = ds.SpeechCommandsDataset(DATA_DIR, num_samples=1)
    band_biquad_op = audio.BandBiquad(8000, 200.0)
    # Filtered waveform by bandbiquad
    dataset = dataset.map(input_columns=["waveform"], operations=band_biquad_op, num_parallel_workers=4)
    i = 0
    for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        i += 1
    assert i == 1


if __name__ == '__main__':
    test_speech_commands_basic()
    test_speech_commands_sequential_sampler()
    test_speech_commands_exception()
    test_speech_commands_usage()
    test_speech_commands_pipeline()
