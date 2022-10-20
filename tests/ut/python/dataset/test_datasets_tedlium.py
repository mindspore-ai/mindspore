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
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.audio as audio

DATA_DIR_TEDLIUM_RELEASE12 = "../data/dataset/testTedliumData/TEDLIUM_release1"
DATA_DIR_TEDLIUM_RELEASE3 = "../data/dataset/testTedliumData/TEDLIUM_release3"
RELEASE1 = "release1"
RELEASE2 = "release2"
RELEASE3 = "release3"

NO_SPH_DIR_TEDLIUM12 = "../data/dataset/testTedliumData/else"


def test_tedlium_basic():
    """
    Feature: TedliumDataset
    Description: Use different data to test the functions of different versions
    Expectation: num_samples
                        set     1   2   4
                        get     1   2   4
                num_parallel_workers
                        set     1   2   4(num_samples=4)
                        get     4  4  4
                num repeat
                        set     3(num_samples=5)
                        get     15
    """
    # case1 test num_samples
    data11 = ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE12, RELEASE1, num_samples=1)
    data12 = ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE12, RELEASE2, num_samples=2)
    data13 = ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE3, RELEASE3, num_samples=4)
    num_iter11 = 0
    num_iter12 = 0
    num_iter13 = 0
    for _ in data11.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter11 += 1

    for _ in data12.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter12 += 1

    for _ in data13.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter13 += 1

    assert num_iter11 == 1
    assert num_iter12 == 2
    assert num_iter13 == 4

    # case2 test num_parallel_workers
    data21 = ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE12, RELEASE1, num_samples=4, num_parallel_workers=1)
    data22 = ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE12, RELEASE2, num_samples=4, num_parallel_workers=2)
    data23 = ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE3, RELEASE3, num_samples=4, num_parallel_workers=4)
    num_iter21 = 0
    num_iter22 = 0
    num_iter23 = 0
    for _ in data21.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter21 += 1

    for _ in data22.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter22 += 1

    for _ in data23.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter23 += 1

    assert num_iter21 == 4
    assert num_iter22 == 4
    assert num_iter23 == 4

    # case3 test repeat
    data3 = ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE12, RELEASE1, num_samples=5)
    data3 = data3.repeat(3)
    num_iter3 = 0
    for _ in data3.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter3 += 1

    assert num_iter3 == 15


def test_tedlium_content_check():
    """
    Feature: TedliumDataset
    Description: Check content of the first sample
    Expectation: Correct content
    """
    data1 = ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE12, RELEASE1, num_samples=1, shuffle=False)
    data3 = ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE3, RELEASE3, num_samples=1, shuffle=False)
    num_iter1 = 0
    num_iter3 = 0
    for d in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        waveform = d["waveform"]
        sample_rate = d["sample_rate"]
        transcript = d["transcript"]
        talk_id = d["talk_id"]
        speaker_id = d["speaker_id"]
        identifier = d["identifier"]
        assert waveform.dtype == np.float32
        assert waveform.shape == (1, 480)
        assert sample_rate == 16000
        assert sample_rate.dtype == np.int32
        assert talk_id == "test1"
        assert speaker_id == "test1"
        assert transcript == "this is record 1 of test1."
        assert identifier == "<o,f0,female>"
        num_iter1 += 1
    for d in data3.create_dict_iterator(num_epochs=1, output_numpy=True):
        waveform = d["waveform"]
        sample_rate = d["sample_rate"]
        transcript = d["transcript"]
        talk_id = d["talk_id"]
        speaker_id = d["speaker_id"]
        identifier = d["identifier"]
        assert waveform.dtype == np.float32
        assert waveform.shape == (1, 160)
        assert sample_rate == 16000
        assert sample_rate.dtype == np.int32
        assert talk_id == "test3"
        assert speaker_id == "test3"
        assert transcript == "this is record 1 of test3."
        assert identifier == "<o,f0,female>"
        num_iter3 += 1
    assert num_iter1 == 1
    assert num_iter3 == 1


def test_tedlium_exceptions():
    """
    Feature: TedliumDataset
    Description: Send error when error occur
    Expectation: Send error
    """
    error_msg_1 = "sampler and shuffle cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_1):
        ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE12, RELEASE1, shuffle=False, sampler=ds.PKSampler(3))

    error_msg_2 = "sampler and sharding cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_2):
        ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE12, RELEASE1, sampler=ds.PKSampler(3), num_shards=2, shard_id=0)

    error_msg_3 = "num_shards is specified and currently requires shard_id as well"
    with pytest.raises(RuntimeError, match=error_msg_3):
        ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE12, RELEASE2, num_shards=10)

    error_msg_4 = "shard_id is specified but num_shards is not"
    with pytest.raises(RuntimeError, match=error_msg_4):
        ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE12, RELEASE2, shard_id=0)

    error_msg_5 = "Input shard_id is not within the required interval"
    with pytest.raises(ValueError, match=error_msg_5):
        ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE3, RELEASE3, num_shards=2, shard_id=-1)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE3, RELEASE3, num_shards=2, shard_id=5)

    error_msg_6 = "num_parallel_workers exceeds"
    with pytest.raises(ValueError, match=error_msg_6):
        ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE3, RELEASE3, shuffle=False, num_parallel_workers=0)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE3, RELEASE3, shuffle=False, num_parallel_workers=256)

    error_msg_7 = "Invalid data, no valid data matching the dataset API TedliumDataset"
    with pytest.raises(RuntimeError, match=error_msg_7):
        ds1 = ds.TedliumDataset(NO_SPH_DIR_TEDLIUM12, RELEASE1, "train")
        for _ in ds1.__iter__():
            pass


def test_tedlium_exception_file_path():
    """
    Feature: TedliumDataset
    Description: Error test
    Expectation: Throw error
    """
    def exception_func(item):
        raise Exception("Error occur!")

    try:
        data = ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE12, RELEASE1)
        data = data.map(operations=exception_func, input_columns=["waveform"], num_parallel_workers=1)
        num_rows = 0
        for _ in data.create_dict_iterator():
            num_rows += 1
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    try:
        data = ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE12, RELEASE1)
        data = data.map(operations=exception_func, input_columns=["sample_rate"], num_parallel_workers=1)
        num_rows = 0
        for _ in data.create_dict_iterator():
            num_rows += 1
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    try:
        data = ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE12, RELEASE2)
        data = data.map(operations=exception_func, input_columns=["transcript"], num_parallel_workers=1)
        num_rows = 0
        for _ in data.create_dict_iterator():
            num_rows += 1
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    try:
        data = ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE12, RELEASE2)
        data = data.map(operations=exception_func, input_columns=["talk_id"], num_parallel_workers=1)
        num_rows = 0
        for _ in data.create_dict_iterator():
            num_rows += 1
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    try:
        data = ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE3, RELEASE3)
        data = data.map(operations=exception_func, input_columns=["speaker_id"], num_parallel_workers=1)
        num_rows = 0
        for _ in data.create_dict_iterator():
            num_rows += 1
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    try:
        data = ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE3, RELEASE3)
        data = data.map(operations=exception_func, input_columns=["identifier"], num_parallel_workers=1)
        num_rows = 0
        for _ in data.create_dict_iterator():
            num_rows += 1
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)


def test_tedlium_extensions():
    """
    Feature: TedliumDataset
    Description: Test extensions of tedlium
    Expectation: Extensions
                    set     invalid data
                    get     throw error
    """
    try:
        data = ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE12, RELEASE1, "train", "invalid")
        for _ in data.create_dict_iterator(output_numpy=True):
            pass
        assert False
    except RuntimeError as e:
        assert "is not supported." in str(e)


def test_tedlium_release():
    """
    Feature: TedliumDataset
    Description: Test release of tedlium
    Expectation: Release
                    set     invalid data
                    get     throw error
    """
    def test_config(release):
        try:
            ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE12, release)
        except (ValueError, TypeError, RuntimeError) as e:
            return str(e)
        return None

    # test the release
    assert "release is not within the valid set of ['release1', 'release2', 'release3']" in test_config("invalid")
    assert "Argument release with value None is not of type [<class 'str'>]" in test_config(None)
    assert "Argument release with value ['list'] is not of type [<class 'str'>]" in test_config(["list"])


def test_tedlium_sequential_sampler():
    """
    Feature: TedliumDataset
    Description: Test tedlium sequential sampler
    Expectation: Correct data
    """
    num_samples = 3
    sampler = ds.SequentialSampler(num_samples=num_samples)
    data21 = ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE12, RELEASE2, sampler=sampler)
    data22 = ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE12, RELEASE2, shuffle=False, num_samples=num_samples)
    num_iter2 = 0
    for item1, item2 in zip(data21.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data22.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_equal(item1["waveform"], item2["waveform"])
        num_iter2 += 1

    assert num_iter2 == num_samples


def test_tedlium_sampler_get_dataset_size():
    """
    Feature: TedliumDataset
    Description: Test TedliumDataset with SequentialSampler and get_dataset_size
    Expectation: num_samples
                    set 5
                    get 5
    """
    sampler = ds.SequentialSampler(start_index=0, num_samples=5)
    data3 = ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE3, RELEASE3, sampler=sampler)
    num_iter3 = 0
    ds_sz3 = data3.get_dataset_size()
    for _ in data3.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter3 += 1

    assert ds_sz3 == num_iter3 == 5


def test_tedlium_usage():
    """
    Feature: TedliumDataset
    Description: Test usage of tedlium
    Expectation: Usage
                    set     valid data      invalid data
                    get     correct data    throw error
    """
    def test_config_tedlium12(usage):

        try:
            data1 = ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE12, RELEASE1, usage=usage)
            data2 = ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE12, RELEASE2, usage=usage)
            num_rows = 0
            for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
                num_rows += 1
            for _ in data2.create_dict_iterator(num_epochs=1, output_numpy=True):
                num_rows += 1
        except (ValueError, TypeError, RuntimeError) as e:
            return str(e)
        return num_rows

    # test the usage of TEDLIUM
    assert test_config_tedlium12("dev") == 1 + 1
    assert test_config_tedlium12("test") == 2 + 2
    assert test_config_tedlium12("train") == 3 + 3
    assert test_config_tedlium12("all") == 1 + 1 + 2 + 2 + 3 + 3
    assert "usage is not within the valid set of ['train', 'test', 'dev', 'all']" in test_config_tedlium12("invalid")
    assert "Argument usage with value ['list'] is not of type [<class 'str'>]" in test_config_tedlium12(["list"])


def test_tedlium_with_chained_sampler_get_dataset_size():
    """
    Feature: TedliumDataset
    Description: Test TedliumDataset with RandomSampler chained with a SequentialSampler and get_dataset_size
    Expectation: num_samples
                    set 2
                    get 2
    """
    sampler = ds.SequentialSampler(start_index=0, num_samples=2)
    child_sampler = ds.RandomSampler()
    sampler.add_child(child_sampler)
    data1 = ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE12, RELEASE1, sampler=sampler)
    num_iter1 = 0
    ds_sz1 = data1.get_dataset_size()
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter1 += 1

    assert ds_sz1 == num_iter1 == 2


def test_tedlium_pipeline():
    """
    Feature: TedliumDataset
    Description: Read a sample
    Expectation: The amount of each function are equal
    """
    # Original waveform
    dataset = ds.TedliumDataset(DATA_DIR_TEDLIUM_RELEASE12, RELEASE1, num_samples=1)
    band_biquad_op = audio.BandBiquad(8000, 200.0)
    # Filtered waveform by bandbiquad
    dataset = dataset.map(input_columns=["waveform"], operations=band_biquad_op, num_parallel_workers=2)
    i = 0
    for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        i += 1
    assert i == 1


if __name__ == '__main__':
    test_tedlium_basic()
    test_tedlium_content_check()
    test_tedlium_exceptions()
    test_tedlium_exception_file_path()
    test_tedlium_extensions()
    test_tedlium_release()
    test_tedlium_sequential_sampler()
    test_tedlium_sampler_get_dataset_size()
    test_tedlium_usage()
    test_tedlium_with_chained_sampler_get_dataset_size()
    test_tedlium_pipeline()
