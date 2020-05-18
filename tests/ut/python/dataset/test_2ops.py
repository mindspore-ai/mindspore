# Copyright 2019 Huawei Technologies Co., Ltd
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
from mindspore import log as logger
from util import save_and_check

DATA_DIR = ["../data/dataset/testTFTestAllTypes/test.data"]
SCHEMA_DIR = "../data/dataset/testTFTestAllTypes/datasetSchema.json"
COLUMNS = ["col_1d", "col_2d", "col_3d", "col_binary", "col_float",
           "col_sint16", "col_sint32", "col_sint64"]
GENERATE_GOLDEN = False


def test_2ops_repeat_shuffle():
    """
    Test Repeat then Shuffle
    """
    logger.info("Test Repeat then Shuffle")
    # define parameters
    repeat_count = 2
    buffer_size = 5
    seed = 0
    parameters = {"params": {'repeat_count': repeat_count,
                             'buffer_size': buffer_size,
                             'seed': seed}}

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    data1 = data1.repeat(repeat_count)
    ds.config.set_seed(seed)
    data1 = data1.shuffle(buffer_size=buffer_size)

    filename = "test_2ops_repeat_shuffle.npz"
    save_and_check(data1, parameters, filename, generate_golden=GENERATE_GOLDEN)


def skip_test_2ops_shuffle_repeat():
    """
    Test Shuffle then Repeat
    """
    logger.info("Test Shuffle then Repeat")
    # define parameters
    repeat_count = 2
    buffer_size = 5
    seed = 0
    parameters = {"params": {'repeat_count': repeat_count,
                             'buffer_size': buffer_size,
                             'reshuffle_each_iteration': False,
                             'seed': seed}}

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    ds.config.set_seed(seed)
    data1 = data1.shuffle(buffer_size=buffer_size)
    data1 = data1.repeat(repeat_count)

    filename = "test_2ops_shuffle_repeat.npz"
    save_and_check(data1, parameters, filename, generate_golden=GENERATE_GOLDEN)


def test_2ops_repeat_batch():
    """
    Test Repeat then Batch
    """
    logger.info("Test Repeat then Batch")
    # define parameters
    repeat_count = 2
    batch_size = 5
    parameters = {"params": {'repeat_count': repeat_count,
                             'batch_size': batch_size}}

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    data1 = data1.repeat(repeat_count)
    data1 = data1.batch(batch_size, drop_remainder=True)

    filename = "test_2ops_repeat_batch.npz"
    save_and_check(data1, parameters, filename, generate_golden=GENERATE_GOLDEN)


def test_2ops_batch_repeat():
    """
    Test Batch then Repeat
    """
    logger.info("Test Batch then Repeat")
    # define parameters
    repeat_count = 2
    batch_size = 5
    parameters = {"params": {'repeat_count': repeat_count,
                             'batch_size': batch_size}}

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    data1 = data1.batch(batch_size, drop_remainder=True)
    data1 = data1.repeat(repeat_count)

    filename = "test_2ops_batch_repeat.npz"
    save_and_check(data1, parameters, filename, generate_golden=GENERATE_GOLDEN)


def test_2ops_batch_shuffle():
    """
    Test Batch then Shuffle
    """
    logger.info("Test Batch then Shuffle")
    # define parameters
    buffer_size = 5
    seed = 0
    batch_size = 2
    parameters = {"params": {'buffer_size': buffer_size,
                             'seed': seed,
                             'batch_size': batch_size}}

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    data1 = data1.batch(batch_size, drop_remainder=True)
    ds.config.set_seed(seed)
    data1 = data1.shuffle(buffer_size=buffer_size)

    filename = "test_2ops_batch_shuffle.npz"
    save_and_check(data1, parameters, filename, generate_golden=GENERATE_GOLDEN)


def test_2ops_shuffle_batch():
    """
    Test Shuffle then Batch
    """
    logger.info("Test Shuffle then Batch")
    # define parameters
    buffer_size = 5
    seed = 0
    batch_size = 2
    parameters = {"params": {'buffer_size': buffer_size,
                             'seed': seed,
                             'batch_size': batch_size}}

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    ds.config.set_seed(seed)
    data1 = data1.shuffle(buffer_size=buffer_size)
    data1 = data1.batch(batch_size, drop_remainder=True)

    filename = "test_2ops_shuffle_batch.npz"
    save_and_check(data1, parameters, filename, generate_golden=GENERATE_GOLDEN)


if __name__ == '__main__':
    test_2ops_repeat_shuffle()
    # test_2ops_shuffle_repeat()
    test_2ops_repeat_batch()
    test_2ops_batch_repeat()
    test_2ops_batch_shuffle()
    test_2ops_shuffle_batch()
