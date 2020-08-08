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
"""
Testing cache operator with mappable datasets
"""
import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.c_transforms as c_vision
from mindspore import log as logger
from util import save_and_check_md5

DATA_DIR = "../data/dataset/testImageNetData/train/"

GENERATE_GOLDEN = False


def test_cache_map_basic1():
    """
    Test mappable leaf with cache op right over the leaf

       Repeat
         |
     Map(decode)
         |
       Cache
         |
     ImageFolder
    """

    logger.info("Test cache map basic 1")

    some_cache = ds.DatasetCache(session_id=1, size=0, spilling=True)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDatasetV2(dataset_dir=DATA_DIR, cache=some_cache)
    decode_op = c_vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op)
    ds1 = ds1.repeat(4)

    filename = "cache_map_01_result.npz"
    save_and_check_md5(ds1, filename, generate_golden=GENERATE_GOLDEN)

    logger.info("test_cache_map_basic1 Ended.\n")


def test_cache_map_basic2():
    """
    Test mappable leaf with the cache op later in the tree above the map(decode)

       Repeat
         |
       Cache
         |
     Map(decode)
         |
     ImageFolder
    """

    logger.info("Test cache map basic 2")

    some_cache = ds.DatasetCache(session_id=1, size=0, spilling=True)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDatasetV2(dataset_dir=DATA_DIR)
    decode_op = c_vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op, cache=some_cache)
    ds1 = ds1.repeat(4)

    filename = "cache_map_02_result.npz"
    save_and_check_md5(ds1, filename, generate_golden=GENERATE_GOLDEN)

    logger.info("test_cache_map_basic2 Ended.\n")


def test_cache_map_basic3():
    """
    Test a repeat under mappable cache

        Cache
          |
      Map(decode)
          |
        Repeat
          |
      ImageFolder
    """

    logger.info("Test cache basic 3")

    some_cache = ds.DatasetCache(session_id=1, size=0, spilling=True)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDatasetV2(dataset_dir=DATA_DIR)
    decode_op = c_vision.Decode()
    ds1 = ds1.repeat(4)
    ds1 = ds1.map(input_columns=["image"], operations=decode_op, cache=some_cache)
    logger.info("ds1.dataset_size is ", ds1.get_dataset_size())

    num_iter = 0
    for _ in ds1.create_dict_iterator():
        logger.info("get data from dataset")
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 8
    logger.info('test_cache_basic3 Ended.\n')


def test_cache_map_basic4():
    """
    Test different rows result in core dump
    """
    logger.info("Test cache basic 4")
    some_cache = ds.DatasetCache(session_id=1, size=0, spilling=True)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDatasetV2(dataset_dir=DATA_DIR, cache=some_cache)
    decode_op = c_vision.Decode()
    ds1 = ds1.repeat(4)
    ds1 = ds1.map(input_columns=["image"], operations=decode_op)
    logger.info("ds1.dataset_size is ", ds1.get_dataset_size())
    shape = ds1.output_shapes()
    logger.info(shape)
    num_iter = 0
    for _ in ds1.create_dict_iterator():
        logger.info("get data from dataset")
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 8
    logger.info('test_cache_basic3 Ended.\n')


def test_cache_map_failure1():
    """
    Test nested cache (failure)

        Repeat
          |
        Cache
          |
      Map(decode)
          |
        Cache
          |
      ImageFolder

    """
    logger.info("Test cache failure 1")

    some_cache = ds.DatasetCache(session_id=1, size=0, spilling=True)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDatasetV2(dataset_dir=DATA_DIR, cache=some_cache)
    decode_op = c_vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op, cache=some_cache)
    ds1 = ds1.repeat(4)

    try:
        num_iter = 0
        for _ in ds1.create_dict_iterator():
            num_iter += 1
    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Nested cache operations is not supported!" in str(e)

    assert num_iter == 0
    logger.info('test_cache_failure1 Ended.\n')


if __name__ == '__main__':
    test_cache_map_basic1()
    logger.info("test_cache_map_basic1 success.")
    test_cache_map_basic2()
    logger.info("test_cache_map_basic2 success.")
    test_cache_map_basic3()
    logger.info("test_cache_map_basic3 success.")
    test_cache_map_basic4()
    logger.info("test_cache_map_basic3 success.")
    test_cache_map_failure1()
    logger.info("test_cache_map_failure1 success.")
