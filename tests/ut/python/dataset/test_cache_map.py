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
"""
Testing cache operation with mappable datasets
"""
import os
import pytest
import numpy as np
import mindspore.dataset as ds
from mindspore.dataset import vision
from mindspore import log as logger
from util import save_and_check_md5

DATA_DIR = "../data/dataset/testImageNetData/train/"
COCO_DATA_DIR = "../data/dataset/testCOCO/train/"
COCO_ANNOTATION_FILE = "../data/dataset/testCOCO/annotations/train.json"
NO_IMAGE_DIR = "../data/dataset/testRandomData/"
MNIST_DATA_DIR = "../data/dataset/testMnistData/"
CELEBA_DATA_DIR = "../data/dataset/testCelebAData/"
VOC_DATA_DIR = "../data/dataset/testVOC2012/"
MANIFEST_DATA_FILE = "../data/dataset/testManifestData/test.manifest"
CIFAR10_DATA_DIR = "../data/dataset/testCifar10Data/"
CIFAR100_DATA_DIR = "../data/dataset/testCifar100Data/"
MIND_RECORD_DATA_DIR = "../data/mindrecord/testTwoImageData/twobytes.mindrecord"
GENERATE_GOLDEN = False


def test_disable_cache_in_independent_mode():
    """
    Feature: Disable cache in independent mode.
    Description: Test cache is disable in independent mode.
    Expectation: Exception raised to notify feature not supported.
    """
    os.environ["MS_INDEPENDENT_DATASET"] = "True"
    with pytest.raises(RuntimeError) as err:
        some_cache = ds.DatasetCache(session_id=1, size=0)
        _ = ds.ImageFolderDataset(dataset_dir=DATA_DIR, cache=some_cache)
    assert "Dataset Cache is not supported in Dataset Independent mode" in str(err.value)
    os.environ["MS_INDEPENDENT_DATASET"] = "False"


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_basic1():
    """
    Feature: DatasetCache op
    Description: Test mappable leaf with Cache op right over the leaf

       Repeat
         |
     Map(Decode)
         |
       Cache
         |
     ImageFolder

    Expectation: Passes the md5 check test
    """
    logger.info("Test cache map basic 1")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDataset(dataset_dir=DATA_DIR, cache=some_cache)
    decode_op = vision.Decode()
    ds1 = ds1.map(operations=decode_op, input_columns=["image"])
    ds1 = ds1.repeat(4)

    filename = "cache_map_01_result.npz"
    save_and_check_md5(ds1, filename, generate_golden=GENERATE_GOLDEN)

    logger.info("test_cache_map_basic1 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_basic2():
    """
    Feature: DatasetCache op
    Description: Test mappable leaf with the Cache op later in the tree above the Map (Decode)

       Repeat
         |
       Cache
         |
     Map(Decode)
         |
     ImageFolder

    Expectation: Passes the md5 check test
    """
    logger.info("Test cache map basic 2")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDataset(dataset_dir=DATA_DIR)
    decode_op = vision.Decode()
    ds1 = ds1.map(operations=decode_op, input_columns=[
        "image"], cache=some_cache)
    ds1 = ds1.repeat(4)

    filename = "cache_map_02_result.npz"
    save_and_check_md5(ds1, filename, generate_golden=GENERATE_GOLDEN)

    logger.info("test_cache_map_basic2 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_basic3():
    """
    Feature: DatasetCache op
    Description: Test mappable leaf with the Cache op later in the tree below the Map (Decode)
    Expectation: Runs successfully
    """
    logger.info("Test cache basic 3")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")
    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDataset(dataset_dir=DATA_DIR, cache=some_cache)
    decode_op = vision.Decode()
    ds1 = ds1.repeat(4)
    ds1 = ds1.map(operations=decode_op, input_columns=["image"])
    logger.info("ds1.dataset_size is ", ds1.get_dataset_size())
    shape = ds1.output_shapes()
    logger.info(shape)
    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        logger.info("get data from dataset")
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 8
    logger.info('test_cache_basic3 Ended.\n')


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_basic4():
    """
    Feature: DatasetCache op
    Description: Test Map containing random op above Cache

                Repeat
                  |
             Map(Decode, RandomCrop)
                  |
                Cache
                  |
             ImageFolder

    Expectation: Runs successfully
    """
    logger.info("Test cache basic 4")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This DATA_DIR only has 2 images in it
    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, cache=some_cache)
    random_crop_op = vision.RandomCrop([512, 512], [200, 200, 200, 200])
    decode_op = vision.Decode()

    data = data.map(input_columns=["image"], operations=decode_op)
    data = data.map(input_columns=["image"], operations=random_crop_op)
    data = data.repeat(4)

    num_iter = 0
    for _ in data.create_dict_iterator(num_epochs=1):
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 8
    logger.info('test_cache_basic4 Ended.\n')


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_basic5():
    """
    Feature: DatasetCache op
    Description: Test Cache as root node

       Cache
         |
      ImageFolder

    Expectation: Runs successfully
    """
    logger.info("Test cache basic 5")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")
    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDataset(dataset_dir=DATA_DIR, cache=some_cache)
    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        logger.info("get data from dataset")
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 2
    logger.info('test_cache_basic5 Ended.\n')


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_failure2():
    """
    Feature: DatasetCache op
    Description: Test Zip under Cache

               Repeat
                  |
                Cache
                  |
             Map(Decode)
                  |
                 Zip
                |    |
      ImageFolder     ImageFolder

    Expectation: Correct error is raised as expected
    """
    logger.info("Test cache failure 2")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDataset(dataset_dir=DATA_DIR)
    ds2 = ds.ImageFolderDataset(dataset_dir=DATA_DIR)
    dsz = ds.zip((ds1, ds2))
    decode_op = vision.Decode()
    dsz = dsz.map(input_columns=["image"],
                  operations=decode_op, cache=some_cache)
    dsz = dsz.repeat(4)

    with pytest.raises(RuntimeError) as e:
        num_iter = 0
        for _ in dsz.create_dict_iterator(num_epochs=1):
            num_iter += 1
    assert "ZipNode is not supported as a descendant operator under a cache" in str(
        e.value)

    assert num_iter == 0
    logger.info('test_cache_failure2 Ended.\n')


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_failure5():
    """
    Feature: DatasetCache op
    Description: Test Map containing Random operation under Cache

               Repeat
                  |
                Cache
                  |
             Map(Decode, RandomCrop)
                  |
              Manifest

    Expectation: Correct error is raised as expected
    """
    logger.info("Test cache failure 5")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This dataset has 4 records
    data = ds.ManifestDataset(MANIFEST_DATA_FILE, decode=True)
    random_crop_op = vision.RandomCrop([512, 512], [200, 200, 200, 200])
    decode_op = vision.Decode()

    data = data.map(input_columns=["image"], operations=decode_op)
    data = data.map(input_columns=["image"],
                    operations=random_crop_op, cache=some_cache)
    data = data.repeat(4)

    with pytest.raises(RuntimeError) as e:
        num_iter = 0
        for _ in data.create_dict_iterator(num_epochs=1):
            num_iter += 1
    assert "MapNode containing random operation is not supported as a descendant of cache" in str(
        e.value)

    assert num_iter == 0
    logger.info('test_cache_failure5 Ended.\n')


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_failure7():
    """
    Feature: DatasetCache op
    Description: Test no-cache-supporting Generator leaf with Map under Cache

               Repeat
                  |
                Cache
                  |
            Map(lambda x: x)
                  |
              Generator

    Expectation: Correct error is raised as expected
    """
    def generator_1d():
        for i in range(64):
            yield (np.array(i),)

    logger.info("Test cache failure 7")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    data = ds.GeneratorDataset(generator_1d, ["data"])
    data = data.map(vision.not_random(lambda x: x), ["data"], cache=some_cache)
    data = data.repeat(4)

    with pytest.raises(RuntimeError) as e:
        num_iter = 0
        for _ in data.create_dict_iterator(num_epochs=1):
            num_iter += 1
    assert "There is currently no support for GeneratorOp under cache" in str(
        e.value)

    assert num_iter == 0
    logger.info('test_cache_failure7 Ended.\n')


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_failure11():
    """
    Feature: DatasetCache op
    Description: Test set spilling=true when Cache server is started without spilling support

         Cache(spilling=true)
                 |
             ImageFolder

    Expectation: Correct error is raised as expected
    """
    logger.info("Test cache failure 11")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0, spilling=True)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDataset(dataset_dir=DATA_DIR, cache=some_cache)

    with pytest.raises(RuntimeError) as e:
        num_iter = 0
        for _ in ds1.create_dict_iterator(num_epochs=1):
            num_iter += 1
    assert "Server is not set up with spill support" in str(
        e.value)

    assert num_iter == 0
    logger.info('test_cache_failure11 Ended.\n')


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_split1():
    """
    Feature: DatasetCache op
    Description: Test split (after a non-source node, which is implemented with TakeOp/SkipOp) under Cache

               Repeat
                  |
                Cache
                  |
             Map(Resize)
                  |
                Split
                  |
             Map(Decode)
                  |
             ImageFolder

    Expectation: Correct error is raised as expected
    """
    logger.info("Test cache split 1")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDataset(dataset_dir=DATA_DIR)

    decode_op = vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op)
    ds1, ds2 = ds1.split([0.5, 0.5])
    resize_op = vision.Resize((224, 224))
    ds1 = ds1.map(input_columns=["image"],
                  operations=resize_op, cache=some_cache)
    ds2 = ds2.map(input_columns=["image"],
                  operations=resize_op, cache=some_cache)
    ds1 = ds1.repeat(4)
    ds2 = ds2.repeat(4)

    with pytest.raises(RuntimeError) as e:
        num_iter = 0
        for _ in ds1.create_dict_iterator(num_epochs=1):
            num_iter += 1
    assert "TakeNode (possibly from Split) is not supported as a descendant operator under a cache" in str(
        e.value)

    with pytest.raises(RuntimeError) as e:
        num_iter = 0
        for _ in ds2.create_dict_iterator(num_epochs=1):
            num_iter += 1
    assert "TakeNode (possibly from Split) is not supported as a descendant operator under a cache" in str(
        e.value)
    logger.info('test_cache_split1 Ended.\n')


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_parameter_check():
    """
    Feature: DatasetCache op
    Description: Test illegal parameters for DatasetCache op
    Expectation: Correct error is raised as expected
    """
    logger.info("Test cache map parameter check")

    with pytest.raises(ValueError) as info:
        ds.DatasetCache(session_id=-1, size=0)
    assert "Input is not within the required interval" in str(info.value)

    with pytest.raises(TypeError) as info:
        ds.DatasetCache(session_id="1", size=0)
    assert "Argument session_id with value 1 is not of type" in str(info.value)

    with pytest.raises(TypeError) as info:
        ds.DatasetCache(session_id=None, size=0)
    assert "Argument session_id with value None is not of type" in str(
        info.value)

    with pytest.raises(ValueError) as info:
        ds.DatasetCache(session_id=1, size=-1)
    assert "Input size must be greater than 0" in str(info.value)

    with pytest.raises(TypeError) as info:
        ds.DatasetCache(session_id=1, size="1")
    assert "Argument size with value 1 is not of type" in str(info.value)

    with pytest.raises(TypeError) as info:
        ds.DatasetCache(session_id=1, size=None)
    assert "Argument size with value None is not of type" in str(info.value)

    with pytest.raises(TypeError) as info:
        ds.DatasetCache(session_id=1, size=0, spilling="illegal")
    assert "Argument spilling with value illegal is not of type" in str(
        info.value)

    with pytest.raises(TypeError) as err:
        ds.DatasetCache(session_id=1, size=0, hostname=50052)
    assert "Argument hostname with value 50052 is not of type" in str(
        err.value)

    with pytest.raises(RuntimeError) as err:
        ds.DatasetCache(session_id=1, size=0, hostname="illegal")
    assert "now cache client has to be on the same host with cache server" in str(
        err.value)

    with pytest.raises(RuntimeError) as err:
        ds.DatasetCache(session_id=1, size=0, hostname="127.0.0.2")
    assert "now cache client has to be on the same host with cache server" in str(
        err.value)

    with pytest.raises(TypeError) as info:
        ds.DatasetCache(session_id=1, size=0, port="illegal")
    assert "Argument port with value illegal is not of type" in str(info.value)

    with pytest.raises(TypeError) as info:
        ds.DatasetCache(session_id=1, size=0, port="50052")
    assert "Argument port with value 50052 is not of type" in str(info.value)

    with pytest.raises(ValueError) as err:
        ds.DatasetCache(session_id=1, size=0, port=0)
    assert "Input port is not within the required interval of [1025, 65535]" in str(
        err.value)

    with pytest.raises(ValueError) as err:
        ds.DatasetCache(session_id=1, size=0, port=65536)
    assert "Input port is not within the required interval of [1025, 65535]" in str(
        err.value)

    with pytest.raises(TypeError) as err:
        ds.ImageFolderDataset(dataset_dir=DATA_DIR, cache=True)
    assert "Argument cache with value True is not of type" in str(err.value)

    logger.info("test_cache_map_parameter_check Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_running_twice1():
    """
    Feature: DatasetCache op
    Description: Test executing the same pipeline for twice (from Python), with Cache injected after Map

       Repeat
         |
       Cache
         |
     Map(decode)
         |
     ImageFolder

    Expectation: Output is the same as expected output
    """
    logger.info("Test cache map running twice 1")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDataset(dataset_dir=DATA_DIR)
    decode_op = vision.Decode()
    ds1 = ds1.map(input_columns=["image"],
                  operations=decode_op, cache=some_cache)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        num_iter += 1
    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 8

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        num_iter += 1
    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 8

    logger.info("test_cache_map_running_twice1 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_running_twice2():
    """
    Feature: DatasetCache op
    Description: Test executing the same pipeline for twice (from shell), with Cache injected after leaf

       Repeat
         |
     Map(Decode)
         |
       Cache
         |
     ImageFolder

    Expectation: Output is the same as expected output
    """
    logger.info("Test cache map running twice 2")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDataset(dataset_dir=DATA_DIR, cache=some_cache)
    decode_op = vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 8
    logger.info("test_cache_map_running_twice2 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_extra_small_size1():
    """
    Feature: DatasetCache op
    Description: Test running pipeline with Cache of extra small size and spilling=True

       Repeat
         |
     Map(Decode)
         |
       Cache
         |
     ImageFolder

    Expectation: Output is the same as expected output
    """
    logger.info("Test cache map extra small size 1")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=1, spilling=True)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDataset(dataset_dir=DATA_DIR, cache=some_cache)
    decode_op = vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 8
    logger.info("test_cache_map_extra_small_size1 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_extra_small_size2():
    """
    Feature: DatasetCache op
    Description: Test running pipeline with Cache of extra small size and spilling=False

       Repeat
         |
       Cache
         |
     Map(Decode)
         |
     ImageFolder

    Expectation: Output is the same as expected output
    """
    logger.info("Test cache map extra small size 2")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=1, spilling=False)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDataset(dataset_dir=DATA_DIR)
    decode_op = vision.Decode()
    ds1 = ds1.map(input_columns=["image"],
                  operations=decode_op, cache=some_cache)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 8
    logger.info("test_cache_map_extra_small_size2 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_no_image():
    """
    Feature: DatasetCache op
    Description: Test Cache with no dataset existing in the path

       Repeat
         |
     Map(Decode)
         |
       Cache
         |
     ImageFolder

    Expectation: Error is raised as expected
    """
    logger.info("Test cache map no image")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=1, spilling=False)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDataset(dataset_dir=NO_IMAGE_DIR, cache=some_cache)
    decode_op = vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op)
    ds1 = ds1.repeat(4)

    with pytest.raises(RuntimeError):
        num_iter = 0
        for _ in ds1.create_dict_iterator(num_epochs=1):
            num_iter += 1

    assert num_iter == 0
    logger.info("test_cache_map_no_image Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_parallel_pipeline1(shard):
    """
    Feature: DatasetCache op
    Description: Test running two parallel pipelines (sharing Cache) with Cache injected after leaf op

       Repeat
         |
     Map(Decode)
         |
       Cache
         |
     ImageFolder

    Expectation: Output is the same as expected output
    """
    logger.info("Test cache map parallel pipeline 1")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDataset(
        dataset_dir=DATA_DIR, num_shards=2, shard_id=int(shard), cache=some_cache)
    decode_op = vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 4
    logger.info("test_cache_map_parallel_pipeline1 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_parallel_pipeline2(shard):
    """
    Feature: DatasetCache op
    Description: Test running two parallel pipelines (sharing Cache) with Cache injected after Map op

       Repeat
         |
       Cache
         |
     Map(Decode)
         |
     ImageFolder

    Expectation: Output is the same as expected output
    """
    logger.info("Test cache map parallel pipeline 2")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDataset(
        dataset_dir=DATA_DIR, num_shards=2, shard_id=int(shard))
    decode_op = vision.Decode()
    ds1 = ds1.map(input_columns=["image"],
                  operations=decode_op, cache=some_cache)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 4
    logger.info("test_cache_map_parallel_pipeline2 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_parallel_workers():
    """
    Feature: DatasetCache op
    Description: Test Cache with num_parallel_workers > 1 set for Map op and leaf op

       Repeat
         |
       Cache
         |
     Map(Decode)
         |
      ImageFolder

    Expectation: Output is the same as expected output
    """
    logger.info("Test cache map parallel workers")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDataset(dataset_dir=DATA_DIR, num_parallel_workers=4)
    decode_op = vision.Decode()
    ds1 = ds1.map(input_columns=[
        "image"], operations=decode_op, num_parallel_workers=4, cache=some_cache)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 8
    logger.info("test_cache_map_parallel_workers Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_server_workers_1():
    """
    Feature: DatasetCache op
    Description: Start Cache server with --workers 1 and then test Cache function

       Repeat
         |
       Cache
         |
     Map(Decode)
         |
      ImageFolder

    Expectation: Output is the same as expected output
    """
    logger.info("Test cache map server workers 1")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDataset(dataset_dir=DATA_DIR)
    decode_op = vision.Decode()
    ds1 = ds1.map(input_columns=["image"],
                  operations=decode_op, cache=some_cache)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 8
    logger.info("test_cache_map_server_workers_1 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_server_workers_100():
    """
    Feature: DatasetCache op
    Description: Start Cache server with --workers 100 and then test Cache function

       Repeat
         |
     Map(decode)
         |
       Cache
         |
      ImageFolder

    Expectation: Output is the same as expected output
    """
    logger.info("Test cache map server workers 100")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDataset(dataset_dir=DATA_DIR, cache=some_cache)
    decode_op = vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 8
    logger.info("test_cache_map_server_workers_100 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_num_connections_1():
    """
    Feature: DatasetCache op
    Description: Test setting num_connections=1 in DatasetCache

       Repeat
         |
       Cache
         |
     Map(Decode)
         |
      ImageFolder

    Expectation: Output is the same as expected output
    """
    logger.info("Test cache map num_connections 1")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(
        session_id=session_id, size=0, num_connections=1)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDataset(dataset_dir=DATA_DIR)
    decode_op = vision.Decode()
    ds1 = ds1.map(input_columns=["image"],
                  operations=decode_op, cache=some_cache)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 8
    logger.info("test_cache_map_num_connections_1 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_num_connections_100():
    """
    Feature: DatasetCache op
    Description: Test setting num_connections=100 in DatasetCache

       Repeat
         |
     Map(Decode)
         |
       Cache
         |
      ImageFolder

    Expectation: Output is the same as expected output
    """
    logger.info("Test cache map num_connections 100")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(
        session_id=session_id, size=0, num_connections=100)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDataset(dataset_dir=DATA_DIR, cache=some_cache)
    decode_op = vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 8
    logger.info("test_cache_map_num_connections_100 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_prefetch_size_1():
    """
    Feature: DatasetCache op
    Description: Test setting prefetch_size=1 in DatasetCache

       Repeat
         |
       Cache
         |
     Map(Decode)
         |
      ImageFolder

    Expectation: Output is the same as expected output
    """
    logger.info("Test cache map prefetch_size 1")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(
        session_id=session_id, size=0, prefetch_size=1)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDataset(dataset_dir=DATA_DIR)
    decode_op = vision.Decode()
    ds1 = ds1.map(input_columns=["image"],
                  operations=decode_op, cache=some_cache)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 8
    logger.info("test_cache_map_prefetch_size_1 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_prefetch_size_100():
    """
    Feature: DatasetCache op
    Description: Test setting prefetch_size=100 in DatasetCache

       Repeat
         |
     Map(Decode)
         |
       Cache
         |
      ImageFolder

    Expectation: Output is the same as expected output
    """
    logger.info("Test cache map prefetch_size 100")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(
        session_id=session_id, size=0, prefetch_size=100)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDataset(dataset_dir=DATA_DIR, cache=some_cache)
    decode_op = vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 8
    logger.info("test_cache_map_prefetch_size_100 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_device_que():
    """
    Feature: DatasetCache op
    Description: Test Cache with device_que

     DeviceQueue
         |
      EpochCtrl
         |
       Repeat
         |
     Map(Decode)
         |
       Cache
         |
      ImageFolder

    Expectation: Output is the same as expected output
    """
    logger.info("Test cache map device_que")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDataset(dataset_dir=DATA_DIR)
    decode_op = vision.Decode()
    ds1 = ds1.map(input_columns=["image"],
                  operations=decode_op, cache=some_cache)
    ds1 = ds1.repeat(4)
    ds1 = ds1.device_que()
    ds1.send()

    logger.info("test_cache_map_device_que Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_epoch_ctrl1():
    """
    Feature: DatasetCache op
    Description: Test using two-loops method to run several epochs

     Map(Decode)
         |
       Cache
         |
      ImageFolder

    Expectation: Output is the same as expected output
    """
    logger.info("Test cache map epoch ctrl1")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDataset(dataset_dir=DATA_DIR, cache=some_cache)
    decode_op = vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op)

    num_epoch = 5
    iter1 = ds1.create_dict_iterator(num_epochs=num_epoch)

    epoch_count = 0
    for _ in range(num_epoch):
        row_count = 0
        for _ in iter1:
            row_count += 1
        logger.info("Number of data in ds1: {} ".format(row_count))
        assert row_count == 2
        epoch_count += 1
    assert epoch_count == num_epoch
    logger.info("test_cache_map_epoch_ctrl1 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_epoch_ctrl2():
    """
    Feature: DatasetCache op
    Description: Test using two-loops method with infinite epochs

        Cache
         |
     Map(Decode)
         |
      ImageFolder

    Expectation: Output is the same as expected output
    """
    logger.info("Test cache map epoch ctrl2")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDataset(dataset_dir=DATA_DIR)
    decode_op = vision.Decode()
    ds1 = ds1.map(input_columns=["image"],
                  operations=decode_op, cache=some_cache)

    num_epoch = 5
    # iter1 will always assume there is a next epoch and never shutdown
    iter1 = ds1.create_dict_iterator(num_epochs=-1)

    epoch_count = 0
    for _ in range(num_epoch):
        row_count = 0
        for _ in iter1:
            row_count += 1
        logger.info("Number of data in ds1: {} ".format(row_count))
        assert row_count == 2
        epoch_count += 1
    assert epoch_count == num_epoch

    # manually stop the iterator
    iter1.stop()
    logger.info("test_cache_map_epoch_ctrl2 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_epoch_ctrl3():
    """
    Feature: DatasetCache op
    Description: Test using two-loops method with infinite epochs over repeat

       Repeat
         |
     Map(Decode)
         |
       Cache
         |
      ImageFolder

    Expectation: Output is the same as expected output
    """
    logger.info("Test cache map epoch ctrl3")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDataset(dataset_dir=DATA_DIR, cache=some_cache)
    decode_op = vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op)
    ds1 = ds1.repeat(2)

    num_epoch = 5
    # iter1 will always assume there is a next epoch and never shutdown
    iter1 = ds1.create_dict_iterator(num_epochs=num_epoch)

    epoch_count = 0
    for _ in range(num_epoch):
        row_count = 0
        for _ in iter1:
            row_count += 1
        logger.info("Number of data in ds1: {} ".format(row_count))
        assert row_count == 4
        epoch_count += 1
    assert epoch_count == num_epoch

    # reply on garbage collector to destroy iter1

    logger.info("test_cache_map_epoch_ctrl3 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_manifest1():
    """
    Feature: DatasetCache op
    Description: Test mappable Manifest leaf with Cache op right over the leaf

       Cache
         |
      Manifest

    Expectation: Output is the same as expected output
    """
    logger.info("Test cache map manifest1")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This dataset has 4 records
    ds1 = ds.ManifestDataset(MANIFEST_DATA_FILE, decode=True, cache=some_cache)

    num_epoch = 4
    iter1 = ds1.create_dict_iterator(num_epochs=num_epoch)

    epoch_count = 0
    for _ in range(num_epoch):
        assert sum(1 for _ in iter1) == 4
        epoch_count += 1
    assert epoch_count == num_epoch

    logger.info("test_cache_map_manifest1 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_manifest2():
    """
    Feature: DatasetCache op
    Description: Test mappable Manifest leaf with the Cache op later in the tree above the Map(Resize)

       Cache
         |
     Map(Resize)
         |
      Manifest

    Expectation: Output is the same as expected output
    """
    logger.info("Test cache map manifest2")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This dataset has 4 records
    ds1 = ds.ManifestDataset(MANIFEST_DATA_FILE, decode=True)
    resize_op = vision.Resize((224, 224))
    ds1 = ds1.map(input_columns=["image"],
                  operations=resize_op, cache=some_cache)

    num_epoch = 4
    iter1 = ds1.create_dict_iterator(num_epochs=num_epoch)

    epoch_count = 0
    for _ in range(num_epoch):
        assert sum(1 for _ in iter1) == 4
        epoch_count += 1
    assert epoch_count == num_epoch

    logger.info("test_cache_map_manifest2 Ended.\n")


class ReverseSampler(ds.Sampler):
    def __iter__(self):
        yield from range(self.dataset_size - 1, -1, -1)


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_mindrecord1():
    """
    Feature: DatasetCache op
    Description: Test mappable MindRecord leaf with Cache op right over the leaf

       Cache
         |
    MindRecord

    Expectation: Output is the same as expected output
    """
    logger.info("Test cache map mindrecord1")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This dataset has 5 records
    columns_list = ["id", "file_name", "label_name", "img_data", "label_data"]
    ds1 = ds.MindDataset(MIND_RECORD_DATA_DIR, columns_list, cache=some_cache)

    num_epoch = 4
    iter1 = ds1.create_dict_iterator(num_epochs=num_epoch, output_numpy=True)

    epoch_count = 0
    for _ in range(num_epoch):
        assert sum(1 for _ in iter1) == 5
        epoch_count += 1
    assert epoch_count == num_epoch

    logger.info("test_cache_map_mindrecord1 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_mindrecord2():
    """
    Feature: DatasetCache op
    Description: Test mappable MindRecord leaf with the Cache op later in the tree above the Map(Decode)

       Cache
         |
     Map(Decode)
         |
     MindRecord

    Expectation: Output is the same as expected output
    """
    logger.info("Test cache map mindrecord2")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This dataset has 5 records
    columns_list = ["id", "file_name", "label_name", "img_data", "label_data"]
    ds1 = ds.MindDataset(MIND_RECORD_DATA_DIR, columns_list)

    decode_op = vision.Decode()
    ds1 = ds1.map(input_columns=["img_data"],
                  operations=decode_op, cache=some_cache)

    num_epoch = 4
    iter1 = ds1.create_dict_iterator(num_epochs=num_epoch, output_numpy=True)

    epoch_count = 0
    for _ in range(num_epoch):
        assert sum(1 for _ in iter1) == 5
        epoch_count += 1
    assert epoch_count == num_epoch

    logger.info("test_cache_map_mindrecord2 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_mindrecord3():
    """
    Feature: DatasetCache op
    Description: Test Cache sharing between the following two pipelines with MindRecord leaf:

       Cache                                    Cache
         |                                      |
     Map(Decode)                            Map(Decode)
         |                                      |
      MindRecord(num_parallel_workers=5)    MindRecord(num_parallel_workers=6)

    Expectation: Output is the same as expected output
    """
    logger.info("Test cache map mindrecord3")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This dataset has 5 records
    columns_list = ["id", "file_name", "label_name", "img_data", "label_data"]
    decode_op = vision.Decode()

    ds1 = ds.MindDataset(MIND_RECORD_DATA_DIR, columns_list=columns_list,
                         num_parallel_workers=5, shuffle=True)
    ds1 = ds1.map(input_columns=["img_data"],
                  operations=decode_op, cache=some_cache)

    ds2 = ds.MindDataset(MIND_RECORD_DATA_DIR, columns_list=columns_list,
                         num_parallel_workers=6, shuffle=True)
    ds2 = ds2.map(input_columns=["img_data"],
                  operations=decode_op, cache=some_cache)

    iter1 = ds1.create_dict_iterator(num_epochs=1, output_numpy=True)
    iter2 = ds2.create_dict_iterator(num_epochs=1, output_numpy=True)

    assert sum(1 for _ in iter1) == 5
    assert sum(1 for _ in iter2) == 5

    logger.info("test_cache_map_mindrecord3 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_python_sampler1():
    """
    Feature: DatasetCache op
    Description: Test using a Python sampler, and Cache after leaf

        Repeat
         |
     Map(Decode)
         |
       Cache
         |
      ImageFolder

    Expectation: Output is the same as expected output
    """
    logger.info("Test cache map python sampler1")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDataset(
        dataset_dir=DATA_DIR, sampler=ReverseSampler(), cache=some_cache)
    decode_op = vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        num_iter += 1
    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 8
    logger.info("test_cache_map_python_sampler1 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_python_sampler2():
    """
    Feature: DatasetCache op
    Description: Test using a Python sampler, and Cache after Map

       Repeat
         |
       Cache
         |
     Map(Decode)
         |
      ImageFolder

    Expectation: Output is the same as expected output
    """
    logger.info("Test cache map python sampler2")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDataset(dataset_dir=DATA_DIR, sampler=ReverseSampler())
    decode_op = vision.Decode()
    ds1 = ds1.map(input_columns=["image"],
                  operations=decode_op, cache=some_cache)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        num_iter += 1
    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 8
    logger.info("test_cache_map_python_sampler2 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_map_nested_repeat():
    """
    Feature: DatasetCache op
    Description: Test Cache on pipeline with nested Repeat ops

        Repeat
          |
      Map(Decode)
          |
        Repeat
          |
        Cache
          |
      ImageFolder

    Expectation: Output is the same as expected output
    """
    logger.info("Test cache map nested repeat")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This DATA_DIR only has 2 images in it
    ds1 = ds.ImageFolderDataset(dataset_dir=DATA_DIR, cache=some_cache)
    decode_op = vision.Decode()
    ds1 = ds1.repeat(4)
    ds1 = ds1.map(operations=decode_op, input_columns=["image"])
    ds1 = ds1.repeat(2)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        logger.info("get data from dataset")
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 16
    logger.info('test_cache_map_nested_repeat Ended.\n')


if __name__ == '__main__':
    # This is just a list of tests, don't try to run these tests with 'python test_cache_map.py'
    # since cache server is required to be brought up first
    test_cache_map_basic1()
    test_cache_map_basic2()
    test_cache_map_basic3()
    test_cache_map_basic4()
    test_cache_map_basic5()
    test_cache_map_failure2()
    test_cache_map_failure5()
    test_cache_map_failure7()
    test_cache_map_failure11()
    test_cache_map_split1()
    test_cache_map_parameter_check()
    test_cache_map_running_twice1()
    test_cache_map_running_twice2()
    test_cache_map_extra_small_size1()
    test_cache_map_extra_small_size2()
    test_cache_map_no_image()
    test_cache_map_parallel_pipeline1(shard=0)
    test_cache_map_parallel_pipeline2(shard=1)
    test_cache_map_parallel_workers()
    test_cache_map_server_workers_1()
    test_cache_map_server_workers_100()
    test_cache_map_num_connections_1()
    test_cache_map_num_connections_100()
    test_cache_map_prefetch_size_1()
    test_cache_map_prefetch_size_100()
    test_cache_map_device_que()
    test_cache_map_epoch_ctrl1()
    test_cache_map_epoch_ctrl2()
    test_cache_map_epoch_ctrl3()
    test_cache_map_manifest1()
    test_cache_map_manifest2()
    test_cache_map_mindrecord1()
    test_cache_map_mindrecord2()
    test_cache_map_python_sampler1()
    test_cache_map_python_sampler2()
    test_cache_map_nested_repeat()
