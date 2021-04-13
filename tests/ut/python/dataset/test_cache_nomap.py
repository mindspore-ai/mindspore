# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
Testing cache operator with non-mappable datasets
"""
import os
import itertools
import numpy as np
import pytest
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.text as text
import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore import log as logger

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

TEXT_TF_DATA_DIR = ["../data/dataset/testTextTFRecord/text.tfrecord"]
SCHEMA_DIR2 = "../data/dataset/testTextTFRecord/datasetSchema.json"

TRAIN_DATA_DIR = ["../data/dataset/test_tf_file_3_images2/train-0000-of-0001.data",
                  "../data/dataset/test_tf_file_3_images2/train-0000-of-0002.data",
                  "../data/dataset/test_tf_file_3_images2/train-0000-of-0003.data",
                  "../data/dataset/test_tf_file_3_images2/train-0000-of-0004.data"]
TRAIN_SCHEMA_DIR = "../data/dataset/test_tf_file_3_images2/datasetSchema.json"

IMAGE_FOLDER_DATA_DIR = "../data/dataset/testImageNetData/train/"
CLUE_DATA_DIR = '../data/dataset/testCLUE/afqmc/train.json'
CSV_DATA_DIR = '../data/dataset/testCSV/1.csv'
TEXT_FILE_DATA_DIR = "../data/dataset/testTextFileDataset/1.txt"

PYFUNC_DATA_DIR = ["../data/dataset/testPyfuncMap/data.data"]
PYFUNC_SCHEMA_DIR = "../data/dataset/testPyfuncMap/schema.json"

GENERATE_GOLDEN = False


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_basic1():
    """
    A random dataset (a non mappable dataset) with a cache over it just after the leaf
    """

    logger.info("Test cache nomap basic 1")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    schema = ds.Schema()
    schema.add_column('image', de_type=mstype.uint8,
                      shape=[640, 480, 3])  # 921600 bytes (a bit less than 1 MB per image)
    schema.add_column('label', de_type=mstype.uint8, shape=[1])

    # create a cache.  arbitrary session_id for now
    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # User-created sampler here
    ds1 = ds.RandomDataset(schema=schema, total_rows=10, num_parallel_workers=4, cache=some_cache)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for data in ds1.create_dict_iterator(num_epochs=1):
        logger.info("printing the label: {}".format(data["label"]))
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 40
    logger.info("test_cache_nomap_basic1 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_basic2():
    """
    A random dataset (a non mappable dataset) with a cache over it just after the leaf
    """

    logger.info("Test cache nomap basic 2")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    schema = ds.Schema()
    schema.add_column('image', de_type=mstype.uint8,
                      shape=[640, 480, 3])  # 921600 bytes (a bit less than 1 MB per image)
    schema.add_column('label', de_type=mstype.uint8, shape=[1])

    # create a cache.  arbitrary session_id for now
    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # sampler arg not given directly, however any of these args will auto-generate an appropriate sampler:
    # num_samples, shuffle, num_shards, shard_id
    # In this case, the presence of num_samples chooses a sampler.
    ds1 = ds.RandomDataset(schema=schema, total_rows=20, num_samples=20, num_parallel_workers=4, cache=some_cache)
    ds1 = ds1.repeat(2)

    num_iter = 0
    for data in ds1.create_dict_iterator(num_epochs=1):
        logger.info("printing the label: {}".format(data["label"]))
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 40
    logger.info("test_cache_nomap_basic2 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_basic3():
    """
    A TF reader dataset (a non mappable dataset) with a cache over it just after the leaf

       Repeat
         |
     Map(decode)
         |
       Cache
         |
      TFReader
    """

    logger.info("Test cache nomap basic 3")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False, cache=some_cache)
    decode_op = c_vision.Decode()
    ds1 = ds1.map(operations=decode_op, input_columns=["image"])
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 12

    # Contact the server to get the statistics
    stat = some_cache.GetStat()
    cache_sz = stat.avg_cache_sz
    num_mem_cached = stat.num_mem_cached
    num_disk_cached = stat.num_disk_cached

    logger.info("Number of rows cached in memory: {}".format(num_mem_cached))
    logger.info("Number of rows spilled to disk: {}".format(num_disk_cached))
    logger.info("Average row cache size: {}".format(cache_sz))

    logger.info("test_cache_nomap_basic3 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_basic4():
    """
    A TF reader dataset (a non mappable dataset) with a map decode and cache after it
    Since a global shuffle is used for the tf reader, it will inject a shuffle op over the tf.
    But, if there's a cache later, that shuffle becomes invalid and should be removed.

       Repeat
         |
       Cache
         |
     Map(decode)
         |
      TFReader
    """

    logger.info("Test cache nomap basic 4")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    # This dataset has 3 records in it only
    some_cache = ds.DatasetCache(session_id=session_id, size=0)
    # With shuffle not being set, TF defaults to a "global" shuffle when there is no cache
    # in the picture.  This causes a shuffle-injection over the TF.  For clarify, this test will
    # explicitly give the global option, even though it's the default in python.
    # But, when caching is added in the ascendent tree above TF, we do global shuffling
    # through the sampler over the cache, not by the shuffle op.  In that case, tree prepare
    # will remove the shuffle op that got injected by the initial tree creation.
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=ds.Shuffle.GLOBAL)
    decode_op = c_vision.Decode()

    ds1 = ds1.map(operations=decode_op, input_columns=["image"], cache=some_cache)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 12
    logger.info("test_cache_nomap_basic4 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_basic5():
    """
    A TF reader dataset (a non mappable dataset) with a cache over it just after the leaf
    Same as test 3, but this one does not have shuffle arg, causing tf to default to global
    shuffle which attempts to inject a shuffle operator.  However, since there is a cache
    we do not need global shuffle, so the shuffle will not be built.  It ends up being
    identical to test basic 3, however we arrive at the same tree in different codepaths
    (if there was no cache, then the shuffle IS built)

       Repeat
         |
     Map(decode)
         |
       Cache
         |
      TFReader
    """

    logger.info("Test cache nomap basic 5")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    # This dataset has 3 records in it only
    some_cache = ds.DatasetCache(session_id=session_id, size=0)
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], cache=some_cache)
    decode_op = c_vision.Decode()
    ds1 = ds1.map(operations=decode_op, input_columns=["image"])
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 12
    logger.info("test_cache_nomap_basic5 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_basic6():
    """
    A TF reader dataset (a non mappable dataset) with a cache over it just after the leaf
    In this one, the tf dataset will be given sharding configuration, however since a cache is
    used, the tree prepare should undo the sharding configuration and instead, a distributed
    sampler will be chosen with the same shard config.

       Repeat
         |
     Map(decode)
         |
       Cache
         |
      TFReader
    """

    logger.info("Test cache nomap basic 6")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    # This dataset has 3 records in it only
    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # With only 3 records shard into 3, we expect only 1 record returned for this shard
    # However, the sharding will be done by the sampler, not by the tf record leaf node
    # In this case, it is a row-based sharding, not the file-based sharding that would happen if
    # there was not any cache.
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], num_shards=3, shard_id=1, cache=some_cache)
    decode_op = c_vision.Decode()
    ds1 = ds1.map(operations=decode_op, input_columns=["image"])
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 4
    logger.info("test_cache_nomap_basic6 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_basic7():
    """
    A TF reader dataset (a non mappable dataset) that uses global shuffle, and is cached followed by
    map.
    In this one, the tf dataset with global shuffle might want to inject a shuffle op over top of the
    tf reader, but since a cache is given, it will choose not to.

       Repeat
         |
     Map(decode)
         |
       cache
         |
      TFReader
    """

    logger.info("Test cache nomap basic 7")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This dataset has 3 records in it only
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=ds.Shuffle.GLOBAL, cache=some_cache)
    decode_op = c_vision.Decode()
    ds1 = ds1.map(operations=decode_op, input_columns=["image"])
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 12
    logger.info("test_cache_nomap_basic7 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_basic8():
    """
    Test cache as root node

       cache
         |
      TFReader
    """
    logger.info("Test cache basic 8")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")
    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This dataset has 3 records in it only
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, cache=some_cache)
    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        logger.info("get data from dataset")
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 3
    logger.info('test_cache_basic8 Ended.\n')


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_basic9():
    """
    Testing the GetStat interface for getting some info from server, but this should fail if the cache is not created
    in a pipeline.
    """

    logger.info("Test cache nomap basic 9")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # Contact the server to get the statistics, this should fail because we have not used this cache in any pipeline
    # so there will not be any cache to get stats on.
    with pytest.raises(RuntimeError) as e:
        stat = some_cache.GetStat()
        cache_sz = stat.avg_cache_sz
        logger.info("Average row cache size: {}".format(cache_sz))
    assert "Unexpected error" in str(e.value)

    logger.info("test_cache_nomap_basic9 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_allowed_share1():
    """
    It is allowed to share the cache between the following two trees:

       Repeat     Shuffle
         |           |
       Cache       Cache
         |           |
      TFReader    TFReader
    """

    logger.info("Test cache nomap allowed share 1")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    ds.config.set_seed(1)
    # This dataset has 3 records in it only
    some_cache = ds.DatasetCache(session_id=session_id, size=0, prefetch_size=32)
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False, cache=some_cache)
    ds1 = ds1.repeat(4)

    ds2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False, cache=some_cache)
    ds2 = ds2.shuffle(buffer_size=2)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 12
    logger.info("Number of data in ds1: {} ".format(num_iter))

    num_iter = 0
    for _ in ds2.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 3
    logger.info("test_cache_nomap_allowed_share1 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_allowed_share2():
    """
    It is allowed to share the cache between the following two trees (with map decode):

       Repeat     Shuffle
         |           |
       Cache       Cache
         |           |
     Map(decode) Map(decode)
         |           |
      TFReader    TFReader
    """

    logger.info("Test cache nomap allowed share 2")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    ds.config.set_seed(1)
    # This dataset has 3 records in it only
    some_cache = ds.DatasetCache(session_id=session_id, size=0)
    decode_op = c_vision.Decode()

    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    ds1 = ds1.map(operations=decode_op, input_columns=["image"], cache=some_cache)
    ds1 = ds1.repeat(4)

    ds2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    ds2 = ds2.map(operations=decode_op, input_columns=["image"], cache=some_cache)
    ds2 = ds2.shuffle(buffer_size=2)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        num_iter += 1
    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 12

    num_iter = 0
    for _ in ds2.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 3
    logger.info("test_cache_nomap_allowed_share2 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_allowed_share3():
    """
    It is allowed to share the cache between the following two trees (different shard ids):

       Repeat                     Repeat
         |                          |
       Cache                      Cache
         |                          |
      TFReader(shard_id = 0)     TFReader(shard_id = 1)
    """

    logger.info("Test cache nomap allowed share 3")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    tf_files = ["../data/dataset/tf_file_dataset/test1.data", "../data/dataset/tf_file_dataset/test2.data"]
    ds1 = ds.TFRecordDataset(tf_files, num_shards=2, shard_id=0, num_samples=3, shuffle=False, cache=some_cache)
    ds1 = ds1.repeat(4)

    ds2 = ds.TFRecordDataset(tf_files, num_shards=2, shard_id=1, num_samples=3, shuffle=False, cache=some_cache)
    ds2 = ds2.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        num_iter += 1
    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 12

    num_iter = 0
    for _ in ds2.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 12
    logger.info("test_cache_nomap_allowed_share3 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_allowed_share4():
    """
    It is allowed to share the cache between the following two trees:

       Cache                                  Cache
         |                                      |
     Map(decode, num_parallel_workers=1)    Map(decode, num_parallel_workers=2)
         |                                      |
      TFReader                              TFReader
    """

    logger.info("Test cache nomap allowed share 4")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    # This dataset has 3 records in it only
    some_cache = ds.DatasetCache(session_id=session_id, size=0)
    decode_op = c_vision.Decode()

    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    ds1 = ds1.map(operations=decode_op, input_columns=["image"], cache=some_cache, num_parallel_workers=1)

    ds2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    ds2 = ds2.map(operations=decode_op, input_columns=["image"], cache=some_cache, num_parallel_workers=2)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        num_iter += 1
    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 3

    num_iter = 0
    for _ in ds2.create_dict_iterator(num_epochs=1):
        num_iter += 1
    logger.info("Number of data in ds2: {} ".format(num_iter))
    assert num_iter == 3

    logger.info("test_cache_nomap_allowed_share4 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_disallowed_share1():
    """
    It is not allowed to share the cache between the following two trees:

       Cache       Cache
         |           |
     Map(decode) Map(rescale)
         |           |
      TFReader    TFReader
    """

    logger.info("Test cache nomap disallowed share1")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    # This dataset has 3 records in it only
    some_cache = ds.DatasetCache(session_id=session_id, size=0)
    decode_op = c_vision.Decode()
    rescale_op = c_vision.Rescale(1.0 / 255.0, -1.0)

    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    ds1 = ds1.map(operations=decode_op, input_columns=["image"], cache=some_cache)

    ds2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    ds2 = ds2.map(operations=rescale_op, input_columns=["image"], cache=some_cache)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        num_iter += 1
    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 3

    with pytest.raises(RuntimeError) as e:
        sum([1 for _ in ds2])
    assert "Cannot re-use a cache for a different tree!" in str(e.value)

    logger.info("test_cache_nomap_disallowed_share1 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_running_twice1():
    """
    Executing the same pipeline for twice (from python), with cache injected after map

       Repeat
         |
       Cache
         |
     Map(decode)
         |
     TFRecord
    """

    logger.info("Test cache nomap running twice 1")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This dataset has 3 records in it only
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR)
    decode_op = c_vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op, cache=some_cache)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator():
        num_iter += 1
    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 12

    num_iter = 0
    for _ in ds1.create_dict_iterator():
        num_iter += 1
    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 12

    logger.info("test_cache_nomap_running_twice1 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_running_twice2():
    """
    Executing the same pipeline for twice (from shell), with cache injected after leaf

       Repeat
         |
     Map(decode)
         |
       Cache
         |
     TFRecord
    """

    logger.info("Test cache nomap running twice 2")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This dataset has 3 records in it only
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, cache=some_cache)
    decode_op = c_vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator():
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 12
    logger.info("test_cache_nomap_running_twice2 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_extra_small_size1():
    """
    Test running pipeline with cache of extra small size and spilling true

       Repeat
         |
     Map(decode)
         |
       Cache
         |
     TFRecord
    """

    logger.info("Test cache nomap extra small size 1")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")
    some_cache = ds.DatasetCache(session_id=session_id, size=1, spilling=True)

    # This dataset has 3 records in it only
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, cache=some_cache)
    decode_op = c_vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator():
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 12
    logger.info("test_cache_nomap_extra_small_size1 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_extra_small_size2():
    """
    Test running pipeline with cache of extra small size and spilling false (failure)

       Repeat
         |
       Cache
         |
     Map(decode)
         |
     TFRecord
    """

    logger.info("Test cache nomap extra small size 2")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")
    some_cache = ds.DatasetCache(session_id=session_id, size=1, spilling=False)

    # This dataset has 3 records in it only
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR)
    decode_op = c_vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op, cache=some_cache)
    ds1 = ds1.repeat(4)

    with pytest.raises(RuntimeError) as e:
        sum([1 for _ in ds1])
    assert "Out of memory" in str(e.value)
    logger.info("test_cache_nomap_extra_small_size2 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_parallel_pipeline1(shard):
    """
    Test running two parallel pipelines (sharing cache) with cache injected after leaf op

       Repeat
         |
     Map(decode)
         |
       cache
         |
      TFReader
    """

    logger.info("Test cache nomap parallel pipeline 1")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")
    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This dataset has 3 records in it only
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, num_shards=3, shard_id=int(shard), cache=some_cache)
    decode_op = c_vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 4
    logger.info("test_cache_nomap_parallel_pipeline1 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_parallel_pipeline2(shard):
    """
    Test running two parallel pipelines (sharing cache) with cache injected after map op

       Repeat
         |
       cache
         |
     Map(decode)
         |
      TFReader
    """

    logger.info("Test cache nomap parallel pipeline 2")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")
    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This dataset has 3 records in it only
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, num_shards=3, shard_id=int(shard))
    decode_op = c_vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op, cache=some_cache)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 4
    logger.info("test_cache_nomap_parallel_pipeline2 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_parallel_workers():
    """
    Test cache with num_parallel_workers > 1 set for map op and leaf op

       Repeat
         |
     Map(decode)
         |
       cache
         |
      TFReader
    """

    logger.info("Test cache nomap parallel workers")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")
    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This dataset has 3 records in it only
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, num_parallel_workers=4)
    decode_op = c_vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op, num_parallel_workers=4, cache=some_cache)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 12
    logger.info("test_cache_nomap_parallel_workers Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_server_workers_1():
    """
    start cache server with --workers 1 and then test cache function

       Repeat
         |
       cache
         |
     Map(decode)
         |
      TFRecord
    """

    logger.info("Test cache nomap server workers 1")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This dataset has 3 records in it only
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR)
    decode_op = c_vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op, cache=some_cache)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator():
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 12
    logger.info("test_cache_nomap_server_workers_1 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_server_workers_100():
    """
    start cache server with --workers 100 and then test cache function

       Repeat
         |
     Map(decode)
         |
       cache
         |
      TFRecord
    """

    logger.info("Test cache nomap server workers 100")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This dataset has 3 records in it only
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, cache=some_cache)
    decode_op = c_vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator():
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 12
    logger.info("test_cache_nomap_server_workers_100 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_num_connections_1():
    """
    Test setting num_connections=1 in DatasetCache

       Repeat
         |
       cache
         |
     Map(decode)
         |
      TFRecord
    """

    logger.info("Test cache nomap num_connections 1")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0, num_connections=1)

    # This dataset has 3 records in it only
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR)
    decode_op = c_vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op, cache=some_cache)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator():
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 12
    logger.info("test_cache_nomap_num_connections_1 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_num_connections_100():
    """
    Test setting num_connections=100 in DatasetCache

       Repeat
         |
     Map(decode)
         |
       cache
         |
      TFRecord
    """

    logger.info("Test cache nomap num_connections 100")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0, num_connections=100)

    # This dataset has 3 records in it only
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, cache=some_cache)
    decode_op = c_vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator():
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 12
    logger.info("test_cache_nomap_num_connections_100 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_prefetch_size_1():
    """
    Test setting prefetch_size=1 in DatasetCache

       Repeat
         |
       cache
         |
     Map(decode)
         |
      TFRecord
    """

    logger.info("Test cache nomap prefetch_size 1")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0, prefetch_size=1)

    # This dataset has 3 records in it only
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR)
    decode_op = c_vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op, cache=some_cache)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator():
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 12
    logger.info("test_cache_nomap_prefetch_size_1 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_prefetch_size_100():
    """
    Test setting prefetch_size=100 in DatasetCache

       Repeat
         |
     Map(decode)
         |
       cache
         |
      TFRecord
    """

    logger.info("Test cache nomap prefetch_size 100")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0, prefetch_size=100)

    # This dataset has 3 records in it only
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, cache=some_cache)
    decode_op = c_vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator():
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 12
    logger.info("test_cache_nomap_prefetch_size_100 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_to_device():
    """
    Test cache with to_device

     DeviceQueue
         |
      EpochCtrl
         |
       Repeat
         |
     Map(decode)
         |
       cache
         |
      TFReader
    """

    logger.info("Test cache nomap to_device")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This dataset has 3 records in it only
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR)
    decode_op = c_vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op, cache=some_cache)
    ds1 = ds1.repeat(4)
    ds1 = ds1.to_device()
    ds1.send()

    logger.info("test_cache_nomap_to_device Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_session_destroy():
    """
    Test executing cache_admin -d while the pipeline is running

       Repeat
         |
       Cache
         |
     RandomDataset
    """

    logger.info("Test cache nomap session destroy")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    schema = ds.Schema()
    schema.add_column('image', de_type=mstype.uint8,
                      shape=[640, 480, 3])  # 921600 bytes (a bit less than 1 MB per image)
    schema.add_column('label', de_type=mstype.uint8, shape=[1])

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # User-created sampler here
    ds1 = ds.RandomDataset(schema=schema, num_parallel_workers=4, cache=some_cache)
    ds1 = ds1.repeat()

    with pytest.raises(RuntimeError) as e:
        num_iter = 0
        for _ in ds1.create_dict_iterator():
            num_iter += 1
    assert "Unexpected error" in str(e.value)

    logger.info("test_cache_nomap_session_destroy Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_server_stop():
    """
    Test executing cache_admin --stop while the pipeline is running

       Repeat
         |
       Cache
         |
     RandomDataset
    """

    logger.info("Test cache nomap server stop")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    schema = ds.Schema()
    schema.add_column('image', de_type=mstype.uint8,
                      shape=[640, 480, 3])  # 921600 bytes (a bit less than 1 MB per image)
    schema.add_column('label', de_type=mstype.uint8, shape=[1])

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # User-created sampler here
    ds1 = ds.RandomDataset(schema=schema, num_parallel_workers=4, cache=some_cache)
    ds1 = ds1.repeat()

    with pytest.raises(RuntimeError) as e:
        num_iter = 0
        for _ in ds1.create_dict_iterator():
            num_iter += 1
    assert "Network error. Cache server with port 50052 is unreachable. Make sure the server is running." in \
           str(e.value)

    logger.info("test_cache_nomap_server_stop Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_interrupt_and_rerun():
    """
    Test interrupt a running pipeline and then re-use the same cache to run another pipeline

       Cache
         |
     RandomDataset
    """

    logger.info("Test cache nomap interrupt and rerun")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    schema = ds.Schema()
    schema.add_column('image', de_type=mstype.uint8,
                      shape=[640, 480, 3])  # 921600 bytes (a bit less than 1 MB per image)
    schema.add_column('label', de_type=mstype.uint8, shape=[1])

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # User-created sampler here
    ds1 = ds.RandomDataset(schema=schema, total_rows=10000, num_parallel_workers=4, cache=some_cache)
    iter1 = ds1.create_dict_iterator()

    num_iter = 0
    with pytest.raises(AttributeError) as e:
        for _ in iter1:
            num_iter += 1
            if num_iter == 10:
                iter1.stop()
    assert "'DictIterator' object has no attribute '_runtime_context'" in str(e.value)

    num_epoch = 2
    iter2 = ds1.create_dict_iterator(num_epochs=num_epoch)
    epoch_count = 0
    for _ in range(num_epoch):
        num_iter = 0
        for _ in iter2:
            num_iter += 1
        logger.info("Number of data in ds1: {} ".format(num_iter))
        assert num_iter == 10000
        epoch_count += 1

    cache_stat = some_cache.GetStat()
    assert cache_stat.num_mem_cached == 10000

    logger.info("test_cache_nomap_interrupt_and_rerun Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_epoch_ctrl1():
    """
    Test using two-loops method to run several epochs

     Map(decode)
         |
       cache
         |
      TFRecord
    """

    logger.info("Test cache nomap epoch ctrl1")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This dataset has 3 records in it only
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, cache=some_cache)
    decode_op = c_vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op)

    num_epoch = 5
    iter1 = ds1.create_dict_iterator(num_epochs=num_epoch)

    epoch_count = 0
    for _ in range(num_epoch):
        row_count = 0
        for _ in iter1:
            row_count += 1
        logger.info("Number of data in ds1: {} ".format(row_count))
        assert row_count == 3
        epoch_count += 1
    assert epoch_count == num_epoch
    logger.info("test_cache_nomap_epoch_ctrl1 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_epoch_ctrl2():
    """
    Test using two-loops method with infinite epochs

        cache
         |
     Map(decode)
         |
      TFRecord
    """

    logger.info("Test cache nomap epoch ctrl2")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This dataset has 3 records in it only
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR)
    decode_op = c_vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op, cache=some_cache)

    num_epoch = 5
    # iter1 will always assume there is a next epoch and never shutdown
    iter1 = ds1.create_dict_iterator()

    epoch_count = 0
    for _ in range(num_epoch):
        row_count = 0
        for _ in iter1:
            row_count += 1
        logger.info("Number of data in ds1: {} ".format(row_count))
        assert row_count == 3
        epoch_count += 1
    assert epoch_count == num_epoch

    # manually stop the iterator
    iter1.stop()
    logger.info("test_cache_nomap_epoch_ctrl2 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_epoch_ctrl3():
    """
    Test using two-loops method with infinite epochs over repeat

       repeat
         |
     Map(decode)
         |
       cache
         |
      TFRecord
    """

    logger.info("Test cache nomap epoch ctrl3")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This dataset has 3 records in it only
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, cache=some_cache)
    decode_op = c_vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op)
    ds1 = ds1.repeat(2)

    num_epoch = 5
    # iter1 will always assume there is a next epoch and never shutdown
    iter1 = ds1.create_dict_iterator()

    epoch_count = 0
    for _ in range(num_epoch):
        row_count = 0
        for _ in iter1:
            row_count += 1
        logger.info("Number of data in ds1: {} ".format(row_count))
        assert row_count == 6
        epoch_count += 1
    assert epoch_count == num_epoch

    # reply on garbage collector to destroy iter1

    logger.info("test_cache_nomap_epoch_ctrl3 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_epoch_ctrl4():
    """
    Test using two-loops method with repeat under cache

        cache
         |
     Map(decode)
         |
       repeat
         |
      TFRecord
    """

    logger.info("Test cache nomap epoch ctrl4")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This dataset has 3 records in it only
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR)
    ds1 = ds1.repeat(2)
    decode_op = c_vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op, cache=some_cache)

    num_epoch = 5
    iter1 = ds1.create_dict_iterator(num_epochs=num_epoch)

    epoch_count = 0
    for _ in range(num_epoch):
        row_count = 0
        for _ in iter1:
            row_count += 1
        logger.info("Number of data in ds1: {} ".format(row_count))
        assert row_count == 6
        epoch_count += 1
    assert epoch_count == num_epoch

    logger.info("test_cache_nomap_epoch_ctrl4 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_multiple_cache1():
    """
    Test multiple cache in the same python script

       cache                  cache
         |                      |
    Map(decode)             Map(decode)
         |                      |
    TFRecord(train)        TFRecord(eval)
    """

    logger.info("Test cache nomap multiple cache 1")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    train_cache = ds.DatasetCache(session_id=session_id, size=0)
    eval_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This dataset has 12 records in it
    train_dataset = ds.TFRecordDataset(TRAIN_DATA_DIR, TRAIN_SCHEMA_DIR)
    decode_op = c_vision.Decode()
    train_dataset = train_dataset.map(input_columns=["image"], operations=decode_op, cache=train_cache)

    # This dataset has 3 records in it only
    eval_dataset = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR)
    eval_dataset = eval_dataset.map(input_columns=["image"], operations=decode_op, cache=eval_cache)

    num_epoch = 5
    train_iter = train_dataset.create_dict_iterator(num_epochs=num_epoch)
    eval_iter = eval_dataset.create_dict_iterator(num_epochs=num_epoch)

    epoch_count = 0
    for _ in range(num_epoch):
        assert sum([1 for _ in train_iter]) == 12
        assert sum([1 for _ in eval_iter]) == 3
        epoch_count += 1
    assert epoch_count == num_epoch

    logger.info("test_cache_nomap_multiple_cache1 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_multiple_cache2():
    """
    Test multiple cache in the same python script

       cache
         |
    Map(decode)               cache
         |                      |
    TFRecord(image)        TFRecord(text)
    """

    logger.info("Test cache nomap multiple cache 2")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    image_cache = ds.DatasetCache(session_id=session_id, size=0)
    text_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This dataset has 3 records in it only
    image_dataset = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR)
    decode_op = c_vision.Decode()
    image_dataset = image_dataset.map(input_columns=["image"], operations=decode_op, cache=image_cache)

    # This dataset has 3 records in it only
    text_dataset = ds.TFRecordDataset(TEXT_TF_DATA_DIR, SCHEMA_DIR2, cache=text_cache)

    num_epoch = 5
    image_iter = image_dataset.create_dict_iterator(num_epochs=num_epoch)
    text_iter = text_dataset.create_dict_iterator(num_epochs=num_epoch, output_numpy=True)

    epoch_count = 0
    for _ in range(num_epoch):
        row_count = 0
        for _, _ in itertools.zip_longest(image_iter, text_iter):
            row_count += 1
        assert row_count == 3
        epoch_count += 1
    assert epoch_count == num_epoch

    logger.info("test_cache_nomap_multiple_cache2 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_multiple_cache3():
    """
    Test multiple cache in the same python script

       cache                   cache
         |                      |
    Map(decode)             Map(decode)
         |                      |
    TFRecord                ImageFolder
    """

    logger.info("Test cache nomap multiple cache 3")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    tf_cache = ds.DatasetCache(session_id=session_id, size=0)
    image_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This dataset has 3 records in it only
    tf_dataset = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR)
    decode_op = c_vision.Decode()
    tf_dataset = tf_dataset.map(input_columns=["image"], operations=decode_op, cache=tf_cache)

    # This DATA_DIR only has 2 images in it
    image_dataset = ds.ImageFolderDataset(dataset_dir=IMAGE_FOLDER_DATA_DIR)
    image_dataset = image_dataset.map(input_columns=["image"], operations=decode_op, cache=image_cache)

    num_epoch = 5
    tf_iter = tf_dataset.create_dict_iterator(num_epochs=num_epoch)
    image_iter = image_dataset.create_dict_iterator(num_epochs=num_epoch)

    epoch_count = 0
    for _ in range(num_epoch):
        assert sum([1 for _ in tf_iter]) == 3
        assert sum([1 for _ in image_iter]) == 2
        epoch_count += 1
    assert epoch_count == num_epoch

    logger.info("test_cache_nomap_multiple_cache3 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_multiple_cache_train():
    """
    Test multiple cache in different python scripts. This test case is going to run concurrently with
    test_cache_nomap_multiple_cache_eval.

       cache
         |
    Map(decode)
         |
    TFRecord(train)
    """

    logger.info("Test cache nomap multiple cache train")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    train_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This dataset has 12 records in it
    train_dataset = ds.TFRecordDataset(TRAIN_DATA_DIR, TRAIN_SCHEMA_DIR)
    decode_op = c_vision.Decode()
    train_dataset = train_dataset.map(input_columns=["image"], operations=decode_op, cache=train_cache)

    num_epoch = 5
    train_iter = train_dataset.create_dict_iterator(num_epochs=num_epoch)

    epoch_count = 0
    for _ in range(num_epoch):
        assert sum([1 for _ in train_iter]) == 12
        epoch_count += 1
    assert epoch_count == num_epoch

    logger.info("test_cache_nomap_multiple_cache_train Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_multiple_cache_eval():
    """
    Test multiple cache in different python scripts. This test case is going to run concurrently with
    test_cache_nomap_multiple_cache_train.

       cache
         |
    Map(decode)
         |
    TFRecord(eval)
    """

    logger.info("Test cache nomap multiple cache eval")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    eval_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This dataset only has 3 records in it
    eval_dataset = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR)
    decode_op = c_vision.Decode()
    eval_dataset = eval_dataset.map(input_columns=["image"], operations=decode_op, cache=eval_cache)

    num_epoch = 5
    eval_iter = eval_dataset.create_dict_iterator(num_epochs=num_epoch)

    epoch_count = 0
    for _ in range(num_epoch):
        assert sum([1 for _ in eval_iter]) == 3
        epoch_count += 1
    assert epoch_count == num_epoch

    logger.info("test_cache_nomap_multiple_cache_eval Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_clue1():
    """
    A clue dataset (a non mappable dataset) with a cache over it just after the leaf
    In this one, the clue dataset will be given sharding configuration, however since a cache is
    used, the tree prepare should undo the sharding configuration and instead, a distributed
    sampler will be chosen with the same shard config.

       Cache
         |
       CLUE
    """

    logger.info("Test cache nomap clue 1")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # With only 3 records shard into 3, we expect only 1 record returned for this shard
    # However, the sharding will be done by the sampler, not by the clue leaf node
    # In this case, it is a row-based sharding, not the file-based sharding that would happen if
    # there was not any cache.
    ds1 = ds.CLUEDataset(CLUE_DATA_DIR, task='AFQMC', usage='train', num_shards=3, shard_id=1, cache=some_cache)

    num_epoch = 4
    iter1 = ds1.create_dict_iterator(num_epochs=num_epoch, output_numpy=True)

    epoch_count = 0
    for _ in range(num_epoch):
        assert sum([1 for _ in iter1]) == 1
        epoch_count += 1
    assert epoch_count == num_epoch

    logger.info("test_cache_nomap_clue1 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_clue2():
    """
    A clue dataset (a non mappable dataset) with a cache over it after map
    In this one, a num_samples argument is given

       Cache
         |
    map(lambda x: x)
         |
       CLUE
    """

    logger.info("Test cache nomap clue 2")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    ds1 = ds.CLUEDataset(CLUE_DATA_DIR, task='AFQMC', usage='train', num_samples=2)
    ds1 = ds1.map(py_vision.not_random(lambda x: x), ["label"], cache=some_cache)

    num_epoch = 4
    iter1 = ds1.create_dict_iterator(num_epochs=num_epoch, output_numpy=True)

    epoch_count = 0
    for _ in range(num_epoch):
        assert sum([1 for _ in iter1]) == 2
        epoch_count += 1
    assert epoch_count == num_epoch

    logger.info("test_cache_nomap_clue2 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_csv1():
    """
    A csv dataset (a non mappable dataset) with a cache over it just after the leaf
    In this one, the csv dataset will be given sharding configuration, however since a cache is
    used, the tree prepare should undo the sharding configuration and instead, a distributed
    sampler will be chosen with the same shard config.

       Cache
         |
       CSV
    """

    logger.info("Test cache nomap csv 1")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # With only 3 records shard into 3, we expect only 1 record returned for this shard
    # However, the sharding will be done by the sampler, not by the clue leaf node
    # In this case, it is a row-based sharding, not the file-based sharding that would happen if
    # there was not any cache.
    ds1 = ds.CSVDataset(CSV_DATA_DIR, column_defaults=["1", "2", "3", "4"],
                        column_names=['col1', 'col2', 'col3', 'col4'], num_shards=3, shard_id=1, cache=some_cache)

    num_epoch = 4
    iter1 = ds1.create_dict_iterator(num_epochs=num_epoch, output_numpy=True)

    epoch_count = 0
    for _ in range(num_epoch):
        assert sum([1 for _ in iter1]) == 1
        epoch_count += 1
    assert epoch_count == num_epoch

    logger.info("test_cache_nomap_csv1 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_csv2():
    """
    A csv dataset (a non mappable dataset) with a cache over it after map
    In this one, a num_samples argument is given

       Cache
         |
    map(lambda x: x)
         |
       CSV
    """

    logger.info("Test cache nomap csv 2")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    ds1 = ds.CSVDataset(CSV_DATA_DIR, column_defaults=["1", "2", "3", "4"],
                        column_names=['col1', 'col2', 'col3', 'col4'], num_samples=2)
    ds1 = ds1.map(py_vision.not_random(lambda x: x), ["col1"], cache=some_cache)

    num_epoch = 4
    iter1 = ds1.create_dict_iterator(num_epochs=num_epoch, output_numpy=True)

    epoch_count = 0
    for _ in range(num_epoch):
        assert sum([1 for _ in iter1]) == 2
        epoch_count += 1
    assert epoch_count == num_epoch

    logger.info("test_cache_nomap_csv2 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_textfile1():
    """
    A text file dataset (a non mappable dataset) with a cache over it just after the leaf
    In this one, the text file dataset will be given sharding configuration, however since a cache is
    used, the tree prepare should undo the sharding configuration and instead, a distributed
    sampler will be chosen with the same shard config.

       Cache
         |
     TextFile
    """

    logger.info("Test cache nomap textfile 1")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # With only 3 records shard into 3, we expect only 1 record returned for this shard
    # However, the sharding will be done by the sampler, not by the clue leaf node
    # In this case, it is a row-based sharding, not the file-based sharding that would happen if
    # there was not any cache.
    ds1 = ds.TextFileDataset(TEXT_FILE_DATA_DIR, num_shards=3, shard_id=1, cache=some_cache)

    num_epoch = 4
    iter1 = ds1.create_dict_iterator(num_epochs=num_epoch, output_numpy=True)

    epoch_count = 0
    for _ in range(num_epoch):
        assert sum([1 for _ in iter1]) == 1
        epoch_count += 1
    assert epoch_count == num_epoch

    logger.info("test_cache_nomap_textfile1 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_textfile2():
    """
    A text file dataset (a non mappable dataset) with a cache over it after map
    In this one, a num_samples argument is given

       Cache
         |
    Map(tokenizer)
         |
     TextFile
    """

    def my_tokenizer(line):
        words = line.split()
        if not words:
            return [""]
        return words

    logger.info("Test cache nomap textfile 2")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    ds1 = ds.TextFileDataset(TEXT_FILE_DATA_DIR, num_samples=2)
    tokenizer = text.PythonTokenizer(my_tokenizer)
    ds1 = ds1.map(operations=tokenizer, cache=some_cache)

    num_epoch = 4
    iter1 = ds1.create_dict_iterator(num_epochs=num_epoch, output_numpy=True)

    epoch_count = 0
    for _ in range(num_epoch):
        assert sum([1 for _ in iter1]) == 2
        epoch_count += 1
    assert epoch_count == num_epoch

    logger.info("test_cache_nomap_textfile2 Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_nested_repeat():
    """
    Test cache on pipeline with nested repeat ops

        Repeat
          |
        Cache
          |
      Map(decode)
          |
        Repeat
          |
      TFRecord
    """

    logger.info("Test cache nomap nested repeat")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This dataset has 3 records in it only
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR)
    decode_op = c_vision.Decode()
    ds1 = ds1.repeat(4)
    ds1 = ds1.map(operations=decode_op, input_columns=["image"], cache=some_cache)
    ds1 = ds1.repeat(2)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        logger.info("get data from dataset")
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 24
    logger.info('test_cache_nomap_nested_repeat Ended.\n')


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_get_repeat_count():
    """
    Test get_repeat_count() for a pipeline with cache and nested repeat ops

        Cache
          |
      Map(decode)
          |
        Repeat
          |
      TFRecord
    """

    logger.info("Test cache nomap get_repeat_count")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This dataset has 3 records in it only
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    ds1 = ds1.repeat(4)
    decode_op = c_vision.Decode()
    ds1 = ds1.map(operations=decode_op, input_columns=["image"], cache=some_cache)

    repeat_count = ds1.get_repeat_count()
    logger.info("repeat_count: {}".format(repeat_count))
    assert repeat_count == 4

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        logger.info("get data from dataset")
        num_iter += 1
    assert num_iter == 12


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_long_file_list():
    """
    Test cache after TFRecord with a long list of files as arguments

        Cache
          |
      TFRecord
    """

    logger.info("Test cache nomap long file list")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=1)

    ds1 = ds.TFRecordDataset([DATA_DIR[0] for _ in range(0, 1000)], SCHEMA_DIR, columns_list=["image"],
                             cache=some_cache)

    with pytest.raises(RuntimeError) as e:
        sum([1 for _ in ds1])
    assert "Out of memory" in str(e.value)
    logger.info("test_cache_nomap_long_file_list Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_failure1():
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
      TFRecord

    """
    logger.info("Test cache nomap failure 1")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This dataset has 3 records in it only
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, cache=some_cache)
    decode_op = c_vision.Decode()
    ds1 = ds1.map(operations=decode_op, input_columns=["image"], cache=some_cache)
    ds1 = ds1.repeat(4)

    with pytest.raises(RuntimeError) as e:
        ds1.get_batch_size()
    assert "Nested cache operations" in str(e.value)

    with pytest.raises(RuntimeError) as e:
        num_iter = 0
        for _ in ds1.create_dict_iterator(num_epochs=1):
            num_iter += 1
    assert "Nested cache operations" in str(e.value)

    assert num_iter == 0
    logger.info('test_cache_nomap_failure1 Ended.\n')


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_failure2():
    """
    Test zip under cache (failure)

               repeat
                  |
                Cache
                  |
             Map(decode)
                  |
                 Zip
                |    |
           Random    Random

    """
    logger.info("Test cache nomap failure 2")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    schema = ds.Schema()
    schema.add_column('image', de_type=mstype.uint8,
                      shape=[640, 480, 3])  # 921600 bytes (a bit less than 1 MB per image)
    schema.add_column('label', de_type=mstype.uint8, shape=[1])

    ds1 = ds.RandomDataset(schema=schema)
    ds2 = ds.RandomDataset(schema=schema)
    dsz = ds.zip((ds1, ds2))
    decode_op = c_vision.Decode()
    dsz = dsz.map(input_columns=["image"], operations=decode_op, cache=some_cache)
    dsz = dsz.repeat(4)

    with pytest.raises(RuntimeError) as e:
        num_iter = 0
        for _ in dsz.create_dict_iterator():
            num_iter += 1
    assert "ZipNode is not supported as a descendant operator under a cache" in str(e.value)

    assert num_iter == 0
    logger.info('test_cache_nomap_failure2 Ended.\n')


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_failure3():
    """
    Test batch under cache (failure)

               repeat
                  |
                Cache
                  |
             Map(resize)
                  |
                Batch
                  |
                Clue
    """
    logger.info("Test cache nomap failure 3")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    ds1 = ds.CLUEDataset(CLUE_DATA_DIR, task='AFQMC', usage='train')
    ds1 = ds1.batch(2)
    resize_op = c_vision.Resize((224, 224))
    ds1 = ds1.map(input_columns=["image"], operations=resize_op, cache=some_cache)
    ds1 = ds1.repeat(4)

    with pytest.raises(RuntimeError) as e:
        num_iter = 0
        for _ in ds1.create_dict_iterator():
            num_iter += 1
    assert "BatchNode is not supported as a descendant operator under a cache" in str(e.value)

    assert num_iter == 0
    logger.info('test_cache_nomap_failure3 Ended.\n')


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_failure4():
    """
    Test filter under cache (failure)

               repeat
                  |
                Cache
                  |
             Map(decode)
                  |
                Filter
                  |
                 CSV

    """
    logger.info("Test cache nomap failure 4")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    ds1 = ds.CSVDataset(CSV_DATA_DIR, column_defaults=["1", "2", "3", "4"],
                        column_names=['col1', 'col2', 'col3', 'col4'])
    ds1 = ds1.filter(predicate=lambda data: data < 11, input_columns=["label"])

    decode_op = c_vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op, cache=some_cache)
    ds1 = ds1.repeat(4)

    with pytest.raises(RuntimeError) as e:
        num_iter = 0
        for _ in ds1.create_dict_iterator():
            num_iter += 1
    assert "FilterNode is not supported as a descendant operator under a cache" in str(e.value)

    assert num_iter == 0
    logger.info('test_cache_nomap_failure4 Ended.\n')


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_failure5():
    """
    Test Map containing random operation under cache (failure)

               repeat
                  |
                Cache
                  |
             Map(decode, randomCrop)
                  |
              TextFile

    """
    logger.info("Test cache nomap failure 5")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    data = ds.TextFileDataset(TEXT_FILE_DATA_DIR)
    random_crop_op = c_vision.RandomCrop([512, 512], [200, 200, 200, 200])
    decode_op = c_vision.Decode()

    data = data.map(input_columns=["image"], operations=decode_op)
    data = data.map(input_columns=["image"], operations=random_crop_op, cache=some_cache)
    data = data.repeat(4)

    with pytest.raises(RuntimeError) as e:
        num_iter = 0
        for _ in data.create_dict_iterator():
            num_iter += 1
    assert "MapNode containing random operation is not supported as a descendant of cache" in str(e.value)

    assert num_iter == 0
    logger.info('test_cache_nomap_failure5 Ended.\n')


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_pyfunc_lambda():
    """
    Test cache after map op with a python lambda function.
    Only allowed if the lambda function is wrapped by 'pyvision.not_random', otherwise an error will be raised.

        Cache
          |
        Map(lambda function1, lambda function2)
          |
      TFRecord
    """
    logger.info("Test cache nomap pyfunc lambda")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # This dataset has 12 records in it
    data1 = ds.TFRecordDataset(PYFUNC_DATA_DIR, PYFUNC_SCHEMA_DIR, shuffle=False)
    transforms = [py_vision.not_random(lambda x: x + x), py_vision.not_random(lambda x: x - 1)]
    data1 = data1.map(operations=transforms, input_columns="col0", cache=some_cache)

    num_iter = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 12

    other_cache = ds.DatasetCache(session_id=session_id, size=0)
    ds2 = ds.TFRecordDataset(PYFUNC_DATA_DIR, PYFUNC_SCHEMA_DIR, shuffle=False)
    ds2 = ds2.map(operations=[(lambda x: x + x)], input_columns=["col0"], cache=other_cache)

    with pytest.raises(RuntimeError) as e:
        num_iter = 0
        for _ in ds2.create_dict_iterator():
            num_iter += 1
    assert "MapNode containing random operation is not supported as a descendant of cache" in str(e.value)
    logger.info("test_cache_nomap_pyfunc_lambda Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_pyfunc_builtin():
    """
    Test cache after map op with a python builtin PyFunc.
    An error will be raised if the builtin pyfunc containing random operation.

        Cache
          |
     Map([builtin pyfunc1, builtin pyfunc2])
          |
      TFRecord
    """
    logger.info("Test cache nomap pyfunc builtin")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)
    # This dataset has 3 records in it only
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"])
    ds1 = ds1.map(operations=[py_vision.Decode(), py_vision.ToTensor()], input_columns=["image"], cache=some_cache)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 3

    other_cache = ds.DatasetCache(session_id=session_id, size=0)
    # This dataset has 3 records in it only
    ds2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"])
    ds2 = ds2.map(operations=[py_vision.Decode(), py_vision.RandomCrop(224), py_vision.ToTensor()],
                  input_columns=["image"], cache=other_cache)

    with pytest.raises(RuntimeError) as e:
        num_iter = 0
        for _ in ds2.create_dict_iterator():
            num_iter += 1
    assert "MapNode containing random operation is not supported as a descendant of cache" in str(e.value)
    logger.info("test_cache_nomap_pyfunc_builtin Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_pyfunc_function():
    """
    Test cache after map op with a python customized function.
    Only allowed if the function is decorated with 'py_vision.not_random', otherwise an error will be raised.

        Cache
          |
     Map([function1, function2])
          |
      TFRecord
    """

    @py_vision.not_random
    def not_random_func(x):
        return np.ones(x.shape, dtype=x.dtype)

    def normal_func(x):
        return np.ones(x.shape, dtype=x.dtype)

    logger.info("Test cache nomap pyfunc function")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    some_cache = ds.DatasetCache(session_id=session_id, size=0)
    # This dataset has 3 records in it only
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"])
    ds1 = ds1.map(operations=[not_random_func, not_random_func], input_columns=["image"], cache=some_cache)

    num_iter = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 3

    other_cache = ds.DatasetCache(session_id=session_id, size=0)
    # This dataset has 3 records in it only
    ds2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"])
    ds2 = ds2.map(operations=[not_random_func, normal_func], input_columns=["image"], cache=other_cache)

    with pytest.raises(RuntimeError) as e:
        num_iter = 0
        for _ in ds2.create_dict_iterator():
            num_iter += 1
    assert "MapNode containing random operation is not supported as a descendant of cache" in str(e.value)
    logger.info("test_cache_nomap_pyfunc_function Ended.\n")


@pytest.mark.skipif(os.environ.get('RUN_CACHE_TEST') != 'TRUE', reason="Require to bring up cache server")
def test_cache_nomap_all_rows_cached():
    """
    Make sure all rows are cached before we switch to the fetching phase

       Cache
         |
     RandomDataset
    """

    logger.info("Test cache nomap all rows cached")
    if "SESSION_ID" in os.environ:
        session_id = int(os.environ['SESSION_ID'])
    else:
        raise RuntimeError("Testcase requires SESSION_ID environment variable")

    schema = ds.Schema()
    schema.add_column('image', de_type=mstype.uint8,
                      shape=[450, 450, 3])
    schema.add_column('label', de_type=mstype.uint8, shape=[1])

    some_cache = ds.DatasetCache(session_id=session_id, size=0)

    # easier to reproduce the problem with 271 total rows
    num_total_rows = 271
    # User-created sampler here
    ds1 = ds.RandomDataset(schema=schema, total_rows=num_total_rows, num_parallel_workers=4, cache=some_cache)
    iter1 = ds1.create_dict_iterator()

    num_iter = 0
    for _ in iter1:
        num_iter += 1
    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == num_total_rows

    cache_stat = some_cache.GetStat()
    assert cache_stat.num_mem_cached == num_total_rows

    logger.info("test_cache_nomap_all_rows_cached Ended.\n")


if __name__ == '__main__':
    # This is just a list of tests, don't try to run these tests with 'python test_cache_nomap.py'
    # since cache server is required to be brought up first
    test_cache_nomap_basic1()
    test_cache_nomap_basic2()
    test_cache_nomap_basic3()
    test_cache_nomap_basic4()
    test_cache_nomap_basic5()
    test_cache_nomap_basic6()
    test_cache_nomap_basic7()
    test_cache_nomap_basic8()
    test_cache_nomap_basic9()
    test_cache_nomap_allowed_share1()
    test_cache_nomap_allowed_share2()
    test_cache_nomap_allowed_share3()
    test_cache_nomap_allowed_share4()
    test_cache_nomap_disallowed_share1()
    test_cache_nomap_running_twice1()
    test_cache_nomap_running_twice2()
    test_cache_nomap_extra_small_size1()
    test_cache_nomap_extra_small_size2()
    test_cache_nomap_parallel_pipeline1(shard=0)
    test_cache_nomap_parallel_pipeline2(shard=1)
    test_cache_nomap_parallel_workers()
    test_cache_nomap_server_workers_1()
    test_cache_nomap_server_workers_100()
    test_cache_nomap_num_connections_1()
    test_cache_nomap_num_connections_100()
    test_cache_nomap_prefetch_size_1()
    test_cache_nomap_prefetch_size_100()
    test_cache_nomap_to_device()
    test_cache_nomap_session_destroy()
    test_cache_nomap_server_stop()
    test_cache_nomap_epoch_ctrl1()
    test_cache_nomap_epoch_ctrl2()
    test_cache_nomap_epoch_ctrl3()
    test_cache_nomap_epoch_ctrl4()
    test_cache_nomap_multiple_cache1()
    test_cache_nomap_multiple_cache2()
    test_cache_nomap_multiple_cache3()
    test_cache_nomap_multiple_cache_train()
    test_cache_nomap_multiple_cache_eval()
    test_cache_nomap_clue1()
    test_cache_nomap_clue2()
    test_cache_nomap_csv1()
    test_cache_nomap_csv2()
    test_cache_nomap_textfile1()
    test_cache_nomap_textfile2()
    test_cache_nomap_nested_repeat()
    test_cache_nomap_get_repeat_count()
    test_cache_nomap_long_file_list()
    test_cache_nomap_failure1()
    test_cache_nomap_failure2()
    test_cache_nomap_failure3()
    test_cache_nomap_failure4()
    test_cache_nomap_failure5()
    test_cache_nomap_pyfunc_lambda()
    test_cache_nomap_pyfunc_builtin()
    test_cache_nomap_pyfunc_function()
