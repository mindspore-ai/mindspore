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
Testing cache operator with non-mappable datasets
"""
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.c_transforms as c_vision
from mindspore import log as logger

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

GENERATE_GOLDEN = False

def test_cache_nomap_basic1():
    """
    A random dataset (a non mappable dataset) with a cache over it just after the leaf
    """

    logger.info("Test cache nomap basic 1")

    schema = ds.Schema()
    schema.add_column('image', de_type=mstype.uint8,
                      shape=[640, 480, 3])  # 921600 bytes (a bit less than 1 MB per image)
    schema.add_column('label', de_type=mstype.uint8, shape=[1])

    # create a cache.  arbitrary session_id for now
    some_cache = ds.DatasetCache(session_id=1, size=0, spilling=True)

    # User-created sampler here
    ds1 = ds.RandomDataset(schema=schema, total_rows=10, num_parallel_workers=4, cache=some_cache)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for data in ds1.create_dict_iterator():
        logger.info("printing the label: {}".format(data["label"]))
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 40
    logger.info("test_cache_nomap_basic1 Ended.\n")


def test_cache_nomap_basic2():
    """
    A random dataset (a non mappable dataset) with a cache over it just after the leaf
    """

    logger.info("Test cache nomap basic 2")

    schema = ds.Schema()
    schema.add_column('image', de_type=mstype.uint8,
                      shape=[640, 480, 3])  # 921600 bytes (a bit less than 1 MB per image)
    schema.add_column('label', de_type=mstype.uint8, shape=[1])

    # create a cache.  arbitrary session_id for now
    some_cache = ds.DatasetCache(session_id=1, size=0, spilling=True)

    # sampler arg not given directly, however any of these args will auto-generate an appropriate sampler:
    # num_samples, shuffle, num_shards, shard_id
    # In this case, the presence of num_samples chooses a sampler.
    ds1 = ds.RandomDataset(schema=schema, total_rows=20, num_samples=20, num_parallel_workers=4, cache=some_cache)
    ds1 = ds1.repeat(2)

    num_iter = 0
    for data in ds1.create_dict_iterator():
        logger.info("printing the label: {}".format(data["label"]))
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 40
    logger.info("test_cache_nomap_basic2 Ended.\n")


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

    some_cache = ds.DatasetCache(session_id=1, size=0, spilling=True)
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False, cache=some_cache)
    decode_op = c_vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator():
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 12
    logger.info("test_cache_nomap_basic3 Ended.\n")


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

    # This dataset has 3 records in it only
    some_cache = ds.DatasetCache(session_id=1, size=0, spilling=True)
    # With shuffle not being set, TF defaults to a "global" shuffle when there is no cache
    # in the picture.  This causes a shuffle-injection over the TF.  For clarify, this test will
    # explicitly give the global option, even though it's the default in python.
    # But, when caching is added in the ascendent tree above TF, we do global shuffling
    # through the sampler over the cache, not by the shuffle op.  In that case, tree prepare
    # will remove the shuffle op that got injected by the initial tree creation.
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=ds.Shuffle.GLOBAL)
    decode_op = c_vision.Decode()

    ds1 = ds1.map(input_columns=["image"], operations=decode_op, cache=some_cache)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator():
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 12
    logger.info("test_cache_nomap_basic4 Ended.\n")


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

    # This dataset has 3 records in it only
    some_cache = ds.DatasetCache(session_id=1, size=0, spilling=True)
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], cache=some_cache)
    decode_op = c_vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator():
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 12
    logger.info("test_cache_nomap_basic5 Ended.\n")


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

    # This dataset has 3 records in it only
    some_cache = ds.DatasetCache(session_id=1, size=0, spilling=True)

    # With only 3 records shard into 3, we expect only 1 record returned for this shard
    # However, the sharding will be done by the sampler, not by the tf record leaf node
    # In this case, it is a row-based sharding, not the file-based sharding that would happen if
    # there was not any cache.
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], num_shards=3, shard_id=1, cache=some_cache)
    decode_op = c_vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator():
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 4
    logger.info("test_cache_nomap_basic6 Ended.\n")


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

    # This dataset has 3 records in it only
    some_cache = ds.DatasetCache(session_id=1, size=0, spilling=True)

    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=ds.Shuffle.GLOBAL, cache=some_cache)
    decode_op = c_vision.Decode()
    ds1 = ds1.map(input_columns=["image"], operations=decode_op)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator():
        num_iter += 1

    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 12
    logger.info("test_cache_nomap_basic7 Ended.\n")


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

    ds.config.set_seed(1)
    # This dataset has 3 records in it only
    some_cache = ds.DatasetCache(session_id=1, size=0, spilling=True)
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False, cache=some_cache)
    ds1 = ds1.repeat(4)

    ds2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False, cache=some_cache)
    ds2 = ds2.shuffle(buffer_size=2)

    num_iter = 0
    for _ in ds1.create_dict_iterator():
        num_iter += 1
    assert num_iter == 12
    logger.info("Number of data in ds1: {} ".format(num_iter))

    num_iter = 0
    for _ in ds2.create_dict_iterator():
        num_iter += 1
    assert num_iter == 3
    logger.info("test_cache_nomap_allowed_share1 Ended.\n")


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

    ds.config.set_seed(1)
    # This dataset has 3 records in it only
    some_cache = ds.DatasetCache(session_id=2, size=0, spilling=True)
    decode_op = c_vision.Decode()

    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    ds1 = ds1.map(input_columns=["image"], operations=decode_op, cache=some_cache)
    ds1 = ds1.repeat(4)

    ds2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    ds2 = ds2.map(input_columns=["image"], operations=decode_op, cache=some_cache)
    ds2 = ds2.shuffle(buffer_size=2)

    num_iter = 0
    for _ in ds1.create_dict_iterator():
        num_iter += 1
    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 12

    num_iter = 0
    for _ in ds2.create_dict_iterator():
        num_iter += 1
    assert num_iter == 3
    logger.info("test_cache_nomap_allowed_share2 Ended.\n")


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

    some_cache = ds.DatasetCache(session_id=1, size=0, spilling=True)

    tf_files = ["../data/dataset/tf_file_dataset/test1.data", "../data/dataset/tf_file_dataset/test2.data"]
    ds1 = ds.TFRecordDataset(tf_files, num_shards=2, shard_id=0, num_samples=3, shuffle=False, cache=some_cache)
    ds1 = ds1.repeat(4)

    ds2 = ds.TFRecordDataset(tf_files, num_shards=2, shard_id=1, num_samples=3, shuffle=False, cache=some_cache)
    ds2 = ds2.repeat(4)

    num_iter = 0
    for _ in ds1.create_dict_iterator():
        num_iter += 1
    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 12

    num_iter = 0
    for _ in ds2.create_dict_iterator():
        num_iter += 1
    assert num_iter == 12
    logger.info("test_cache_nomap_allowed_share3 Ended.\n")


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

    # This dataset has 3 records in it only
    some_cache = ds.DatasetCache(session_id=2, size=0, spilling=True)
    decode_op = c_vision.Decode()

    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    ds1 = ds1.map(input_columns=["image"], operations=decode_op, cache=some_cache, num_parallel_workers=1)

    ds2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    ds2 = ds2.map(input_columns=["image"], operations=decode_op, cache=some_cache, num_parallel_workers=2)

    num_iter = 0
    for _ in ds1.create_dict_iterator():
        num_iter += 1
    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 3

    num_iter = 0
    for _ in ds2.create_dict_iterator():
        num_iter += 1
    logger.info("Number of data in ds2: {} ".format(num_iter))
    assert num_iter == 3

    logger.info("test_cache_nomap_allowed_share4 Ended.\n")


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

    # This dataset has 3 records in it only
    some_cache = ds.DatasetCache(session_id=1, size=0, spilling=True)
    decode_op = c_vision.Decode()
    rescale_op = c_vision.Rescale(1.0 / 255.0, -1.0)

    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    ds1 = ds1.map(input_columns=["image"], operations=decode_op, cache=some_cache)

    ds2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    ds2 = ds2.map(input_columns=["image"], operations=rescale_op, cache=some_cache)

    num_iter = 0
    for _ in ds1.create_dict_iterator():
        num_iter += 1
    logger.info("Number of data in ds1: {} ".format(num_iter))
    assert num_iter == 3

    try:
        sum([1 for _ in ds2])
    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Attempt to re-use a cache for a different tree!" in str(e)

    logger.info("test_cache_nomap_disallowed_share1 Ended.\n")


if __name__ == '__main__':
    test_cache_nomap_basic1()
    test_cache_nomap_basic2()
    test_cache_nomap_basic3()
    test_cache_nomap_basic4()
    test_cache_nomap_basic5()
    test_cache_nomap_basic6()
    test_cache_nomap_basic7()
    test_cache_nomap_allowed_share1()
    test_cache_nomap_allowed_share2()
    test_cache_nomap_allowed_share3()
    test_cache_nomap_allowed_share4()
    test_cache_nomap_disallowed_share1()
