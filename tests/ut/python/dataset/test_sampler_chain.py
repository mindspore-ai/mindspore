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
"""Dataset regression coverage for sampler chain."""
import numpy as np
import pytest
import mindspore.dataset as ds
from mindspore import log as logger
from util import save_and_check_md5

GENERATE_GOLDEN = False

IMAGENET_RAWDATA_DIR = "../data/dataset/testImageNetData2/train"
IMAGENET_TFFILE_DIR = ["../data/dataset/test_tf_file_3_images2/train-0000-of-0001.data",
                       "../data/dataset/test_tf_file_3_images2/train-0000-of-0002.data",
                       "../data/dataset/test_tf_file_3_images2/train-0000-of-0003.data",
                       "../data/dataset/test_tf_file_3_images2/train-0000-of-0004.data"]
MNIST_DATA_DIR = "../data/dataset/testMnistData"
MANIFEST_DATA_FILE = "../data/dataset/testManifestData/test.manifest"
CIFAR10_DATA_DIR = "../data/dataset/testCifar10Data"
COCO_DATA_DIR = "../data/dataset/testCOCO/train/"
ANNOTATION_FILE = "../data/dataset/testCOCO/annotations/train.json"
VOC_DATA_DIR = "../data/dataset/testVOC2012"


def test_numpyslices_sampler_no_chain():
    """
    Feature: Chained Sampler
    Description: NumpySlicesDataset with sampler, no chain
    Expectation: Data verified to be correct
    """
    logger.info("test_numpyslices_sampler_no_chain")

    # Create NumpySlicesDataset with sampler, no chain
    np_data = [1, 2, 3, 4]
    sampler = ds.SequentialSampler(start_index=1, num_samples=2)
    data1 = ds.NumpySlicesDataset(np_data, sampler=sampler)

    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 2

    # Verify number of rows
    assert sum((1 for _ in data1)) == 2

    # Verify dataset contents
    res = []
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        logger.info("item: {}".format(item))
        res.append(item)
    logger.info("dataset: {}".format(res))

    np.testing.assert_array_equal(res, [[2], [3]])


def test_numpyslices_sampler_chain():
    """
    Feature: Chained Sampler
    Description: NumpySlicesDataset with sampler chain; add child sampler with 1 statement
    Expectation: Data verified to be correct
    """
    logger.info("test_numpyslices_sampler_chain")

    # Create NumpySlicesDataset with sampler chain
    # Use 1 statement to add child sampler
    np_data = [1, 2, 3, 4]
    sampler = ds.SequentialSampler(start_index=1, num_samples=2)
    sampler.add_child(ds.SequentialSampler(start_index=1, num_samples=2))
    data1 = ds.NumpySlicesDataset(np_data, sampler=sampler)

    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 1

    # Verify number of rows
    assert sum((1 for _ in data1)) == 1

    # Verify dataset contents
    res = []
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        logger.info("item: {}".format(item))
        res.append(item)
    logger.info("dataset: {}".format(res))

    np.testing.assert_array_equal(res, [[3]])


def test_numpyslices_sampler_chain2():
    """
    Feature: Chained Sampler
    Description: NumpySlicesDataset with sampler chain; add child sampler with 2 statements
    Expectation: Data verified to be correct
    """
    logger.info("test_numpyslices_sampler_chain2")

    # Create NumpySlicesDataset with sampler chain
    # Use 2 statements to add child sampler
    np_data = [1, 2, 3, 4]
    sampler = ds.SequentialSampler(start_index=1, num_samples=1)
    child_sampler = ds.SequentialSampler(start_index=1, num_samples=2)
    sampler.add_child(child_sampler)
    data1 = ds.NumpySlicesDataset(np_data, sampler=sampler)

    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 1

    # Verify number of rows
    assert sum((1 for _ in data1)) == 1

    # Verify dataset contents
    res = []
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        logger.info("item: {}".format(item))
        res.append(item)
    logger.info("dataset: {}".format(res))

    np.testing.assert_array_equal(res, [[3]])


def test_numpyslices_sampler_chain_multi_add_child():
    """
    Feature: Chained Sampler
    Description: NumpySlicesDataset with sampler chain with multiple add_child() invocations
    Expectation: Data verified to be correct. A subsequent add_child() invocation replaces the prior
    child sampler (if any).
    """
    logger.info("test_numpyslices_sampler_chain_multi_add_child")

    # Create NumpySlicesDataset with sampler chain
    # Call add_child() multiple times in succession
    np_data = [1, 2, 3, 4, 5, 6, 7, 8]
    sampler = ds.SequentialSampler(start_index=1, num_samples=None)
    sampler.add_child(ds.SequentialSampler(start_index=1, num_samples=6))
    # Expect the second child will fail
    with pytest.raises(RuntimeError) as info:
        sampler.add_child(ds.SequentialSampler(start_index=4, num_samples=2))

    error_msg = "Cannot add child sampler, this sampler already has a child."
    assert error_msg in str(info.value)

    data1 = ds.NumpySlicesDataset(np_data, sampler=sampler)

    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 5

    # Verify number of rows
    assert sum((1 for _ in data1)) == 5

    # Verify dataset contents
    res = []
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        logger.info("item: {}".format(item))
        res.append(item)
    logger.info("dataset: {}".format(res))

    np.testing.assert_array_equal(res, [[3], [4], [5], [6], [7]])




def test_mnist_sampler_chain():
    """
    Feature: Chained Sampler
    Description: MnistDataset with sampler chain; add child sampler with 2 statements
    Expectation: Data verified to be correct
    """
    logger.info("test_mnist_sampler_chain")

    sampler = ds.DistributedSampler(num_shards=1, shard_id=0, shuffle=False, num_samples=3, offset=1)
    child_sampler = ds.RandomSampler(replacement=True, num_samples=4)
    sampler.add_child(child_sampler)
    data1 = ds.MnistDataset(MNIST_DATA_DIR, sampler=sampler)

    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 3
    # Verify number of rows
    assert sum((1 for _ in data1)) == 3

    # Verify dataset contents
    res = []
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        logger.info("item: {}".format(item))
        res.append(item)
    logger.info("dataset: {}".format(res))


def test_manifest_sampler_chain():
    """
    Feature: Chained Sampler
    Description: ManifestDataset with sampler chain; add child sampler with 2 statements
    Expectation: Data verified to be correct
    """
    logger.info("test_manifest_sampler_chain")

    sampler = ds.RandomSampler(replacement=True, num_samples=2)
    child_sampler = ds.DistributedSampler(num_shards=1, shard_id=0, shuffle=False, num_samples=3, offset=1)
    sampler.add_child(child_sampler)
    data1 = ds.ManifestDataset(MANIFEST_DATA_FILE, sampler=sampler)

    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 2
    # Verify number of rows
    assert sum((1 for _ in data1)) == 2

    # Verify dataset contents
    res = []
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        logger.info("item: {}".format(item))
        res.append(item)
    logger.info("dataset: {}".format(res))


def test_coco_sampler_chain():
    """
    Feature: Chained Sampler
    Description: CocoDataset with sampler chain; add child sampler with 2 statements
    Expectation: Data verified to be correct
    """
    logger.info("test_coco_sampler_chain")

    sampler = ds.DistributedSampler(num_shards=2, shard_id=0, shuffle=False, num_samples=5)
    child_sampler = ds.RandomSampler(replacement=True, num_samples=2)
    sampler.add_child(child_sampler)
    data1 = ds.CocoDataset(COCO_DATA_DIR, annotation_file=ANNOTATION_FILE, task="Detection", decode=True,
                           sampler=sampler)

    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 1

    # Verify number of rows
    assert sum((1 for _ in data1)) == 1

    # Verify dataset contents
    res = []
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        logger.info("item: {}".format(item))
        res.append(item)
    logger.info("dataset: {}".format(res))


def test_cifar_sampler_chain():
    """
    Feature: Chained Sampler
    Description: CifarDataset with sampler chain, including nested child sampler
    Expectation: Data verified to be correct
    """
    logger.info("test_cifar_sampler_chain")

    sampler = ds.DistributedSampler(num_shards=2, shard_id=0, shuffle=False, num_samples=5)
    child_sampler = ds.RandomSampler(replacement=True, num_samples=4)
    child_sampler2 = ds.SequentialSampler(start_index=0, num_samples=2)
    # Note: Add nested child_sampler2 to child_sampler
    child_sampler.add_child(child_sampler2)
    sampler.add_child(child_sampler)
    data1 = ds.Cifar10Dataset(CIFAR10_DATA_DIR, sampler=sampler)
    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 1

    # Verify number of rows
    assert sum((1 for _ in data1)) == 1

    # Verify dataset contents
    res = []
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        logger.info("item: {}".format(item))
        res.append(item)
    logger.info("dataset: {}".format(res))


def test_voc_sampler_chain():
    """
    Feature: Chained Sampler
    Description: VOCDataset with sampler chain; add child sampler with 2 statements
    Expectation: Data verified to be correct
    """
    logger.info("test_voc_sampler_chain")

    sampler = ds.DistributedSampler(num_shards=2, shard_id=0, shuffle=False, num_samples=5)
    child_sampler = ds.SequentialSampler(start_index=0)
    sampler.add_child(child_sampler)
    data1 = ds.VOCDataset(VOC_DATA_DIR, task="Segmentation", sampler=sampler)

    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 5

    # Verify number of rows
    assert sum((1 for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True))) == 5

    # Verify dataset contents
    res = []
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        logger.info("item: {}".format(item))
        res.append(item)
    logger.info("dataset: {}".format(res))


def test_numpyslices_sampler_chain_batch():
    """
    Feature: Chained Sampler
    Description: NumpySlicesDataset with sampler chain with batch
    Expectation: Data verified to be correct
    """
    logger.info("test_numpyslices_sampler_chain_batch")

    # Create NumpySlicesDataset with sampler chain
    np_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sampler = ds.SequentialSampler(start_index=1, num_samples=8)
    sampler.add_child(ds.SequentialSampler(start_index=1, num_samples=9))
    data1 = ds.NumpySlicesDataset(np_data, sampler=sampler)
    data1 = data1.batch(batch_size=2, drop_remainder=False)

    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 4

    # Verify number of rows
    assert sum((1 for _ in data1)) == 4

    # Verify dataset contents
    res = []
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        logger.info("item: {}".format(item))
        res.append(item)
    logger.info("dataset: {}".format(res))

    np.testing.assert_array_equal(res, [[[3, 4]], [[5, 6]], [[7, 8]], [[9, 10]]])


def test_sampler_chain_errors():
    """
    Feature: Chained Sampler
    Description: Test error cases for sampler chains
    Expectation: Correct error is raised as expected
    """
    logger.info("test_sampler_chain_errors")

    conflict_error = "Conflicting arguments during sampler assignments."

    # Test conflicting arguments (sampler and shuffle=False) for sampler (no chain)
    np_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sampler = ds.SequentialSampler(start_index=1, num_samples=3)
    with pytest.raises(ValueError, match=conflict_error):
        ds.NumpySlicesDataset(np_data, shuffle=False, sampler=sampler)

    # Test conflicting arguments (sampler and shuffle=False) for sampler chaining
    np_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sampler = ds.SequentialSampler(start_index=1, num_samples=3)
    sampler.add_child(ds.SequentialSampler(start_index=1, num_samples=2))
    with pytest.raises(ValueError, match=conflict_error):
        ds.NumpySlicesDataset(np_data, shuffle=False, sampler=sampler)


def test_manifest_sampler_chain_repeat():
    """
    Feature: Chained Sampler
    Description: Test ManifestDataset sampler chain DistributedSampler -> SequentialSampler with repeat
    Expectation: Data verified to be correct
    """
    logger.info("test_manifest_sampler_chain_batch")
    manifest_file = "../data/dataset/testManifestData/test5trainimgs.json"

    # Create sampler chain DistributedSampler->SequentialSampler
    sampler = ds.DistributedSampler(num_shards=1, shard_id=0, shuffle=False, num_samples=5)
    child_sampler = ds.SequentialSampler()
    sampler.add_child(child_sampler)

    # Create ManifestDataset with sampler chain
    data1 = ds.ManifestDataset(manifest_file, sampler=sampler)
    data1 = data1.repeat(count=2)

    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 10

    # Verify number of rows
    assert sum((1 for _ in data1)) == 10

    # Verify dataset contents
    filename = "sampler_chain_manifest_repeat_result.npz"
    save_and_check_md5(data1, filename, generate_golden=GENERATE_GOLDEN)




def test_add_user_defined_sampler_dataset_size():
    """
    Feature: add_sampler
    Description: Test using add_sampler to add a user defined sampler
    Expectation: The dataset size after sampling is correct
    """

    class MySampler(ds.Sampler):
        def __iter__(self):
            if self.num_samples % 2 == 0:
                interval = 2
            elif self.num_samples % 3 == 0:
                interval = 3
            else:
                interval = 1
            yield from range(0, self.num_samples, interval)

    data = np.random.randint(0, 255, (100, 28, 28, 1))
    dataset = ds.NumpySlicesDataset(data, column_names=["data"], shuffle=True)
    assert dataset.get_dataset_size() == 100

    distributed_sampler = ds.DistributedSampler(num_shards=4, shard_id=0)
    dataset.add_sampler(distributed_sampler)
    assert dataset.get_dataset_size() == 25

    sequential_sampler = ds.SequentialSampler(0, num_samples=15)
    dataset.add_sampler(sequential_sampler)
    assert dataset.get_dataset_size() == 15

    my_sampler = ds.IterSampler(MySampler(dataset.get_dataset_size()))
    dataset.add_sampler(my_sampler)
    assert dataset.get_dataset_size() == 5


def test_add_user_defined_sampler_result():
    """
    Feature: add_sampler
    Description: Test using add_sampler to add a user defined sampler
    Expectation: The result after sampling is correct
    """

    class MySampler(ds.Sampler):
        def __iter__(self):
            yield from range(0, self.num_samples, 2)

    data = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"]
    dataset = ds.NumpySlicesDataset(data, column_names=["data"], shuffle=False)
    first_sampler = ds.IterSampler(MySampler(dataset.get_dataset_size()))
    dataset.add_sampler(first_sampler)
    second_sampler = ds.IterSampler(MySampler(dataset.get_dataset_size()))
    dataset.add_sampler(second_sampler)

    expected_result = np.array(["a", "e", "i", "m"])
    for i, item in enumerate(dataset.create_dict_iterator(num_epochs=1, output_numpy=True)):
        assert item["data"] == expected_result[i]


class StrNumberDataset:
    def __init__(self):
        self.data = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def test_generator_sampler_chain_of_sequential_and_user_defined():
    """
    Feature: Chain sampler
    Description: Test adding SequentialSampler on user defined sampler for GeneratorDataset
    Expectation: The result of dataset is as expected
    """

    class UDFSampler(ds.Sampler):
        def __iter__(self):
            yield from range(5, 10)

    # Sequential sampler generates indices from 0 to 9, then user defined sampler selects
    # from the 5th to 9th of them
    sequential_sampler = ds.SequentialSampler(num_samples=10)
    user_defined_sampler = UDFSampler()
    user_defined_sampler.add_child(sequential_sampler)
    dataset = ds.GeneratorDataset(StrNumberDataset(), column_names=["data"], sampler=user_defined_sampler)
    assert dataset.get_dataset_size() == 5
    expected_result = np.array(['5', '6', '7', '8', '9'])
    for index, item in enumerate(dataset.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(item["data"], expected_result[index])


def test_generator_sampler_chain_of_user_defined_and_random():
    """
    Feature: Chain sampler
    Description: Test adding user defined sampler on RandomSampler for GeneratorDataset
    Expectation: The result of dataset is as expected
    """

    original_seed = ds.config.get_seed()
    ds.config.set_seed(0)

    class UDFSampler(ds.Sampler):
        def __iter__(self):
            yield from range(5, 10)

    # User defined sampler generates indices from 5 to 9, then random sampler shuffles them
    user_defined_sampler = UDFSampler()
    random_sampler = ds.RandomSampler()
    random_sampler.add_child(user_defined_sampler)
    dataset = ds.GeneratorDataset(StrNumberDataset(), column_names=["data"], sampler=random_sampler)
    assert dataset.get_dataset_size() == 5
    expected_result = np.array(['7', '9', '8', '5', '6'])
    try:
        for index, item in enumerate(dataset.create_dict_iterator(num_epochs=1, output_numpy=True)):
            np.testing.assert_array_equal(item["data"], expected_result[index])
    finally:
        # Use finally to ensure the seed can be restored even if the test fails
        ds.config.set_seed(original_seed)


def test_generator_sampler_chain_of_user_defined_and_user_defined():
    """
    Feature: Chain sampler
    Description: Test adding user defined sampler on user defined sampler for GeneratorDataset
    Expectation: The result of dataset is as expected
    """

    class UDFJumpSampler(ds.Sampler):
        def __iter__(self):
            yield from range(0, 10, 2)

    class UDFReverseSampler(ds.Sampler):
        def __iter__(self):
            yield from range(4, -1, -1)

    # UDFJumpSampler generates indices from 0 to 9 with step 2, then UDFReverseSampler reverses them
    udf_jump_sampler = UDFJumpSampler()
    udf_reverse_sampler = UDFReverseSampler()
    udf_reverse_sampler.add_child(udf_jump_sampler)
    dataset = ds.GeneratorDataset(StrNumberDataset(), column_names=["data"], sampler=udf_reverse_sampler)
    assert dataset.get_dataset_size() == 5
    expected_result = np.array(['8', '6', '4', '2', '0'])
    for index, item in enumerate(dataset.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(item["data"], expected_result[index])
