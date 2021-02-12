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
import pytest
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as c_transforms
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
    Test NumpySlicesDataset with sampler, no chain
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
    assert sum([1 for _ in data1]) == 2

    # Verify dataset contents
    res = []
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        logger.info("item: {}".format(item))
        res.append(item)
    logger.info("dataset: {}".format(res))


def test_numpyslices_sampler_chain():
    """
    Test NumpySlicesDataset sampler chain
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
    assert sum([1 for _ in data1]) == 1

    # Verify dataset contents
    res = []
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        logger.info("item: {}".format(item))
        res.append(item)
    logger.info("dataset: {}".format(res))


def test_numpyslices_sampler_chain2():
    """
    Test NumpySlicesDataset sampler chain
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
    assert sum([1 for _ in data1]) == 1

    # Verify dataset contents
    res = []
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        logger.info("item: {}".format(item))
        res.append(item)
    logger.info("dataset: {}".format(res))


def test_imagefolder_sampler_chain():
    """
    Test ImageFolderDataset sampler chain
    """
    logger.info("test_imagefolder_sampler_chain")

    sampler = ds.SequentialSampler(start_index=1, num_samples=3)
    child_sampler = ds.PKSampler(2)
    sampler.add_child(child_sampler)
    data1 = ds.ImageFolderDataset(IMAGENET_RAWDATA_DIR, sampler=sampler)
    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 3
    # Verify number of rows
    assert sum([1 for _ in data1]) == 3

    # Verify dataset contents
    res = []
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        logger.info("item: {}".format(item))
        res.append(item)
    logger.info("dataset: {}".format(res))


def test_mnist_sampler_chain():
    """
    Test Mnist sampler chain
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
    assert sum([1 for _ in data1]) == 3

    # Verify dataset contents
    res = []
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        logger.info("item: {}".format(item))
        res.append(item)
    logger.info("dataset: {}".format(res))


def test_manifest_sampler_chain():
    """
    Test Manifest sampler chain
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
    assert sum([1 for _ in data1]) == 2

    # Verify dataset contents
    res = []
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        logger.info("item: {}".format(item))
        res.append(item)
    logger.info("dataset: {}".format(res))


def test_coco_sampler_chain():
    """
    Test Coco sampler chain
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
    assert sum([1 for _ in data1]) == 1

    # Verify dataset contents
    res = []
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        logger.info("item: {}".format(item))
        res.append(item)
    logger.info("dataset: {}".format(res))


def test_cifar_sampler_chain():
    """
    Test Cifar sampler chain
    """
    logger.info("test_cifar_sampler_chain")

    sampler = ds.DistributedSampler(num_shards=2, shard_id=0, shuffle=False, num_samples=5)
    child_sampler = ds.RandomSampler(replacement=True, num_samples=4)
    child_sampler2 = ds.SequentialSampler(start_index=0, num_samples=2)
    child_sampler.add_child(child_sampler2)
    sampler.add_child(child_sampler)
    data1 = ds.Cifar10Dataset(CIFAR10_DATA_DIR, sampler=sampler)
    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 1

    # Verify number of rows
    assert sum([1 for _ in data1]) == 1

    # Verify dataset contents
    res = []
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        logger.info("item: {}".format(item))
        res.append(item)
    logger.info("dataset: {}".format(res))


def test_voc_sampler_chain():
    """
    Test VOC sampler chain
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
    assert sum([1 for _ in data1]) == 5

    # Verify dataset contents
    res = []
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        logger.info("item: {}".format(item))
        res.append(item)
    logger.info("dataset: {}".format(res))


def test_numpyslices_sampler_chain_batch():
    """
    Test NumpySlicesDataset sampler chaining, with batch
    """
    logger.info("test_numpyslices_sampler_chain_batch")

    # Create NumpySlicesDataset with sampler chain
    np_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sampler = ds.SequentialSampler(start_index=1, num_samples=3)
    sampler = sampler.add_child(ds.SequentialSampler(start_index=1, num_samples=2))
    data1 = ds.NumpySlicesDataset(np_data, sampler=sampler)
    data1 = data1.batch(batch_size=3, drop_remainder=False)

    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 4

    # Verify number of rows
    assert sum([1 for _ in data1]) == 4

    # Verify dataset contents
    res = []
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        logger.info("item: {}".format(item))
        res.append(item)
    logger.info("dataset: {}".format(res))


def test_sampler_chain_errors():
    """
    Test error cases for sampler chains
    """
    logger.info("test_sampler_chain_errors")

    error_msg_1 = "'NoneType' object has no attribute 'add_child'"
    # Test add child sampler within child sampler
    sampler = ds.SequentialSampler(start_index=1, num_samples=2)
    sampler = sampler.add_child(ds.SequentialSampler(start_index=1, num_samples=2))
    with pytest.raises(AttributeError, match=error_msg_1):
        sampler.add_child(ds.SequentialSampler(start_index=1, num_samples=2))

    # error_msg_2 = "'NoneType' object has no attribute 'add_child'"
    # Test add second and nested child sampler
    sampler = ds.SequentialSampler(start_index=1, num_samples=2)
    child_sampler = ds.SequentialSampler(start_index=1, num_samples=2)
    sampler.add_child(child_sampler)
    child_sampler2 = ds.SequentialSampler(start_index=1, num_samples=2)
    sampler.add_child(child_sampler2)
    # FIXME - no error is raised; uncomment after code issue is resolved
    # with pytest.raises(AttributeError, match=error_msg_2):
    #     sampler.add_child(child_sampler2)
    #     np_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #     data1 = ds.NumpySlicesDataset(np_data, sampler=sampler)

    error_msg_3 = "Conflicting arguments during sampler assignments."
    # Test conflicting arguments (sampler and shuffle=False) for sampler (no chain)
    np_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sampler = ds.SequentialSampler(start_index=1, num_samples=3)
    with pytest.raises(ValueError, match=error_msg_3):
        ds.NumpySlicesDataset(np_data, shuffle=False, sampler=sampler)

    # error_msg_4 = "Conflicting arguments during sampler assignments."
    # Test conflicting arguments (sampler and shuffle=False) for sampler chaining
    np_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sampler = ds.SequentialSampler(start_index=1, num_samples=3)
    sampler = sampler.add_child(ds.SequentialSampler(start_index=1, num_samples=2))
    # FIXME - no error is raised; uncomment after code issue is resolved
    # with pytest.raises(ValueError, match=error_msg_4):
    #     ds.NumpySlicesDataset(np_data, shuffle=False, sampler=sampler)


def test_manifest_sampler_chain_repeat():
    """
    Test ManifestDataset sampler chain DistributedSampler->SequentialSampler, with repeat
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
    assert sum([1 for _ in data1]) == 10

    # Verify dataset contents
    filename = "sampler_chain_manifest_repeat_result.npz"
    save_and_check_md5(data1, filename, generate_golden=GENERATE_GOLDEN)


def test_manifest_sampler_chain_batch_repeat():
    """
    Test ManifestDataset sampler chain DistributedSampler->SequentialSampler, with batch then repeat
    """
    logger.info("test_manifest_sampler_chain_batch_repeat")
    manifest_file = "../data/dataset/testManifestData/test5trainimgs.json"

    # Create sampler chain DistributedSampler->SequentialSampler
    sampler = ds.DistributedSampler(num_shards=1, shard_id=0, shuffle=False, num_samples=5)
    child_sampler = ds.SequentialSampler()
    sampler.add_child(child_sampler)

    # Create ManifestDataset with sampler chain
    data1 = ds.ManifestDataset(manifest_file, decode=True, sampler=sampler)
    one_hot_encode = c_transforms.OneHot(3)
    data1 = data1.map(operations=one_hot_encode, input_columns=["label"])
    data1 = data1.batch(batch_size=5, drop_remainder=False)
    data1 = data1.repeat(count=2)

    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 2

    # Verify number of rows
    # FIXME: Uncomment the following assert when code issue is resolved
    # assert sum([1 for _ in data1]) == 2


if __name__ == '__main__':
    test_numpyslices_sampler_no_chain()
    test_numpyslices_sampler_chain()
    test_numpyslices_sampler_chain2()
    test_imagefolder_sampler_chain()
    test_mnist_sampler_chain()
    test_manifest_sampler_chain()
    test_coco_sampler_chain()
    test_cifar_sampler_chain()
    test_voc_sampler_chain()
    test_numpyslices_sampler_chain_batch()
    test_sampler_chain_errors()
    test_manifest_sampler_chain_repeat()
    test_manifest_sampler_chain_batch_repeat()
