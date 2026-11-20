# Copyright 2019-2022 Huawei Technologies Co., Ltd
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
Test Cifar10 and Cifar100 dataset operations
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import mindspore.dataset as ds
from mindspore import log as logger
from mindspore.dataset import vision
from mindspore.dataset.vision import Border, Inter
from mindspore.dataset.callback import DSCallback

DATA_DIR_10 = "../data/dataset/testCifar10Data"
DATA_DIR_100 = "../data/dataset/testCifar100Data"
NO_BIN_DIR = "../data/dataset/testMnistData"
DATA_FILE = "../data/dataset/testTextFileDataset/1.txt"


def load_cifar(path, kind="cifar10"):
    """
    load Cifar10/100 data
    """
    raw = np.empty(0, dtype=np.uint8)
    for file_name in os.listdir(path):
        if file_name.endswith(".bin"):
            with open(os.path.join(path, file_name), mode='rb') as file:
                raw = np.append(raw, np.fromfile(file, dtype=np.uint8), axis=0)
    if kind == "cifar10":
        raw = raw.reshape(-1, 3073)
        labels = raw[:, 0]
        images = raw[:, 1:]
    elif kind == "cifar100":
        raw = raw.reshape(-1, 3074)
        labels = raw[:, :2]
        images = raw[:, 2:]
    else:
        raise ValueError("Invalid parameter value")
    images = images.reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)
    return images, labels


def visualize_dataset(images, labels):
    """
    Helper function to visualize the dataset samples
    """
    num_samples = len(images)
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i])
        plt.title(labels[i])
    plt.show()


class UserPyOp:
    '''userpyop'''

    def __init__(self):
        self.ep_num = 0
        self.step_num = 0

    def __call__(self, x):
        return np.array(x + self.step_num ** self.ep_num - 1)

    def update(self, ep_num, step_num):
        self.ep_num = ep_num
        self.step_num = step_num


class UserCallback(DSCallback):
    def __init__(self, py_op, step_size=1):
        super().__init__(step_size)
        self.py_op = py_op

    def ds_step_begin(self, ds_run_context):
        ep_num = ds_run_context.cur_epoch_num
        step_num = ds_run_context.cur_step_num
        self.py_op.update(ep_num, step_num)


def apply_func(dataset):
    '''apply'''
    rescale = 2.0
    shift = 1.0
    meanr = 0.5
    meang = 115.0
    meanb = 100.0
    stdr = 70.0
    stdg = 68.0
    stdb = 71.0

    random_horizon = vision.RandomHorizontalFlip()
    dataset = dataset.map(input_columns=["image"], operations=random_horizon, num_parallel_workers=3)

    random_vertical = vision.RandomVerticalFlip()
    dataset = dataset.map(input_columns=["image"], operations=random_vertical, num_parallel_workers=3)

    rescale_op = vision.Rescale(rescale, shift)
    dataset = dataset.map(input_columns=["image"], operations=rescale_op, num_parallel_workers=3)

    normalize_op = vision.Normalize((meanr, meang, meanb), (stdr, stdg, stdb))
    dataset = dataset.map(input_columns=["image"], operations=normalize_op, num_parallel_workers=3)

    return dataset


def add_one_by_batch_num(batch_info):
    return batch_info.get_batch_num() + 1


def add_one_by_epoch(batch_info):
    return batch_info.get_epoch_num() + 1


def invert_sign_per_batch_multi_col(col_list, batch_info):
    return ([np.copy(((-1) ** batch_info.get_batch_num()) * arr) for arr in col_list],)


class DatasetCTransformsFunc():
    '''c_transform'''

    def batch_map_fun(self, sampler):
        '''Batch map with aligned labels.'''
        skipcount = 5
        takecount = 5
        repeatcount = 10
        l1 = []

        dataset = ds.Cifar10Dataset(DATA_DIR_10, sampler=sampler, num_parallel_workers=3)
        beforesize = 0
        for data in dataset.create_dict_iterator(output_numpy=True):
            l1.append(data['image'])
            beforesize += 1
        l1.clear()

        dataset = dataset.skip(count=skipcount)
        aftersize = 0
        for _ in dataset.create_dict_iterator(output_numpy=True):
            aftersize += 1
        assert (beforesize - aftersize) == skipcount

        num_classes = 100
        def encode_label(label):
            encoded = np.expand_dims(np.asarray(label).squeeze(), -1) == np.arange(num_classes)
            return encoded.astype(label.dtype).squeeze()

        one_hot_encode = encode_label
        dataset = dataset.map(input_columns="label", operations=one_hot_encode, num_parallel_workers=3)

        dataset = dataset.batch(2, drop_remainder=True)
        dataset = dataset.map(input_columns=["image", "label"],
                              operations=lambda image, label: (image.astype(np.float32), label))

        dataset = dataset.take(count=takecount)
        aftersize = 0
        for data in dataset.create_dict_iterator(output_numpy=True):
            aftersize += 1
        assert aftersize == takecount

        dataset = dataset.shuffle(2)

        dataset = dataset.batch(batch_size=add_one_by_epoch, drop_remainder=True, num_parallel_workers=3,
                                input_columns=["image"], per_batch_map=invert_sign_per_batch_multi_col)

        dataset = dataset.repeat(repeatcount)
        for data in dataset.create_dict_iterator(output_numpy=True):
            l1.append(data['image'])
        l1.clear()


### Testcases for Cifar10Dataset Op ###

def test_cifar10_content_check():
    """
    Feature: Cifar10Dataset
    Description: Test Cifar10Dataset with content check on image readings
    Expectation: The dataset is processed as expected
    """
    logger.info("Test Cifar10Dataset Op with content check")
    data1 = ds.Cifar10Dataset(DATA_DIR_10, num_samples=100, shuffle=False)
    images, labels = load_cifar(DATA_DIR_10)
    num_iter = 0
    # in this example, each dictionary has keys "image" and "label"
    for i, d in enumerate(data1.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(d["image"], images[i])
        np.testing.assert_array_equal(d["label"], labels[i])
        num_iter += 1
    assert num_iter == 100


def test_cifar10_basic():
    """
    Feature: Cifar10Dataset
    Description: Test Cifar10Dataset with some basic arguments and methods
    Expectation: The dataset is processed as expected
    """
    logger.info("Test Cifar10Dataset Op")

    # case 0: test loading the whole dataset
    data0 = ds.Cifar10Dataset(DATA_DIR_10)
    num_iter0 = 0
    for _ in data0.create_dict_iterator(num_epochs=1):
        num_iter0 += 1
    assert num_iter0 == 10000

    # case 1: test num_samples
    data1 = ds.Cifar10Dataset(DATA_DIR_10, num_samples=100)
    num_iter1 = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_iter1 += 1
    assert num_iter1 == 100

    # case 2: test num_parallel_workers
    data2 = ds.Cifar10Dataset(DATA_DIR_10, num_samples=50, num_parallel_workers=1)
    num_iter2 = 0
    for _ in data2.create_dict_iterator(num_epochs=1):
        num_iter2 += 1
    assert num_iter2 == 50

    # case 3: test repeat
    data3 = ds.Cifar10Dataset(DATA_DIR_10, num_samples=100)
    data3 = data3.repeat(3)
    num_iter3 = 0
    for _ in data3.create_dict_iterator(num_epochs=1):
        num_iter3 += 1
    assert num_iter3 == 300

    # case 4: test batch with drop_remainder=False
    data4 = ds.Cifar10Dataset(DATA_DIR_10, num_samples=100)
    assert data4.get_dataset_size() == 100
    assert data4.get_batch_size() == 1
    data4 = data4.batch(batch_size=7)  # drop_remainder is default to be False
    assert data4.get_dataset_size() == 15
    assert data4.get_batch_size() == 7
    num_iter4 = 0
    for _ in data4.create_dict_iterator(num_epochs=1):
        num_iter4 += 1
    assert num_iter4 == 15

    # case 5: test batch with drop_remainder=True
    data5 = ds.Cifar10Dataset(DATA_DIR_10, num_samples=100)
    assert data5.get_dataset_size() == 100
    assert data5.get_batch_size() == 1
    data5 = data5.batch(batch_size=7, drop_remainder=True)  # the rest of incomplete batch will be dropped
    assert data5.get_dataset_size() == 14
    assert data5.get_batch_size() == 7
    num_iter5 = 0
    for _ in data5.create_dict_iterator(num_epochs=1):
        num_iter5 += 1
    assert num_iter5 == 14


def test_cifar10_visualize(plot=False):
    """
    Feature: Cifar10Dataset
    Description: Test Cifar10Dataset visualization results
    Expectation: Results are presented as expected
    """
    logger.info("Test Cifar10Dataset visualization")

    data1 = ds.Cifar10Dataset(DATA_DIR_10, num_samples=10, shuffle=False)
    num_iter = 0
    image_list, label_list = [], []
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = item["image"]
        label = item["label"]
        image_list.append(image)
        label_list.append("label {}".format(label))
        assert isinstance(image, np.ndarray)
        assert image.shape == (32, 32, 3)
        assert image.dtype == np.uint8
        assert label.dtype == np.uint32
        num_iter += 1
    assert num_iter == 10
    if plot:
        visualize_dataset(image_list, label_list)


### Testcases for Cifar100Dataset Op ###

def test_cifar100_content_check():
    """
    Feature: Cifar100Dataset
    Description: Test Cifar100Dataset image readings with content check
    Expectation: The dataset is processed as expected
    """
    logger.info("Test Cifar100Dataset with content check")
    data1 = ds.Cifar100Dataset(DATA_DIR_100, num_samples=100, shuffle=False)
    images, labels = load_cifar(DATA_DIR_100, kind="cifar100")
    num_iter = 0
    # in this example, each dictionary has keys "image", "coarse_label" and "fine_image"
    for i, d in enumerate(data1.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(d["image"], images[i])
        np.testing.assert_array_equal(d["coarse_label"], labels[i][0])
        np.testing.assert_array_equal(d["fine_label"], labels[i][1])
        num_iter += 1
    assert num_iter == 100


def test_cifar100_basic():
    """
    Feature: Cifar100Dataset
    Description: Test Cifar100Dataset basic arguments and features
    Expectation: The dataset is processed as expected
    """
    logger.info("Test Cifar100Dataset")

    # case 1: test num_samples
    data1 = ds.Cifar100Dataset(DATA_DIR_100, num_samples=100)
    num_iter1 = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_iter1 += 1
    assert num_iter1 == 100

    # case 2: test repeat
    data1 = data1.repeat(2)
    num_iter2 = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_iter2 += 1
    assert num_iter2 == 200

    # case 3: test num_parallel_workers
    data2 = ds.Cifar100Dataset(DATA_DIR_100, num_samples=100, num_parallel_workers=1)
    num_iter3 = 0
    for _ in data2.create_dict_iterator(num_epochs=1):
        num_iter3 += 1
    assert num_iter3 == 100

    # case 4: test batch with drop_remainder=False
    data3 = ds.Cifar100Dataset(DATA_DIR_100, num_samples=100)
    assert data3.get_dataset_size() == 100
    assert data3.get_batch_size() == 1
    data3 = data3.batch(batch_size=3)
    assert data3.get_dataset_size() == 34
    assert data3.get_batch_size() == 3
    num_iter4 = 0
    for _ in data3.create_dict_iterator(num_epochs=1):
        num_iter4 += 1
    assert num_iter4 == 34

    # case 4: test batch with drop_remainder=True
    data4 = ds.Cifar100Dataset(DATA_DIR_100, num_samples=100)
    data4 = data4.batch(batch_size=3, drop_remainder=True)
    assert data4.get_dataset_size() == 33
    assert data4.get_batch_size() == 3
    num_iter5 = 0
    for _ in data4.create_dict_iterator(num_epochs=1):
        num_iter5 += 1
    assert num_iter5 == 33


def test_cifar100_visualize(plot=False):
    """
    Feature: Cifar100Dataset
    Description: Test Cifar100Dataset visualization results
    Expectation: Results are presented as expected
    """
    logger.info("Test Cifar100Dataset visualization")

    data1 = ds.Cifar100Dataset(DATA_DIR_100, num_samples=10, shuffle=False)
    num_iter = 0
    image_list, label_list = [], []
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = item["image"]
        coarse_label = item["coarse_label"]
        fine_label = item["fine_label"]
        image_list.append(image)
        label_list.append("coarse_label {}\nfine_label {}".format(coarse_label, fine_label))
        assert isinstance(image, np.ndarray)
        assert image.shape == (32, 32, 3)
        assert image.dtype == np.uint8
        assert coarse_label.dtype == np.uint32
        assert fine_label.dtype == np.uint32
        num_iter += 1
    assert num_iter == 10
    if plot:
        visualize_dataset(image_list, label_list)


def test_cifar_usage():
    """
    Feature: Cifar100Dataset
    Description: Test Cifar100Dataset usage flag
    Expectation: The dataset is processed as expected
    """
    logger.info("Test Cifar100Dataset usage flag")

    # flag, if True, test cifar10 else test cifar100
    def test_config(usage, flag=True, cifar_path=None):
        if cifar_path is None:
            cifar_path = DATA_DIR_10 if flag else DATA_DIR_100
        try:
            data = ds.Cifar10Dataset(cifar_path, usage=usage) if flag else ds.Cifar100Dataset(cifar_path, usage=usage)
            num_rows = 0
            for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                num_rows += 1
        except (ValueError, TypeError, RuntimeError) as e:
            return str(e)
        return num_rows

    # test the usage of CIFAR100
    assert test_config("train") == 10000
    assert test_config("all") == 10000
    assert "usage is not within the valid set of ['train', 'test', 'all']" in test_config("invalid")
    assert "Argument usage with value ['list'] is not of type [<class 'str'>]" in test_config(["list"])
    assert "Cifar10Dataset API can't read the data file (interface mismatch or no data found)" in test_config("test")

    # test the usage of CIFAR10
    assert test_config("test", False) == 10000
    assert test_config("all", False) == 10000
    assert "Cifar100Dataset API can't read the data file" in test_config("train", False)
    assert "usage is not within the valid set of ['train', 'test', 'all']" in test_config("invalid", False)

    # change this directory to the folder that contains all cifar10 files
    all_cifar10 = None
    if all_cifar10 is not None:
        assert test_config("train", True, all_cifar10) == 50000
        assert test_config("test", True, all_cifar10) == 10000
        assert test_config("all", True, all_cifar10) == 60000
        assert ds.Cifar10Dataset(all_cifar10, usage="train").get_dataset_size() == 50000
        assert ds.Cifar10Dataset(all_cifar10, usage="test").get_dataset_size() == 10000
        assert ds.Cifar10Dataset(all_cifar10, usage="all").get_dataset_size() == 60000

    # change this directory to the folder that contains all cifar100 files
    all_cifar100 = None
    if all_cifar100 is not None:
        assert test_config("train", False, all_cifar100) == 50000
        assert test_config("test", False, all_cifar100) == 10000
        assert test_config("all", False, all_cifar100) == 60000
        assert ds.Cifar100Dataset(all_cifar100, usage="train").get_dataset_size() == 50000
        assert ds.Cifar100Dataset(all_cifar100, usage="test").get_dataset_size() == 10000
        assert ds.Cifar100Dataset(all_cifar100, usage="all").get_dataset_size() == 60000


def test_cifar_exception_file_path():
    """
    Feature: CifarDataset
    Description: Test Cifar10Dataset and Cifar100Dataset with invalid file path
    Expectation: Error is raised as expected
    """
    def exception_func(item):
        raise Exception("Error occur!")

    try:
        data = ds.Cifar10Dataset(DATA_DIR_10)
        data = data.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        num_rows = 0
        for _ in data.create_dict_iterator(num_epochs=1):
            num_rows += 1
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    try:
        data = ds.Cifar10Dataset(DATA_DIR_10)
        data = data.map(operations=exception_func, input_columns=["label"], num_parallel_workers=1)
        num_rows = 0
        for _ in data.create_dict_iterator(num_epochs=1):
            num_rows += 1
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    try:
        data = ds.Cifar100Dataset(DATA_DIR_100)
        data = data.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        num_rows = 0
        for _ in data.create_dict_iterator(num_epochs=1):
            num_rows += 1
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    try:
        data = ds.Cifar100Dataset(DATA_DIR_100)
        data = data.map(operations=exception_func, input_columns=["coarse_label"], num_parallel_workers=1)
        num_rows = 0
        for _ in data.create_dict_iterator(num_epochs=1):
            num_rows += 1
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    try:
        data = ds.Cifar100Dataset(DATA_DIR_100)
        data = data.map(operations=exception_func, input_columns=["fine_label"], num_parallel_workers=1)
        num_rows = 0
        for _ in data.create_dict_iterator(num_epochs=1):
            num_rows += 1
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)


def dataset_call_py_transforms_func(sampler, sampler_num):
    """
    All of py_transforms
    Returns:

    """

    dataset = ds.Cifar10Dataset(DATA_DIR_10, sampler=sampler, num_parallel_workers=3)

    dataset_num = 0

    for _ in dataset.create_dict_iterator(output_numpy=True):
        dataset_num += 1

    assert dataset_num == sampler_num


def dataset_call_c_transforms_func(sampler, shard_id=0):
    """
    All of c_transforms
    Returns:

    """

    def filter_func_ge(_, label):
        if label > 20:
            return False
        return True

    def flat_map_func(x):
        d = ds.Cifar10Dataset(DATA_DIR_10, num_samples=100)
        return d

    crop_height = 300
    crop_width = 300
    target_height = 200
    target_width = 200
    interpolation_mode = Inter.BILINEAR
    scalelb = 0.5
    scaleub = 200.5
    aspectlb = 200.5
    aspectub = 200.5
    targetheight = 100
    targetwidth = 100
    interpolation = Inter.BILINEAR
    maxiter = 100
    skipcount = 5
    unused_takecount = 1
    unused_repeatcount = 10
    l1 = []

    data = ds.TextFileDataset(DATA_FILE, num_samples=1)
    data = data.flat_map(flat_map_func)

    count = 0
    for d in data:
        assert isinstance(d[0].asnumpy(), np.ndarray)
        count += 1
    assert count == 100

    op1 = UserPyOp()
    cb1 = UserCallback(op1)
    dataset = ds.Cifar10Dataset(DATA_DIR_10)
    dataset = dataset.map(operations=op1, callbacks=cb1)
    dataset = dataset.concat(data)
    i = 0
    for _ in dataset.create_dict_iterator(output_numpy=True):
        i += 1
    assert i == 10100

    dataset1, _ = dataset.split([0.5, 0.5], False)
    i = 0
    for _ in dataset1.create_dict_iterator(output_numpy=True):
        i += 1
    assert i == 5050

    dataset = ds.Cifar10Dataset(DATA_DIR_10, sampler=sampler, num_parallel_workers=3)
    beforesize = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        l1.append(data['image'])
        beforesize += 1
    l1.clear()

    dataset = dataset.skip(count=skipcount)
    aftersize = 0
    for _ in dataset.create_dict_iterator(output_numpy=True):
        aftersize += 1
    assert (beforesize - aftersize) == skipcount

    dataset = dataset.filter(predicate=filter_func_ge, input_columns=["image", "label"], num_parallel_workers=4)

    padded_samples = [{'image': np.zeros((2, 3, 3), np.uint8), 'label': np.array(99, np.uint32)}]
    padded_ds = ds.PaddedDataset(padded_samples)
    dataset = dataset + padded_ds
    testsampler = ds.DistributedSampler(num_shards=2, shard_id=shard_id, shuffle=False, num_samples=None)
    dataset.use_sampler(testsampler)

    randomcrop_op = vision.RandomCrop(size=(crop_height, crop_width), padding=(1, 1), pad_if_needed=True,
                                       fill_value=(1, 1, 0), padding_mode=Border.CONSTANT)
    dataset = dataset.map(input_columns=["image"], operations=randomcrop_op, num_parallel_workers=8)

    randomcrop_op = vision.RandomCrop(size=(crop_height, crop_width), padding=(1, 1), pad_if_needed=True,
                                       fill_value=(1, 1, 0), padding_mode=Border.EDGE)
    dataset = dataset.map(input_columns=["image"], operations=randomcrop_op, num_parallel_workers=3)

    randomcrop_op = vision.RandomCrop(size=(crop_height, crop_width), padding=(1, 1), pad_if_needed=True,
                                       fill_value=(1, 1, 0), padding_mode=Border.REFLECT)
    dataset = dataset.map(input_columns=["image"], operations=randomcrop_op, num_parallel_workers=3)

    randomcrop_op = vision.RandomCrop(size=(crop_height, crop_width), padding=(1, 1), pad_if_needed=True,
                                       fill_value=(1, 1, 0), padding_mode=Border.SYMMETRIC)
    dataset = dataset.map(input_columns=["image"], operations=randomcrop_op, num_parallel_workers=3)

    dataset = dataset.apply(apply_func)

    resize_op = vision.Resize((target_height, target_width), interpolation_mode)
    dataset = dataset.map(input_columns=["image"], operations=resize_op, num_parallel_workers=3)

    randomcropandresize_op = vision.RandomResizedCrop((targetheight, targetwidth), (scalelb, scaleub),
                                                       (aspectlb, aspectub), interpolation, maxiter)
    dataset = dataset.map(input_columns=["image"], operations=randomcropandresize_op, num_parallel_workers=3)

    unused_num_classes = dataset.num_classes()
    unused_num_classes = 100


def test_cifar10_sequential_sampler():
    """
    Feature: Cifar10Dataset
    Description: Test Cifar10Dataset with SequentialSampler
    Expectation: The dataset is processed as expected
    """
    logger.info("Test Cifar10Dataset Op with SequentialSampler")
    num_samples = 30
    sampler = ds.SequentialSampler(num_samples=num_samples)
    data1 = ds.Cifar10Dataset(DATA_DIR_10, sampler=sampler)
    data2 = ds.Cifar10Dataset(DATA_DIR_10, shuffle=False, num_samples=num_samples)
    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_equal(item1["label"], item2["label"])
        num_iter += 1
    assert num_iter == num_samples

    sampler = ds.SequentialSampler(num_samples=100)
    dataset_call_c_transforms_func(sampler=sampler)
    DatasetCTransformsFunc().batch_map_fun(sampler=sampler)

    sampler = ds.SequentialSampler(start_index=2, num_samples=100)
    dataset_call_py_transforms_func(sampler=sampler, sampler_num=100)


def test_cifar10_with_distributed_sampler():
    """
    Feature: Cifar10Dataset
    Description: Test Cifar10Dataset with distributed sampler
    Expectation: The dataset is processed as expected
    """
    # DistributedSampler
    sampler = ds.DistributedSampler(10, 1, num_samples=100)
    dataset_call_c_transforms_func(sampler=sampler)
    DatasetCTransformsFunc().batch_map_fun(sampler=sampler)

    # DistributedSampler with offset
    sampler = ds.DistributedSampler(10, 1, num_samples=100, offset=1)
    dataset_call_py_transforms_func(sampler=sampler, sampler_num=100)


def test_cifar10_with_random_sampler():
    """
    Feature: Cifar10Dataset
    Description: Test Cifar10Dataset with random sampler
    Expectation: The dataset is processed as expected
    """
    sampler = ds.RandomSampler(True, 50)
    dataset_call_c_transforms_func(sampler=sampler)
    DatasetCTransformsFunc().batch_map_fun(sampler=sampler)

    sampler = ds.RandomSampler(True, 50)
    dataset_call_py_transforms_func(sampler=sampler, sampler_num=50)


def test_cifar10_with_udf_sampler():
    """
    Feature: Cifar10Dataset
    Description: Test Cifar10Dataset with udf sampler
    Expectation: The dataset is processed as expected
    """

    class MySampler(ds.Sampler):
        '''test'''

        def __init__(self):
            super().__init__()
            # at this stage, self.dataset_size and self.num_samples are not yet known
            self.cnt = 0

        def __iter__(self):  # first epoch, all 0, second epoch all 1, third all 2 etc.. ...
            return iter([self.cnt for i in range(self.dataset_size)])

        def reset(self):
            self.cnt = (self.cnt + 1) % self.dataset_size

    dataset_call_c_transforms_func(sampler=MySampler())
