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
import os

import json
import matplotlib.pyplot as plt
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.vision as vision


DATASET_DIR = "../data/dataset/testCityscapesData/cityscapes"
DATASET_DIR_TASK_JSON = "../data/dataset/testCityscapesData/cityscapes/testTaskJson"


def test_cityscapes_basic(plot=False):
    """
    Feature: CityscapesDataset
    Description: Test basic read on CityscapesDataset
    Expectation: The dataset is processed as expected
    """
    task = "color"         # instance semantic polygon color
    quality_mode = "fine"  # fine coarse
    usage = "train"        # quality_mode=fine 'train', 'test', 'val', 'all' else 'train', 'train_extra', 'val', 'all'
    data = ds.CityscapesDataset(DATASET_DIR, usage=usage, quality_mode=quality_mode, task=task,
                                decode=True, shuffle=False)
    count = 0
    images_list = []
    task_list = []
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        images_list.append(item['image'])
        task_list.append(item['task'])
        count = count + 1
    assert count == 5
    if plot:
        visualize_dataset(images_list, task_list, task)


def visualize_dataset(images, labels, task):
    """
    Helper function to visualize the dataset samples.
    """
    if task == "polygon":
        return
    image_num = len(images)
    for i in range(image_num):
        plt.subplot(121)
        plt.imshow(images[i])
        plt.title('Original')
        plt.subplot(122)
        plt.imshow(labels[i])
        plt.title(task)
        plt.savefig('./cityscapes_{}_{}.jpg'.format(task, str(i)))


def test_cityscapes_polygon():
    """
    Feature: CityscapesDataset
    Description: Test CityscapesDataset with task of polygon
    Expectation: The dataset is processed as expected
    """
    usage = "train"
    quality_mode = "fine"
    task = "polygon"
    data = ds.CityscapesDataset(DATASET_DIR_TASK_JSON, usage=usage, quality_mode=quality_mode, task=task)
    count = 0
    json_file = os.path.join(DATASET_DIR_TASK_JSON, "gtFine/train/aa/aa_000000_gtFine_polygons.json")
    with open(json_file, "r") as f:
        expected = json.load(f)
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        task_dict = json.loads(str(item['task']))
        assert task_dict == expected
        count = count + 1
    assert count == 1


def test_cityscapes_basic_func():
    """
    Feature: CityscapesDataset
    Description: Test CityscapesDataset with repeat, batch, and getter operation
    Expectation: The dataset is processed as expected
    """
    # case 1: test num_samples
    usage = "train"
    quality_mode = "fine"
    task = "color"
    data1 = ds.CityscapesDataset(DATASET_DIR, usage=usage, quality_mode=quality_mode, task=task, num_samples=4)
    num_iter1 = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_iter1 += 1
    assert num_iter1 == 4

    # case 2: test repeat
    data2 = ds.CityscapesDataset(DATASET_DIR, usage=usage, quality_mode=quality_mode, task=task, num_samples=5)
    data2 = data2.repeat(5)
    num_iter2 = 0
    for _ in data2.create_dict_iterator(num_epochs=1):
        num_iter2 += 1
    assert num_iter2 == 25

    # case 3: test batch with drop_remainder=False
    data3 = ds.CityscapesDataset(DATASET_DIR, usage=usage, quality_mode=quality_mode, task=task, decode=True)
    resize_op = vision.Resize((100, 100))
    data3 = data3.map(operations=resize_op, input_columns=["image"], num_parallel_workers=1)
    data3 = data3.map(operations=resize_op, input_columns=["task"], num_parallel_workers=1)
    assert data3.get_dataset_size() == 5
    assert data3.get_batch_size() == 1
    data3 = data3.batch(batch_size=3)  # drop_remainder is default to be False
    assert data3.get_dataset_size() == 2
    assert data3.get_batch_size() == 3
    num_iter3 = 0
    for _ in data3.create_dict_iterator(num_epochs=1):
        num_iter3 += 1
    assert num_iter3 == 2

    # case 4: test batch with drop_remainder=True
    data4 = ds.CityscapesDataset(DATASET_DIR, usage=usage, quality_mode=quality_mode, task=task, decode=True)
    resize_op = vision.Resize((100, 100))
    data4 = data4.map(operations=resize_op, input_columns=["image"], num_parallel_workers=1)
    data4 = data4.map(operations=resize_op, input_columns=["task"], num_parallel_workers=1)
    assert data4.get_dataset_size() == 5
    assert data4.get_batch_size() == 1
    data4 = data4.batch(batch_size=3, drop_remainder=True)  # the rest of incomplete batch will be dropped
    assert data4.get_dataset_size() == 1
    assert data4.get_batch_size() == 3
    num_iter4 = 0
    for _ in data4.create_dict_iterator(num_epochs=1):
        num_iter4 += 1
    assert num_iter4 == 1

    # case 5: test get_col_names
    data5 = ds.CityscapesDataset(DATASET_DIR, usage=usage, quality_mode=quality_mode, task=task, decode=True)
    assert data5.get_col_names() == ["image", "task"]


def test_cityscapes_sequential_sampler():
    """
    Feature: CityscapesDataset
    Description: Test CityscapesDataset with SequentialSampler
    Expectation: The dataset is processed as expected
    """
    task = "color"
    quality_mode = "fine"
    usage = "train"

    num_samples = 5
    sampler = ds.SequentialSampler(num_samples=num_samples)
    data1 = ds.CityscapesDataset(DATASET_DIR, usage=usage, quality_mode=quality_mode, task=task, sampler=sampler)
    data2 = ds.CityscapesDataset(DATASET_DIR, usage=usage, quality_mode=quality_mode, task=task,
                                 shuffle=False, num_samples=num_samples)
    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(item1["task"], item2["task"])
        num_iter += 1
    assert num_iter == num_samples


def test_cityscapes_exception():
    """
    Feature: CityscapesDataset
    Description: Test CityscapesDataset with wrong parameters
    Expectation: Throw correct error as expected
    """
    task = "color"
    quality_mode = "fine"
    usage = "train"

    error_msg_1 = "does not exist or is not a directory or permission denied!"
    with pytest.raises(ValueError, match=error_msg_1):
        ds.CityscapesDataset("NoExistsDir", usage=usage, quality_mode=quality_mode, task=task)

    error_msg_2 = "sampler and shuffle cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_2):
        ds.CityscapesDataset(DATASET_DIR, usage=usage, quality_mode=quality_mode, task=task, shuffle=False,
                             sampler=ds.PKSampler(3))

    error_msg_3 = "sampler and sharding cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_3):
        ds.CityscapesDataset(DATASET_DIR, usage=usage, quality_mode=quality_mode, task=task, num_shards=2,
                             shard_id=0, sampler=ds.PKSampler(3))

    error_msg_4 = "num_shards is specified and currently requires shard_id as well"
    with pytest.raises(RuntimeError, match=error_msg_4):
        ds.CityscapesDataset(DATASET_DIR, usage=usage, quality_mode=quality_mode, task=task, num_shards=10)

    error_msg_5 = "shard_id is specified but num_shards is not"
    with pytest.raises(RuntimeError, match=error_msg_5):
        ds.CityscapesDataset(DATASET_DIR, usage=usage, quality_mode=quality_mode, task=task, shard_id=0)

    error_msg_6 = "Input shard_id is not within the required interval"
    with pytest.raises(ValueError, match=error_msg_6):
        ds.CityscapesDataset(DATASET_DIR, usage=usage, quality_mode=quality_mode, task=task, num_shards=5, shard_id=-1)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.CityscapesDataset(DATASET_DIR, usage=usage, quality_mode=quality_mode, task=task, num_shards=5, shard_id=5)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.CityscapesDataset(DATASET_DIR, usage=usage, quality_mode=quality_mode, task=task, num_shards=2, shard_id=5)

    error_msg_7 = "num_parallel_workers exceeds"
    with pytest.raises(ValueError, match=error_msg_7):
        ds.CityscapesDataset(DATASET_DIR, usage=usage, quality_mode=quality_mode, task=task, shuffle=False,
                             num_parallel_workers=0)
    with pytest.raises(ValueError, match=error_msg_7):
        ds.CityscapesDataset(DATASET_DIR, usage=usage, quality_mode=quality_mode, task=task, shuffle=False,
                             num_parallel_workers=256)
    with pytest.raises(ValueError, match=error_msg_7):
        ds.CityscapesDataset(DATASET_DIR, usage=usage, quality_mode=quality_mode, task=task, shuffle=False,
                             num_parallel_workers=-2)

    error_msg_8 = "Argument shard_id"
    with pytest.raises(TypeError, match=error_msg_8):
        ds.CityscapesDataset(DATASET_DIR, usage=usage, quality_mode=quality_mode, task=task, num_shards=2, shard_id="0")

    def exception_func(item):
        raise Exception("Error occur!")

    try:
        data = ds.CityscapesDataset(DATASET_DIR, usage=usage, quality_mode=quality_mode, task=task)
        data = data.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        num_rows = 0
        for _ in data.create_dict_iterator(num_epochs=1):
            num_rows += 1
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is:" in str(e)

    try:
        data = ds.CityscapesDataset(DATASET_DIR, usage=usage, quality_mode=quality_mode, task=task)
        data = data.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        num_rows = 0
        for _ in data.create_dict_iterator(num_epochs=1):
            num_rows += 1
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is:" in str(e)


def test_cityscapes_param():
    """
    Feature: CityscapesDataset
    Description: Test CityscapesDataset with basic parameters like usage, quality_mode, and task
    Expectation: The dataset is processed or error is raised as expected
    """
    def test_config(usage="train", quality_mode="fine", task="color"):
        try:
            data = ds.CityscapesDataset(DATASET_DIR, usage=usage, quality_mode=quality_mode, task=task)
            num_rows = 0
            for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                num_rows += 1
        except (ValueError, TypeError, RuntimeError) as e:
            return str(e)
        return num_rows

    assert test_config(usage="train") == 5
    assert test_config(usage="test") == 1
    assert test_config(usage="val") == 1
    assert test_config(usage="all") == 7
    assert "usage is not within the valid set of ['train', 'test', 'val', 'all']" \
           in test_config("invalid", "fine", "instance")
    assert "Argument usage with value ['list'] is not of type [<class 'str'>]" \
           in test_config(["list"], "fine", "instance")
    assert "quality_mode is not within the valid set of ['fine', 'coarse']" \
           in test_config("train", "invalid", "instance")
    assert "Argument quality_mode with value ['list'] is not of type [<class 'str'>]" \
           in test_config("train", ["list"], "instance")
    assert "task is not within the valid set of ['instance', 'semantic', 'polygon', 'color']." \
           in test_config("train", "fine", "invalid")
    assert "Argument task with value ['list'] is not of type [<class 'str'>], but got <class 'list'>." \
           in test_config("train", "fine", ["list"])


if __name__ == "__main__":
    test_cityscapes_basic()
    test_cityscapes_polygon()
    test_cityscapes_basic_func()
    test_cityscapes_sequential_sampler()
    test_cityscapes_exception()
    test_cityscapes_param()
