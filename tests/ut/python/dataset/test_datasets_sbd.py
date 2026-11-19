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
"""Dataset regression coverage for datasets sbd."""
import math

import matplotlib.pyplot as plt
import numpy as np

import mindspore.dataset as ds
from mindspore import log as logger
import mindspore.dataset.vision as vision

DATASET_DIR = "../data/dataset/testSBData/sbd"


def visualize_dataset(images, labels, task):
    """
    Helper function to visualize the dataset samples
    """
    image_num = len(images)
    subplot_rows = 1 if task == "Segmentation" else 4
    for i in range(image_num):
        plt.imshow(images[i])
        plt.title('Original')
        plt.savefig('./sbd_original_{}.jpg'.format(str(i)))
        if task == "Segmentation":
            plt.imshow(labels[i])
            plt.title(task)
            plt.savefig('./sbd_segmentation_{}.jpg'.format(str(i)))
        else:
            b_num = labels[i].shape[0]
            for j in range(b_num):
                plt.subplot(subplot_rows, math.ceil(b_num / subplot_rows), j + 1)
                plt.imshow(labels[i][j])
            plt.savefig('./sbd_boundaries_{}.jpg'.format(str(i)))
        plt.close()


def test_sbd_basic01(plot=False):
    """
    Feature: SBDataset
    Description: Test SBDataset with different usage
    Expectation: The dataset is processed as expected
    """
    task = 'Segmentation'  # Boundaries, Segmentation
    data = ds.SBDataset(DATASET_DIR, task=task, usage='all', shuffle=False, decode=True)
    count = 0
    images_list = []
    task_list = []
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        images_list.append(item['image'])
        task_list.append(item['task'])
        count = count + 1
    assert count == 6
    if plot:
        visualize_dataset(images_list, task_list, task)

    data2 = ds.SBDataset(DATASET_DIR, task=task, usage='train', shuffle=False, decode=False)
    count = 0
    for item in data2.create_dict_iterator(num_epochs=1, output_numpy=True):
        count = count + 1
    assert count == 4

    data3 = ds.SBDataset(DATASET_DIR, task=task, usage='val', shuffle=False, decode=False)
    count = 0
    for item in data3.create_dict_iterator(num_epochs=1, output_numpy=True):
        count = count + 1
    assert count == 2


def test_sbd_basic02():
    """
    Feature: SBDataset
    Description: Test SBDataset with repeat and batch operation
    Expectation: The dataset is processed as expected
    """
    # Boundaries, Segmentation
    # case 1: test num_samples
    data1 = ds.SBDataset(DATASET_DIR, task='Boundaries', usage='train', num_samples=3, shuffle=False)
    num_iter1 = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_iter1 += 1
    assert num_iter1 == 3

    # case 2: test repeat
    data2 = ds.SBDataset(DATASET_DIR, task='Boundaries', usage='train', num_samples=4, shuffle=False)
    data2 = data2.repeat(5)
    num_iter2 = 0
    for _ in data2.create_dict_iterator(num_epochs=1):
        num_iter2 += 1
    assert num_iter2 == 20

    # case 3: test batch with drop_remainder=False
    data3 = ds.SBDataset(DATASET_DIR, task='Segmentation', usage='train', shuffle=False, decode=True)
    resize_op = vision.Resize((100, 100))
    data3 = data3.map(operations=resize_op, input_columns=["image"], num_parallel_workers=1)
    data3 = data3.map(operations=resize_op, input_columns=["task"], num_parallel_workers=1)
    assert data3.get_dataset_size() == 4
    assert data3.get_batch_size() == 1
    data3 = data3.batch(batch_size=3)  # drop_remainder is default to be False
    assert data3.get_dataset_size() == 2
    assert data3.get_batch_size() == 3
    num_iter3 = 0
    for _ in data3.create_dict_iterator(num_epochs=1):
        num_iter3 += 1
    assert num_iter3 == 2

    # case 4: test batch with drop_remainder=True
    data4 = ds.SBDataset(DATASET_DIR, task='Segmentation', usage='train', shuffle=False, decode=True)
    resize_op = vision.Resize((100, 100))
    data4 = data4.map(operations=resize_op, input_columns=["image"], num_parallel_workers=1)
    data4 = data4.map(operations=resize_op, input_columns=["task"], num_parallel_workers=1)
    assert data4.get_dataset_size() == 4
    assert data4.get_batch_size() == 1
    data4 = data4.batch(batch_size=3, drop_remainder=True)  # the rest of incomplete batch will be dropped
    assert data4.get_dataset_size() == 1
    assert data4.get_batch_size() == 3
    num_iter4 = 0
    for _ in data4.create_dict_iterator(num_epochs=1):
        num_iter4 += 1
    assert num_iter4 == 1


def test_sbd_sequential_sampler():
    """
    Feature: SBDataset
    Description: Test SBDataset with SequentialSampler
    Expectation: The dataset is processed as expected
    """
    logger.info("Test SBDataset Op with SequentialSampler")
    num_samples = 5
    sampler = ds.SequentialSampler(num_samples=num_samples)
    data1 = ds.SBDataset(DATASET_DIR, task='Segmentation', usage='all', sampler=sampler)
    data2 = ds.SBDataset(DATASET_DIR, task='Segmentation', usage='all', shuffle=False, num_samples=num_samples)
    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(item1["task"], item2["task"])
        num_iter += 1
    assert num_iter == num_samples




def test_sbd_usage():
    """
    Feature: SBDataset
    Description: Test SBDataset image readings with usage flag
    Expectation: The dataset is processed as expected
    """

    def test_config(usage):
        try:
            data = ds.SBDataset(DATASET_DIR, task='Segmentation', usage=usage)
            num_rows = 0
            for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                num_rows += 1
        except (ValueError, TypeError, RuntimeError) as e:
            return str(e)
        return num_rows

    assert test_config("train") == 4
    assert test_config("train_noval") == 4
    assert test_config("val") == 2
    assert test_config("all") == 6
    assert "usage is not within the valid set of ['train', 'val', 'train_noval', 'all']" in test_config("invalid")
    assert "Argument usage with value ['list'] is not of type [<class 'str'>]" in test_config(["list"])
