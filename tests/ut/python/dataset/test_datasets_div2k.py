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
"""Dataset regression coverage for datasets div2k."""
import matplotlib.pyplot as plt
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.vision as vision


DATASET_DIR = "../data/dataset/testDIV2KData/div2k"


def test_div2k_basic(plot=False):
    """
    Feature: DIV2KDataset
    Description: Test basic read on DIV2KDataset
    Expectation: The dataset is processed as expected
    """
    usage = "train"          # train, valid, all
    downgrade = "bicubic"    # bicubic, unknown, mild, difficult, wild
    scale = 2                # 2, 3, 4, 8

    data = ds.DIV2KDataset(DATASET_DIR, usage=usage, downgrade=downgrade, scale=scale, decode=True)
    count = 0
    hr_images_list = []
    lr_images_list = []
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        hr_images_list.append(item['hr_image'])
        lr_images_list.append(item['lr_image'])
        count = count + 1
    assert count == 5
    if plot:
        flag = "{}_{}_{}".format(usage, scale, downgrade)
        visualize_dataset(hr_images_list, lr_images_list, flag)


def visualize_dataset(hr_images_list, lr_images_list, flag):
    """
    Helper function to visualize the dataset samples
    """
    image_num = len(hr_images_list)
    for i in range(image_num):
        plt.subplot(121)
        plt.imshow(hr_images_list[i])
        plt.title('Original')
        plt.subplot(122)
        plt.imshow(lr_images_list[i])
        plt.title(flag)
        plt.savefig('./div2k_{}_{}.jpg'.format(flag, str(i)))


def test_div2k_basic_func():
    """
    Feature: DIV2KDataset
    Description: Test basic functions for DIV2KDataset
    Expectation: The dataset is processed as expected
    """
    # case 0: test usage equal to `all`
    usage = "all"              # train, valid, all
    downgrade = "bicubic"    # bicubic, unknown, mild, difficult, wild
    scale = 2                  # 2, 3, 4, 8

    data0 = ds.DIV2KDataset(DATASET_DIR, usage=usage, downgrade=downgrade, scale=scale)
    num_iter0 = 0
    for _ in data0.create_dict_iterator(num_epochs=1):
        num_iter0 += 1
    assert num_iter0 == 6

    # case 1: test num_samples
    usage = "train"            # train, valid, all

    data1 = ds.DIV2KDataset(DATASET_DIR, usage=usage, downgrade=downgrade, scale=scale, num_samples=4)
    num_iter1 = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_iter1 += 1
    assert num_iter1 == 4

    # case 2: test repeat
    data2 = ds.DIV2KDataset(DATASET_DIR, usage=usage, downgrade=downgrade, scale=scale, num_samples=3)
    data2 = data2.repeat(5)
    num_iter2 = 0
    for _ in data2.create_dict_iterator(num_epochs=1):
        num_iter2 += 1
    assert num_iter2 == 15

    # case 3: test batch with drop_remainder=False
    data3 = ds.DIV2KDataset(DATASET_DIR, usage=usage, downgrade=downgrade, scale=scale, decode=True)
    assert data3.get_dataset_size() == 5
    assert data3.get_batch_size() == 1
    resize_op = vision.Resize([100, 100])
    data3 = data3.map(operations=resize_op, input_columns=["hr_image"], num_parallel_workers=1)
    data3 = data3.map(operations=resize_op, input_columns=["lr_image"], num_parallel_workers=1)
    data3 = data3.batch(batch_size=3)  # drop_remainder is default to be False
    assert data3.get_dataset_size() == 2
    assert data3.get_batch_size() == 3
    num_iter3 = 0
    for _ in data3.create_dict_iterator(num_epochs=1):
        num_iter3 += 1
    assert num_iter3 == 2

    # case 4: test batch with drop_remainder=True
    data4 = ds.DIV2KDataset(DATASET_DIR, usage=usage, downgrade=downgrade, scale=scale, decode=True)
    assert data4.get_dataset_size() == 5
    assert data4.get_batch_size() == 1
    data4 = data4.map(operations=resize_op, input_columns=["hr_image"], num_parallel_workers=1)
    data4 = data4.map(operations=resize_op, input_columns=["lr_image"], num_parallel_workers=1)
    data4 = data4.batch(batch_size=3, drop_remainder=True)  # the rest of incomplete batch will be dropped
    assert data4.get_dataset_size() == 1
    assert data4.get_batch_size() == 3
    num_iter4 = 0
    for _ in data4.create_dict_iterator(num_epochs=1):
        num_iter4 += 1
    assert num_iter4 == 1

    # case 5: test get_col_names
    data5 = ds.DIV2KDataset(DATASET_DIR, usage=usage, downgrade=downgrade, scale=scale, num_samples=1)
    assert data5.get_col_names() == ["hr_image", "lr_image"]


def test_div2k_sequential_sampler():
    """
    Feature: DIV2KDataset
    Description: Test DIV2KDataset with SequentialSampler
    Expectation: The dataset is processed as expected
    """
    usage = "train"          # train, valid, all
    downgrade = "bicubic"    # bicubic, unknown, mild, difficult, wild
    scale = 2                # 2, 3, 4, 8

    num_samples = 2
    sampler = ds.SequentialSampler(num_samples=num_samples)
    data1 = ds.DIV2KDataset(DATASET_DIR, usage=usage, downgrade=downgrade, scale=scale, sampler=sampler)
    data2 = ds.DIV2KDataset(DATASET_DIR, usage=usage, downgrade=downgrade, scale=scale, shuffle=False,
                            num_samples=num_samples)
    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(item1["hr_image"], item2["hr_image"])
        np.testing.assert_array_equal(item1["lr_image"], item2["lr_image"])
        num_iter += 1
    assert num_iter == num_samples
