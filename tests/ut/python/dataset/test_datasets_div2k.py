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
import matplotlib.pyplot as plt
import numpy as np
import pytest

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


def test_div2k_exception():
    """
    Feature: DIV2KDataset
    Description: Test invalid parameters for DIV2KDataset
    Expectation: Throw correct error as expected
    """
    usage = "train"          # train, valid, all
    downgrade = "bicubic"    # bicubic, unknown, mild, difficult, wild
    scale = 2                # 2, 3, 4, 8

    error_msg_1 = "does not exist or is not a directory or permission denied!"
    with pytest.raises(ValueError, match=error_msg_1):
        ds.DIV2KDataset("NoExistsDir", usage=usage, downgrade=downgrade, scale=scale)

    error_msg_2 = r"Input usage is not within the valid set of \['train', 'valid', 'all'\]."
    with pytest.raises(ValueError, match=error_msg_2):
        ds.DIV2KDataset(DATASET_DIR, usage="test", downgrade=downgrade, scale=scale)

    error_msg_3 = r"Input scale is not within the valid set of \[2, 3, 4, 8\]."
    with pytest.raises(ValueError, match=error_msg_3):
        ds.DIV2KDataset(DATASET_DIR, usage=usage, scale=16, downgrade=downgrade)

    error_msg_4 = r"Input downgrade is not within the valid set of .*"
    with pytest.raises(ValueError, match=error_msg_4):
        ds.DIV2KDataset(DATASET_DIR, usage=usage, scale=scale, downgrade="downgrade")

    error_msg_5 = "sampler and shuffle cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_5):
        ds.DIV2KDataset(DATASET_DIR, usage=usage, downgrade=downgrade, scale=scale, shuffle=False,
                        sampler=ds.PKSampler(3))

    error_msg_6 = "sampler and sharding cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_6):
        ds.DIV2KDataset(DATASET_DIR, usage=usage, downgrade=downgrade, scale=scale, num_shards=2, shard_id=0,
                        sampler=ds.PKSampler(3))

    error_msg_7 = "num_shards is specified and currently requires shard_id as well"
    with pytest.raises(RuntimeError, match=error_msg_7):
        ds.DIV2KDataset(DATASET_DIR, usage=usage, downgrade=downgrade, scale=scale, num_shards=10)

    error_msg_8 = "shard_id is specified but num_shards is not"
    with pytest.raises(RuntimeError, match=error_msg_8):
        ds.DIV2KDataset(DATASET_DIR, usage=usage, downgrade=downgrade, scale=scale, shard_id=0)

    error_msg_9 = "Input shard_id is not within the required interval"
    with pytest.raises(ValueError, match=error_msg_9):
        ds.DIV2KDataset(DATASET_DIR, usage=usage, downgrade=downgrade, scale=scale, num_shards=5, shard_id=-1)
    with pytest.raises(ValueError, match=error_msg_9):
        ds.DIV2KDataset(DATASET_DIR, usage=usage, downgrade=downgrade, scale=scale, num_shards=5, shard_id=5)
    with pytest.raises(ValueError, match=error_msg_9):
        ds.DIV2KDataset(DATASET_DIR, usage=usage, downgrade=downgrade, scale=scale, num_shards=2, shard_id=5)

    error_msg_10 = "num_parallel_workers exceeds"
    with pytest.raises(ValueError, match=error_msg_10):
        ds.DIV2KDataset(DATASET_DIR, usage=usage, downgrade=downgrade, scale=scale, shuffle=False,
                        num_parallel_workers=0)
    with pytest.raises(ValueError, match=error_msg_10):
        ds.DIV2KDataset(DATASET_DIR, usage=usage, downgrade=downgrade, scale=scale, shuffle=False,
                        num_parallel_workers=256)
    with pytest.raises(ValueError, match=error_msg_10):
        ds.DIV2KDataset(DATASET_DIR, usage=usage, downgrade=downgrade, scale=scale, shuffle=False,
                        num_parallel_workers=-2)

    error_msg_11 = "Argument shard_id"
    with pytest.raises(TypeError, match=error_msg_11):
        ds.DIV2KDataset(DATASET_DIR, usage=usage, downgrade=downgrade, scale=scale, num_shards=2, shard_id="0")

    def exception_func(item):
        raise Exception("Error occur!")

    try:
        data = ds.DIV2KDataset(DATASET_DIR, usage=usage, downgrade=downgrade, scale=scale)
        data = data.map(operations=exception_func, input_columns=["hr_image"], num_parallel_workers=1)
        num_rows = 0
        for _ in data.create_dict_iterator(num_epochs=1):
            num_rows += 1
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is:" in str(e)

    try:
        data = ds.DIV2KDataset(DATASET_DIR, usage=usage, downgrade=downgrade, scale=scale)
        data = data.map(operations=exception_func, input_columns=["hr_image"], num_parallel_workers=1)
        num_rows = 0
        for _ in data.create_dict_iterator(num_epochs=1):
            num_rows += 1
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is:" in str(e)


if __name__ == "__main__":
    test_div2k_basic()
    test_div2k_basic_func()
    test_div2k_sequential_sampler()
    test_div2k_exception()
