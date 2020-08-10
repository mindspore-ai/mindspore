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
Testing the MixUpBatch op in DE
"""
import numpy as np
import pytest
import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.c_transforms as vision
import mindspore.dataset.transforms.c_transforms as data_trans
from mindspore import log as logger
from util import save_and_check_md5, diff_mse, visualize_list, config_get_set_seed, \
    config_get_set_num_parallel_workers

DATA_DIR = "../data/dataset/testCifar10Data"

GENERATE_GOLDEN = False

def test_mixup_batch_success1(plot=False):
    """
    Test MixUpBatch op with specified alpha parameter
    """
    logger.info("test_mixup_batch_success1")

    # Original Images
    ds_original = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)
    ds_original = ds_original.batch(5, drop_remainder=True)

    images_original = None
    for idx, (image, _) in enumerate(ds_original):
        if idx == 0:
            images_original = image
        else:
            images_original = np.append(images_original, image, axis=0)

    # MixUp Images
    data1 = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)

    one_hot_op = data_trans.OneHot(num_classes=10)
    data1 = data1.map(input_columns=["label"], operations=one_hot_op)
    mixup_batch_op = vision.MixUpBatch(2)
    data1 = data1.batch(5, drop_remainder=True)
    data1 = data1.map(input_columns=["image", "label"], operations=mixup_batch_op)

    images_mixup = None
    for idx, (image, _) in enumerate(data1):
        if idx == 0:
            images_mixup = image
        else:
            images_mixup = np.append(images_mixup, image, axis=0)
    if plot:
        visualize_list(images_original, images_mixup)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_mixup[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))


def test_mixup_batch_success2(plot=False):
    """
    Test MixUpBatch op without specified alpha parameter.
    Alpha parameter will be selected by default in this case
    """
    logger.info("test_mixup_batch_success2")

    # Original Images
    ds_original = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)
    ds_original = ds_original.batch(5, drop_remainder=True)

    images_original = None
    for idx, (image, _) in enumerate(ds_original):
        if idx == 0:
            images_original = image
        else:
            images_original = np.append(images_original, image, axis=0)

    # MixUp Images
    data1 = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)

    one_hot_op = data_trans.OneHot(num_classes=10)
    data1 = data1.map(input_columns=["label"], operations=one_hot_op)
    mixup_batch_op = vision.MixUpBatch()
    data1 = data1.batch(5, drop_remainder=True)
    data1 = data1.map(input_columns=["image", "label"], operations=mixup_batch_op)

    images_mixup = np.array([])
    for idx, (image, _) in enumerate(data1):
        if idx == 0:
            images_mixup = image
        else:
            images_mixup = np.append(images_mixup, image, axis=0)
    if plot:
        visualize_list(images_original, images_mixup)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_mixup[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))


def test_mixup_batch_md5():
    """
    Test MixUpBatch with MD5:
    """
    logger.info("test_mixup_batch_md5")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # MixUp Images
    data = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)

    one_hot_op = data_trans.OneHot(num_classes=10)
    data = data.map(input_columns=["label"], operations=one_hot_op)
    mixup_batch_op = vision.MixUpBatch()
    data = data.batch(5, drop_remainder=True)
    data = data.map(input_columns=["image", "label"], operations=mixup_batch_op)

    filename = "mixup_batch_c_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_mixup_batch_fail1():
    """
    Test MixUpBatch Fail 1
    We expect this to fail because the images and labels are not batched
    """
    logger.info("test_mixup_batch_fail1")

    # Original Images
    ds_original = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)
    ds_original = ds_original.batch(5)

    images_original = np.array([])
    for idx, (image, _) in enumerate(ds_original):
        if idx == 0:
            images_original = image
        else:
            images_original = np.append(images_original, image, axis=0)

    # MixUp Images
    data1 = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)

    one_hot_op = data_trans.OneHot(num_classes=10)
    data1 = data1.map(input_columns=["label"], operations=one_hot_op)
    mixup_batch_op = vision.MixUpBatch(0.1)
    with pytest.raises(RuntimeError) as error:
        data1 = data1.map(input_columns=["image", "label"], operations=mixup_batch_op)
        for idx, (image, _) in enumerate(data1):
            if idx == 0:
                images_mixup = image
            else:
                images_mixup = np.append(images_mixup, image, axis=0)
        error_message = "You must batch before calling MixUp"
        assert error_message in str(error.value)


def test_mixup_batch_fail2():
    """
    Test MixUpBatch Fail 2
    We expect this to fail because alpha is negative
    """
    logger.info("test_mixup_batch_fail2")

    # Original Images
    ds_original = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)
    ds_original = ds_original.batch(5)

    images_original = np.array([])
    for idx, (image, _) in enumerate(ds_original):
        if idx == 0:
            images_original = image
        else:
            images_original = np.append(images_original, image, axis=0)

    # MixUp Images
    data1 = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)

    one_hot_op = data_trans.OneHot(num_classes=10)
    data1 = data1.map(input_columns=["label"], operations=one_hot_op)
    with pytest.raises(ValueError) as error:
        vision.MixUpBatch(-1)
        error_message = "Input is not within the required interval"
        assert error_message in str(error.value)


def test_mixup_batch_fail3():
    """
    Test MixUpBatch op
    We expect this to fail because label column is not passed to mixup_batch
    """
    # Original Images
    ds_original = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)
    ds_original = ds_original.batch(5, drop_remainder=True)

    images_original = None
    for idx, (image, _) in enumerate(ds_original):
        if idx == 0:
            images_original = image
        else:
            images_original = np.append(images_original, image, axis=0)

    # MixUp Images
    data1 = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)

    one_hot_op = data_trans.OneHot(num_classes=10)
    data1 = data1.map(input_columns=["label"], operations=one_hot_op)
    mixup_batch_op = vision.MixUpBatch()
    data1 = data1.batch(5, drop_remainder=True)
    data1 = data1.map(input_columns=["image"], operations=mixup_batch_op)

    with pytest.raises(RuntimeError) as error:
        images_mixup = np.array([])
        for idx, (image, _) in enumerate(data1):
            if idx == 0:
                images_mixup = image
            else:
                images_mixup = np.append(images_mixup, image, axis=0)
    error_message = "Both images and labels columns are required"
    assert error_message in str(error.value)


if __name__ == "__main__":
    test_mixup_batch_success1(plot=True)
    test_mixup_batch_success2(plot=True)
    test_mixup_batch_md5()
    test_mixup_batch_fail1()
    test_mixup_batch_fail2()
    test_mixup_batch_fail3()
