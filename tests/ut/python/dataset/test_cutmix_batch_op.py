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
Testing the CutMixBatch op in DE
"""
import numpy as np
import pytest
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as vision
import mindspore.dataset.transforms.c_transforms as data_trans
import mindspore.dataset.vision.utils as mode
from mindspore import log as logger
from util import save_and_check_md5, diff_mse, visualize_list, config_get_set_seed, \
    config_get_set_num_parallel_workers

DATA_DIR = "../data/dataset/testCifar10Data"
DATA_DIR2 = "../data/dataset/testImageNetData2/train/"
DATA_DIR3 = "../data/dataset/testCelebAData/"

GENERATE_GOLDEN = False


def test_cutmix_batch_success1(plot=False):
    """
    Test CutMixBatch op with specified alpha and prob parameters on a batch of CHW images
    """
    logger.info("test_cutmix_batch_success1")
    # Original Images
    ds_original = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)
    ds_original = ds_original.batch(5, drop_remainder=True)

    images_original = None
    for idx, (image, _) in enumerate(ds_original):
        if idx == 0:
            images_original = image.asnumpy()
        else:
            images_original = np.append(images_original, image.asnumpy(), axis=0)

    # CutMix Images
    data1 = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)
    hwc2chw_op = vision.HWC2CHW()
    data1 = data1.map(operations=hwc2chw_op, input_columns=["image"])
    one_hot_op = data_trans.OneHot(num_classes=10)
    data1 = data1.map(operations=one_hot_op, input_columns=["label"])
    cutmix_batch_op = vision.CutMixBatch(mode.ImageBatchFormat.NCHW, 2.0, 0.5)
    data1 = data1.batch(5, drop_remainder=True)
    data1 = data1.map(operations=cutmix_batch_op, input_columns=["image", "label"])

    images_cutmix = None
    for idx, (image, _) in enumerate(data1):
        if idx == 0:
            images_cutmix = image.asnumpy().transpose(0, 2, 3, 1)
        else:
            images_cutmix = np.append(images_cutmix, image.asnumpy().transpose(0, 2, 3, 1), axis=0)
    if plot:
        visualize_list(images_original, images_cutmix)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_cutmix[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))


def test_cutmix_batch_success2(plot=False):
    """
    Test CutMixBatch op with default values for alpha and prob on a batch of rescaled HWC images
    """
    logger.info("test_cutmix_batch_success2")

    # Original Images
    ds_original = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)
    ds_original = ds_original.batch(5, drop_remainder=True)

    images_original = None
    for idx, (image, _) in enumerate(ds_original):
        if idx == 0:
            images_original = image.asnumpy()
        else:
            images_original = np.append(images_original, image.asnumpy(), axis=0)

    # CutMix Images
    data1 = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)
    one_hot_op = data_trans.OneHot(num_classes=10)
    data1 = data1.map(operations=one_hot_op, input_columns=["label"])
    rescale_op = vision.Rescale((1.0 / 255.0), 0.0)
    data1 = data1.map(operations=rescale_op, input_columns=["image"])
    cutmix_batch_op = vision.CutMixBatch(mode.ImageBatchFormat.NHWC)
    data1 = data1.batch(5, drop_remainder=True)
    data1 = data1.map(operations=cutmix_batch_op, input_columns=["image", "label"])

    images_cutmix = None
    for idx, (image, _) in enumerate(data1):
        if idx == 0:
            images_cutmix = image.asnumpy()
        else:
            images_cutmix = np.append(images_cutmix, image.asnumpy(), axis=0)
    if plot:
        visualize_list(images_original, images_cutmix)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_cutmix[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))


def test_cutmix_batch_success3(plot=False):
    """
    Test CutMixBatch op with default values for alpha and prob on a batch of HWC images on ImageFolderDataset
    """
    logger.info("test_cutmix_batch_success3")

    ds_original = ds.ImageFolderDataset(dataset_dir=DATA_DIR2, shuffle=False)
    decode_op = vision.Decode()
    ds_original = ds_original.map(operations=[decode_op], input_columns=["image"])
    resize_op = vision.Resize([224, 224])
    ds_original = ds_original.map(operations=[resize_op], input_columns=["image"])
    ds_original = ds_original.batch(4, pad_info={}, drop_remainder=True)

    images_original = None
    for idx, (image, _) in enumerate(ds_original):
        if idx == 0:
            images_original = image.asnumpy()
        else:
            images_original = np.append(images_original, image.asnumpy(), axis=0)

    # CutMix Images
    data1 = ds.ImageFolderDataset(dataset_dir=DATA_DIR2, shuffle=False)

    decode_op = vision.Decode()
    data1 = data1.map(operations=[decode_op], input_columns=["image"])

    resize_op = vision.Resize([224, 224])
    data1 = data1.map(operations=[resize_op], input_columns=["image"])

    one_hot_op = data_trans.OneHot(num_classes=10)
    data1 = data1.map(operations=one_hot_op, input_columns=["label"])

    cutmix_batch_op = vision.CutMixBatch(mode.ImageBatchFormat.NHWC)
    data1 = data1.batch(4, pad_info={}, drop_remainder=True)
    data1 = data1.map(operations=cutmix_batch_op, input_columns=["image", "label"])

    images_cutmix = None
    for idx, (image, _) in enumerate(data1):
        if idx == 0:
            images_cutmix = image.asnumpy()
        else:
            images_cutmix = np.append(images_cutmix, image.asnumpy(), axis=0)
    if plot:
        visualize_list(images_original, images_cutmix)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_cutmix[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))


def test_cutmix_batch_success4(plot=False):
    """
    Test CutMixBatch on a dataset where OneHot returns a 2D vector
    """
    logger.info("test_cutmix_batch_success4")

    ds_original = ds.CelebADataset(DATA_DIR3, shuffle=False)
    decode_op = vision.Decode()
    ds_original = ds_original.map(operations=[decode_op], input_columns=["image"])
    resize_op = vision.Resize([224, 224])
    ds_original = ds_original.map(operations=[resize_op], input_columns=["image"])
    ds_original = ds_original.batch(2, drop_remainder=True)

    images_original = None
    for idx, (image, _) in enumerate(ds_original):
        if idx == 0:
            images_original = image.asnumpy()
        else:
            images_original = np.append(images_original, image.asnumpy(), axis=0)

    # CutMix Images
    data1 = ds.CelebADataset(dataset_dir=DATA_DIR3, shuffle=False)

    decode_op = vision.Decode()
    data1 = data1.map(operations=[decode_op], input_columns=["image"])

    resize_op = vision.Resize([224, 224])
    data1 = data1.map(operations=[resize_op], input_columns=["image"])

    one_hot_op = data_trans.OneHot(num_classes=100)
    data1 = data1.map(operations=one_hot_op, input_columns=["attr"])

    cutmix_batch_op = vision.CutMixBatch(mode.ImageBatchFormat.NHWC, 0.5, 0.9)
    data1 = data1.batch(2, drop_remainder=True)
    data1 = data1.map(operations=cutmix_batch_op, input_columns=["image", "attr"])

    images_cutmix = None
    for idx, (image, _) in enumerate(data1):
        if idx == 0:
            images_cutmix = image.asnumpy()
        else:
            images_cutmix = np.append(images_cutmix, image.asnumpy(), axis=0)
    if plot:
        visualize_list(images_original, images_cutmix)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_cutmix[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))


def test_cutmix_batch_nhwc_md5():
    """
    Test CutMixBatch on a batch of HWC images with MD5:
    """
    logger.info("test_cutmix_batch_nhwc_md5")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # CutMixBatch Images
    data = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)

    one_hot_op = data_trans.OneHot(num_classes=10)
    data = data.map(operations=one_hot_op, input_columns=["label"])
    cutmix_batch_op = vision.CutMixBatch(mode.ImageBatchFormat.NHWC)
    data = data.batch(5, drop_remainder=True)
    data = data.map(operations=cutmix_batch_op, input_columns=["image", "label"])

    filename = "cutmix_batch_c_nhwc_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_cutmix_batch_nchw_md5():
    """
    Test CutMixBatch on a batch of CHW images with MD5:
    """
    logger.info("test_cutmix_batch_nchw_md5")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # CutMixBatch Images
    data = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)
    hwc2chw_op = vision.HWC2CHW()
    data = data.map(operations=hwc2chw_op, input_columns=["image"])
    one_hot_op = data_trans.OneHot(num_classes=10)
    data = data.map(operations=one_hot_op, input_columns=["label"])
    cutmix_batch_op = vision.CutMixBatch(mode.ImageBatchFormat.NCHW)
    data = data.batch(5, drop_remainder=True)
    data = data.map(operations=cutmix_batch_op, input_columns=["image", "label"])

    filename = "cutmix_batch_c_nchw_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_cutmix_batch_fail1():
    """
    Test CutMixBatch Fail 1
    We expect this to fail because the images and labels are not batched
    """
    logger.info("test_cutmix_batch_fail1")

    # CutMixBatch Images
    data1 = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)

    one_hot_op = data_trans.OneHot(num_classes=10)
    data1 = data1.map(operations=one_hot_op, input_columns=["label"])
    cutmix_batch_op = vision.CutMixBatch(mode.ImageBatchFormat.NHWC)
    with pytest.raises(RuntimeError) as error:
        data1 = data1.map(operations=cutmix_batch_op, input_columns=["image", "label"])
        for idx, (image, _) in enumerate(data1):
            if idx == 0:
                images_cutmix = image.asnumpy()
            else:
                images_cutmix = np.append(images_cutmix, image.asnumpy(), axis=0)
        error_message = "You must make sure images are HWC or CHW and batch "
        assert error_message in str(error.value)


def test_cutmix_batch_fail2():
    """
    Test CutMixBatch Fail 2
    We expect this to fail because alpha is negative
    """
    logger.info("test_cutmix_batch_fail2")

    # CutMixBatch Images
    data1 = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)

    one_hot_op = data_trans.OneHot(num_classes=10)
    data1 = data1.map(operations=one_hot_op, input_columns=["label"])
    with pytest.raises(ValueError) as error:
        vision.CutMixBatch(mode.ImageBatchFormat.NHWC, -1)
        error_message = "Input is not within the required interval"
        assert error_message in str(error.value)


def test_cutmix_batch_fail3():
    """
    Test CutMixBatch Fail 2
    We expect this to fail because prob is larger than 1
    """
    logger.info("test_cutmix_batch_fail3")

    # CutMixBatch Images
    data1 = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)

    one_hot_op = data_trans.OneHot(num_classes=10)
    data1 = data1.map(operations=one_hot_op, input_columns=["label"])
    with pytest.raises(ValueError) as error:
        vision.CutMixBatch(mode.ImageBatchFormat.NHWC, 1, 2)
        error_message = "Input is not within the required interval"
        assert error_message in str(error.value)


def test_cutmix_batch_fail4():
    """
    Test CutMixBatch Fail 2
    We expect this to fail because prob is negative
    """
    logger.info("test_cutmix_batch_fail4")

    # CutMixBatch Images
    data1 = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)

    one_hot_op = data_trans.OneHot(num_classes=10)
    data1 = data1.map(operations=one_hot_op, input_columns=["label"])
    with pytest.raises(ValueError) as error:
        vision.CutMixBatch(mode.ImageBatchFormat.NHWC, 1, -1)
        error_message = "Input is not within the required interval"
        assert error_message in str(error.value)


def test_cutmix_batch_fail5():
    """
    Test CutMixBatch op
    We expect this to fail because label column is not passed to cutmix_batch
    """
    logger.info("test_cutmix_batch_fail5")

    # CutMixBatch Images
    data1 = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)

    one_hot_op = data_trans.OneHot(num_classes=10)
    data1 = data1.map(operations=one_hot_op, input_columns=["label"])
    cutmix_batch_op = vision.CutMixBatch(mode.ImageBatchFormat.NHWC)
    data1 = data1.batch(5, drop_remainder=True)
    data1 = data1.map(operations=cutmix_batch_op, input_columns=["image"])

    with pytest.raises(RuntimeError) as error:
        images_cutmix = np.array([])
        for idx, (image, _) in enumerate(data1):
            if idx == 0:
                images_cutmix = image.asnumpy()
            else:
                images_cutmix = np.append(images_cutmix, image.asnumpy(), axis=0)
    error_message = "both image and label columns are required"
    assert error_message in str(error.value)


def test_cutmix_batch_fail6():
    """
    Test CutMixBatch op
    We expect this to fail because image_batch_format passed to CutMixBatch doesn't match the format of the images
    """
    logger.info("test_cutmix_batch_fail6")

    # CutMixBatch Images
    data1 = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)

    one_hot_op = data_trans.OneHot(num_classes=10)
    data1 = data1.map(operations=one_hot_op, input_columns=["label"])
    cutmix_batch_op = vision.CutMixBatch(mode.ImageBatchFormat.NCHW)
    data1 = data1.batch(5, drop_remainder=True)
    data1 = data1.map(operations=cutmix_batch_op, input_columns=["image", "label"])

    with pytest.raises(RuntimeError) as error:
        images_cutmix = np.array([])
        for idx, (image, _) in enumerate(data1):
            if idx == 0:
                images_cutmix = image.asnumpy()
            else:
                images_cutmix = np.append(images_cutmix, image.asnumpy(), axis=0)
    error_message = "image doesn't match the NCHW format."
    assert error_message in str(error.value)


def test_cutmix_batch_fail7():
    """
    Test CutMixBatch op
    We expect this to fail because labels are not in one-hot format
    """
    logger.info("test_cutmix_batch_fail7")

    # CutMixBatch Images
    data1 = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)

    cutmix_batch_op = vision.CutMixBatch(mode.ImageBatchFormat.NHWC)
    data1 = data1.batch(5, drop_remainder=True)
    data1 = data1.map(operations=cutmix_batch_op, input_columns=["image", "label"])

    with pytest.raises(RuntimeError) as error:
        images_cutmix = np.array([])
        for idx, (image, _) in enumerate(data1):
            if idx == 0:
                images_cutmix = image.asnumpy()
            else:
                images_cutmix = np.append(images_cutmix, image.asnumpy(), axis=0)
    error_message = "wrong labels shape. The second column (labels) must have a shape of NC or NLC"
    assert error_message in str(error.value)


def test_cutmix_batch_fail8():
    """
    Test CutMixBatch Fail 8
    We expect this to fail because alpha is zero
    """
    logger.info("test_cutmix_batch_fail8")

    # CutMixBatch Images
    data1 = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)

    one_hot_op = data_trans.OneHot(num_classes=10)
    data1 = data1.map(operations=one_hot_op, input_columns=["label"])
    with pytest.raises(ValueError) as error:
        vision.CutMixBatch(mode.ImageBatchFormat.NHWC, 0.0)
        error_message = "Input is not within the required interval"
        assert error_message in str(error.value)


if __name__ == "__main__":
    test_cutmix_batch_success1(plot=True)
    test_cutmix_batch_success2(plot=True)
    test_cutmix_batch_success3(plot=True)
    test_cutmix_batch_success4(plot=True)
    test_cutmix_batch_nchw_md5()
    test_cutmix_batch_nhwc_md5()
    test_cutmix_batch_fail1()
    test_cutmix_batch_fail2()
    test_cutmix_batch_fail3()
    test_cutmix_batch_fail4()
    test_cutmix_batch_fail5()
    test_cutmix_batch_fail6()
    test_cutmix_batch_fail7()
    test_cutmix_batch_fail8()
