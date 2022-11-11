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
"""
Test CutMixBatch in DE
"""
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
from mindspore import log as logger
from util import save_and_check_md5, diff_mse, visualize_list, config_get_set_seed, \
    config_get_set_num_parallel_workers

DATA_DIR = "../data/dataset/testCifar10Data"
DATA_DIR2 = "../data/dataset/testImageNetData2/train/"
DATA_DIR3 = "../data/dataset/testCelebAData/"

GENERATE_GOLDEN = False


def test_cut_mix_batch_specified_param_on_chw_image(plot=False):
    """
    Feature: CutMixBatch
    Description: Test CutMixBatch with specified alpha and prob parameter on a batch of CHW images
    Expectation: Output is equal to the expected output
    """
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
    one_hot_op = transforms.OneHot(num_classes=10)
    data1 = data1.map(operations=one_hot_op, input_columns=["label"])
    cutmix_batch_op = vision.CutMixBatch(vision.ImageBatchFormat.NCHW, 2.0, 0.5)
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


def test_cut_mix_batch_on_hwc_image(plot=False):
    """
    Feature: CutMixBatch
    Description: Test CutMixBatch with default params on a batch of HWC images
    Expectation: Result is as expected
    """
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
    one_hot_op = transforms.OneHot(num_classes=10)
    data1 = data1.map(operations=one_hot_op, input_columns=["label"])
    rescale_op = vision.Rescale((1.0 / 255.0), 0.0)
    data1 = data1.map(operations=rescale_op, input_columns=["image"])
    cutmix_batch_op = vision.CutMixBatch(vision.ImageBatchFormat.NHWC)
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


def test_cut_mix_batch_on_image_folder(plot=False):
    """
    Feature: CutMixBatch
    Description: Test CutMixBatch with default values for alpha and prob on a batch of HWC images on ImagesFolderDataset
    Expectation: Output is equal to the expected output
    """
    ds_original = ds.ImageFolderDataset(dataset_dir=DATA_DIR2, shuffle=False)
    decode_op = vision.Decode()
    ds_original = ds_original.map(operations=[decode_op], input_columns=["image"])
    resize_op = vision.Resize([224, 224])
    ds_original = ds_original.map(operations=[resize_op], input_columns=["image"])
    ds_original = ds_original.batch(4, drop_remainder=True)

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

    one_hot_op = transforms.OneHot(num_classes=10)
    data1 = data1.map(operations=one_hot_op, input_columns=["label"])

    cutmix_batch_op = vision.CutMixBatch(vision.ImageBatchFormat.NHWC)
    data1 = data1.batch(4, drop_remainder=True)
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


def test_cut_mix_batch_on_2d_label(plot=False):
    """
    Feature: CutMixBatch
    Description: Test CutMixBatch on a dataset where OneHot returns a 2D vector
    Expectation: Output is equal to the expected output
    """
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

    one_hot_op = transforms.OneHot(num_classes=100)
    data1 = data1.map(operations=one_hot_op, input_columns=["attr"])

    cutmix_batch_op = vision.CutMixBatch(vision.ImageBatchFormat.NHWC, 0.5, 0.9)
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


def test_cut_mix_batch_accuracy_on_hwc_image():
    """
    Feature: CutMixBatch
    Description: Test CutMixBatch on a batch of HWC images with md5 comparison check
    Expectation: Passes the md5 check test
    """
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # CutMixBatch Images
    data = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)

    one_hot_op = transforms.OneHot(num_classes=10)
    data = data.map(operations=one_hot_op, input_columns=["label"])
    cutmix_batch_op = vision.CutMixBatch(vision.ImageBatchFormat.NHWC)
    data = data.batch(5, drop_remainder=True)
    data = data.map(operations=cutmix_batch_op, input_columns=["image", "label"])

    filename = "cutmix_batch_c_nhwc_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_cut_mix_batch_accuracy_on_chw_image():
    """
    Feature: CutMixBatch
    Description: Test CutMixBatch on a batch of CHW images with md5 comparison check
    Expectation: Passes the md5 check test
    """
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # CutMixBatch Images
    data = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)
    hwc2chw_op = vision.HWC2CHW()
    data = data.map(operations=hwc2chw_op, input_columns=["image"])
    one_hot_op = transforms.OneHot(num_classes=10)
    data = data.map(operations=one_hot_op, input_columns=["label"])
    cutmix_batch_op = vision.CutMixBatch(vision.ImageBatchFormat.NCHW)
    data = data.batch(5, drop_remainder=True)
    data = data.map(operations=cutmix_batch_op, input_columns=["image", "label"])

    filename = "cutmix_batch_c_nchw_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_cut_mix_batch_float_label():
    """
    Feature: CutMixBatch
    Description: Test CutMixBatch with label in type of float
    Expectation: Output is as expected
    """
    original_seed = config_get_set_seed(0)

    image = np.random.randint(0, 255, (3, 28, 28, 1), dtype=np.uint8)
    label = np.random.randint(0, 5, (3, 1))
    decode_label = transforms.OneHot(5)(label)
    float_label = transforms.TypeCast(float)(decode_label)
    _, mix_label = vision.CutMixBatch(vision.ImageBatchFormat.NHWC)(image, float_label)
    expected_label = np.array([[0., 0.6734694, 0., 0., 0.32653058],
                               [0., 0., 0., 0., 1.],
                               [0., 0.38137758, 0., 0., 0.6186224]])
    np.testing.assert_almost_equal(mix_label, expected_label)

    ds.config.set_seed(original_seed)


def test_cut_mix_batch_then_mix_up_batch():
    """
    Feature: CutMixBatch
    Description: Test CutMixBatch called twice
    Expectation: Output is as expected
    """
    original_seed = config_get_set_seed(2)

    dataset = ds.Cifar10Dataset(DATA_DIR, num_samples=3, shuffle=False)
    one_hot = transforms.OneHot(num_classes=10)
    dataset = dataset.map(operations=one_hot, input_columns=["label"])
    cut_mix_batch = vision.CutMixBatch(vision.ImageBatchFormat.NHWC, 2.0, 0.5)
    mix_up_batch = vision.MixUpBatch()
    dataset = dataset.batch(3, drop_remainder=False)
    dataset = dataset.map(operations=cut_mix_batch, input_columns=["image", "label"])
    dataset = dataset.map(operations=mix_up_batch, input_columns=["image", "label"])

    expected_label = np.array([[0.10036002, 0.02589141, 0.87374854, 0., 0., 0., 0., 0., 0., 0.],
                               [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                               [0.69456184, 0.17918672, 0.12625143, 0., 0., 0., 0., 0., 0., 0.]])
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        np.testing.assert_almost_equal(item["label"], expected_label)

    ds.config.set_seed(original_seed)


def test_cut_mix_batch_without_batch():
    """
    Feature: CutMixBatch
    Description: Test CutMixBatch where images and labels are not batched
    Expectation: Correct error is raised as expected
    """
    # CutMixBatch Images
    data1 = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)

    one_hot_op = transforms.OneHot(num_classes=10)
    data1 = data1.map(operations=one_hot_op, input_columns=["label"])
    cutmix_batch_op = vision.CutMixBatch(vision.ImageBatchFormat.NHWC)
    with pytest.raises(RuntimeError) as error:
        data1 = data1.map(operations=cutmix_batch_op, input_columns=["image", "label"])
        for idx, (image, _) in enumerate(data1):
            if idx == 0:
                images_cutmix = image.asnumpy()
            else:
                images_cutmix = np.append(images_cutmix, image.asnumpy(), axis=0)
        error_message = "input image is not in shape of <B,H,W,C> or <B,C,H,W>"
        assert error_message in str(error.value)
        batch_suggestion = "You may need to perform Batch first"
        assert batch_suggestion in str(error.value)


def test_cut_mix_batch_invalid_alpha():
    """
    Feature: CutMixBatch
    Description: Test CutMixBatch where alpha is invalid
    Expectation: Correct error is raised as expected
    """
    with pytest.raises(ValueError) as error:
        _ = vision.CutMixBatch(vision.ImageBatchFormat.NHWC, -1)
        error_message = "Input is not within the required interval"
        assert error_message in str(error.value)

    with pytest.raises(ValueError) as error:
        _ = vision.CutMixBatch(vision.ImageBatchFormat.NHWC, 0.0)
        error_message = "Input is not within the required interval"
        assert error_message in str(error.value)


def test_cut_mix_batch_invalid_prob():
    """
    Feature: CutMixBatch
    Description: Test CutMixBatch where prob is invalid
    Expectation: Correct error is raised as expected
    """
    with pytest.raises(ValueError) as error:
        _ = vision.CutMixBatch(vision.ImageBatchFormat.NHWC, 1, 2)
        error_message = "Input is not within the required interval"
        assert error_message in str(error.value)

    with pytest.raises(ValueError) as error:
        _ = vision.CutMixBatch(vision.ImageBatchFormat.NHWC, 1, -1)
        error_message = "Input is not within the required interval"
        assert error_message in str(error.value)


def test_cut_mix_batch_invalid_column_size():
    """
    Feature: CutMixBatch
    Description: Test CutMixBatch where label column is not passed to CutMixBatch
    Expectation: Correct error is raised as expected
    """
    # CutMixBatch Images
    data1 = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)

    one_hot_op = transforms.OneHot(num_classes=10)
    data1 = data1.map(operations=one_hot_op, input_columns=["label"])
    cutmix_batch_op = vision.CutMixBatch(vision.ImageBatchFormat.NHWC)
    data1 = data1.batch(5, drop_remainder=True)
    data1 = data1.map(operations=cutmix_batch_op, input_columns=["image"])

    with pytest.raises(RuntimeError) as error:
        images_cutmix = np.array([])
        for idx, (image, _) in enumerate(data1):
            if idx == 0:
                images_cutmix = image.asnumpy()
            else:
                images_cutmix = np.append(images_cutmix, image.asnumpy(), axis=0)
    error_message = "input should have 2 columns (image and label)"
    assert error_message in str(error.value)


def test_cut_mix_batch_invalid_channel():
    """
    Feature: CutMixBatch
    Description: Test CutMixBatch where image passed to CutMixBatch doesn't match the required format
    Expectation: Correct error is raised as expected
    """
    # CutMixBatch Images
    data1 = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)

    one_hot_op = transforms.OneHot(num_classes=10)
    data1 = data1.map(operations=one_hot_op, input_columns=["label"])
    cutmix_batch_op = vision.CutMixBatch(vision.ImageBatchFormat.NCHW)
    data1 = data1.batch(5, drop_remainder=True)
    data1 = data1.map(operations=cutmix_batch_op, input_columns=["image", "label"])

    with pytest.raises(RuntimeError) as error:
        images_cutmix = np.array([])
        for idx, (image, _) in enumerate(data1):
            if idx == 0:
                images_cutmix = image.asnumpy()
            else:
                images_cutmix = np.append(images_cutmix, image.asnumpy(), axis=0)
    error_message = "input image is not in channel of 1 or 3"
    assert error_message in str(error.value)


def test_cut_mix_batch_without_one_hot():
    """
    Feature: CutMixBatch
    Description: Test CutMixBatch where labels are not in one-hot format
    Expectation: Correct error is raised as expected
    """
    # CutMixBatch Images
    data1 = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)

    cutmix_batch_op = vision.CutMixBatch(vision.ImageBatchFormat.NHWC)
    data1 = data1.batch(5, drop_remainder=True)
    data1 = data1.map(operations=cutmix_batch_op, input_columns=["image", "label"])

    with pytest.raises(RuntimeError) as error:
        images_cutmix = np.array([])
        for idx, (image, _) in enumerate(data1):
            if idx == 0:
                images_cutmix = image.asnumpy()
            else:
                images_cutmix = np.append(images_cutmix, image.asnumpy(), axis=0)
    error_message = "input label is not in shape of <Batch,Class> or <Batch,Row,Class>"
    assert error_message in str(error.value)
    one_hot_suggession = "You may need to perform OneHot and Batch first"
    assert one_hot_suggession in str(error.value)


def test_cut_mix_batch_unequal_batch_size():
    """
    Feature: CutMixBatch
    Description: Test CutMixBatch where image and label are in different batch sizes
    Expectation: Correct error is raised as expected
    """
    image = np.random.randint(0, 255, (5, 28, 28, 1))
    label = np.random.randint(0, 10, (4, 1))
    decode_label = transforms.OneHot(10)(label)
    with pytest.raises(RuntimeError) as error:
        _ = vision.CutMixBatch(vision.ImageBatchFormat.NHWC)(image, decode_label)
    error_message = "batch sizes of image and label must be the same"
    assert error_message in str(error.value)


def test_cut_mix_batch_invalid_label_type():
    """
    Feature: CutMixBatch
    Description: Test CutMixBatch with label in str type
    Expectation: Error is raised as expected
    """
    image = np.random.randint(0, 255, (3, 28, 28, 1), dtype=np.uint8)
    label = np.array([["one"], ["two"], ["three"]])
    with pytest.raises(RuntimeError) as error:
        _ = vision.CutMixBatch(vision.ImageBatchFormat.NHWC)(image, label)
    error_message = "CutMixBatch: invalid label type, label must be in a numeric type"
    assert error_message in str(error.value)


if __name__ == "__main__":
    test_cut_mix_batch_specified_param_on_chw_image(plot=True)
    test_cut_mix_batch_on_hwc_image(plot=True)
    test_cut_mix_batch_on_image_folder(plot=True)
    test_cut_mix_batch_on_2d_label(plot=True)
    test_cut_mix_batch_accuracy_on_hwc_image()
    test_cut_mix_batch_accuracy_on_chw_image()
    test_cut_mix_batch_float_label()
    test_cut_mix_batch_then_mix_up_batch()
    test_cut_mix_batch_without_batch()
    test_cut_mix_batch_invalid_alpha()
    test_cut_mix_batch_invalid_prob()
    test_cut_mix_batch_invalid_column_size()
    test_cut_mix_batch_invalid_channel()
    test_cut_mix_batch_without_one_hot()
    test_cut_mix_batch_unequal_batch_size()
    test_cut_mix_batch_invalid_label_type()
