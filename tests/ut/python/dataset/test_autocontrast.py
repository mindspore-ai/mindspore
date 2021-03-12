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
Testing AutoContrast op in DE
"""
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.transforms.py_transforms
import mindspore.dataset.vision.py_transforms as F
import mindspore.dataset.vision.c_transforms as C
from mindspore import log as logger
from util import visualize_list, visualize_one_channel_dataset, diff_mse, save_and_check_md5

DATA_DIR = "../data/dataset/testImageNetData/train/"
MNIST_DATA_DIR = "../data/dataset/testMnistData"

GENERATE_GOLDEN = False


def test_auto_contrast_py(plot=False):
    """
    Test AutoContrast
    """
    logger.info("Test AutoContrast Python Op")

    # Original Images
    data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transforms_original = mindspore.dataset.transforms.py_transforms.Compose([F.Decode(),
                                                                              F.Resize((224, 224)),
                                                                              F.ToTensor()])

    ds_original = data_set.map(operations=transforms_original, input_columns="image")

    ds_original = ds_original.batch(512)

    for idx, (image, _) in enumerate(ds_original):
        if idx == 0:
            images_original = np.transpose(image.asnumpy(), (0, 2, 3, 1))
        else:
            images_original = np.append(images_original,
                                        np.transpose(image.asnumpy(), (0, 2, 3, 1)),
                                        axis=0)

            # AutoContrast Images
    data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transforms_auto_contrast = \
        mindspore.dataset.transforms.py_transforms.Compose([F.Decode(),
                                                            F.Resize((224, 224)),
                                                            F.AutoContrast(cutoff=10.0, ignore=[10, 20]),
                                                            F.ToTensor()])

    ds_auto_contrast = data_set.map(operations=transforms_auto_contrast, input_columns="image")

    ds_auto_contrast = ds_auto_contrast.batch(512)

    for idx, (image, _) in enumerate(ds_auto_contrast):
        if idx == 0:
            images_auto_contrast = np.transpose(image.asnumpy(), (0, 2, 3, 1))
        else:
            images_auto_contrast = np.append(images_auto_contrast,
                                             np.transpose(image.asnumpy(), (0, 2, 3, 1)),
                                             axis=0)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_auto_contrast[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))

    # Compare with expected md5 from images
    filename = "autocontrast_01_result_py.npz"
    save_and_check_md5(ds_auto_contrast, filename, generate_golden=GENERATE_GOLDEN)

    if plot:
        visualize_list(images_original, images_auto_contrast)


def test_auto_contrast_c(plot=False):
    """
    Test AutoContrast C Op
    """
    logger.info("Test AutoContrast C Op")

    # AutoContrast Images
    data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
    data_set = data_set.map(operations=[C.Decode(), C.Resize((224, 224))], input_columns=["image"])
    python_op = F.AutoContrast(cutoff=10.0, ignore=[10, 20])
    c_op = C.AutoContrast(cutoff=10.0, ignore=[10, 20])
    transforms_op = mindspore.dataset.transforms.py_transforms.Compose([lambda img: F.ToPIL()(img.astype(np.uint8)),
                                                                        python_op,
                                                                        np.array])

    ds_auto_contrast_py = data_set.map(operations=transforms_op, input_columns="image")

    ds_auto_contrast_py = ds_auto_contrast_py.batch(512)

    for idx, (image, _) in enumerate(ds_auto_contrast_py):
        if idx == 0:
            images_auto_contrast_py = image.asnumpy()
        else:
            images_auto_contrast_py = np.append(images_auto_contrast_py,
                                                image.asnumpy(),
                                                axis=0)

    data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
    data_set = data_set.map(operations=[C.Decode(), C.Resize((224, 224))], input_columns=["image"])

    ds_auto_contrast_c = data_set.map(operations=c_op, input_columns="image")

    ds_auto_contrast_c = ds_auto_contrast_c.batch(512)

    for idx, (image, _) in enumerate(ds_auto_contrast_c):
        if idx == 0:
            images_auto_contrast_c = image.asnumpy()
        else:
            images_auto_contrast_c = np.append(images_auto_contrast_c,
                                               image.asnumpy(),
                                               axis=0)

    num_samples = images_auto_contrast_c.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_auto_contrast_c[i], images_auto_contrast_py[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))
    np.testing.assert_equal(np.mean(mse), 0.0)

    # Compare with expected md5 from images
    filename = "autocontrast_01_result_c.npz"
    save_and_check_md5(ds_auto_contrast_c, filename, generate_golden=GENERATE_GOLDEN)

    if plot:
        visualize_list(images_auto_contrast_c, images_auto_contrast_py, visualize_mode=2)


def test_auto_contrast_one_channel_c(plot=False):
    """
    Test AutoContrast C op with one channel
    """
    logger.info("Test AutoContrast C Op With One Channel Images")

    # AutoContrast Images
    data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
    data_set = data_set.map(operations=[C.Decode(), C.Resize((224, 224))], input_columns=["image"])
    python_op = F.AutoContrast()
    c_op = C.AutoContrast()
    # not using F.ToTensor() since it converts to floats
    transforms_op = mindspore.dataset.transforms.py_transforms.Compose(
        [lambda img: (np.array(img)[:, :, 0]).astype(np.uint8),
         F.ToPIL(),
         python_op,
         np.array])

    ds_auto_contrast_py = data_set.map(operations=transforms_op, input_columns="image")

    ds_auto_contrast_py = ds_auto_contrast_py.batch(512)

    for idx, (image, _) in enumerate(ds_auto_contrast_py):
        if idx == 0:
            images_auto_contrast_py = image.asnumpy()
        else:
            images_auto_contrast_py = np.append(images_auto_contrast_py,
                                                image.asnumpy(),
                                                axis=0)

    data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
    data_set = data_set.map(operations=[C.Decode(), C.Resize((224, 224)), lambda img: np.array(img[:, :, 0])],
                            input_columns=["image"])

    ds_auto_contrast_c = data_set.map(operations=c_op, input_columns="image")

    ds_auto_contrast_c = ds_auto_contrast_c.batch(512)

    for idx, (image, _) in enumerate(ds_auto_contrast_c):
        if idx == 0:
            images_auto_contrast_c = image.asnumpy()
        else:
            images_auto_contrast_c = np.append(images_auto_contrast_c,
                                               image.asnumpy(),
                                               axis=0)

    num_samples = images_auto_contrast_c.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_auto_contrast_c[i], images_auto_contrast_py[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))
    np.testing.assert_equal(np.mean(mse), 0.0)

    if plot:
        visualize_list(images_auto_contrast_c, images_auto_contrast_py, visualize_mode=2)


def test_auto_contrast_mnist_c(plot=False):
    """
    Test AutoContrast C op with MNIST dataset (Grayscale images)
    """
    logger.info("Test AutoContrast C Op With MNIST Images")
    data_set = ds.MnistDataset(dataset_dir=MNIST_DATA_DIR, num_samples=2, shuffle=False)
    ds_auto_contrast_c = data_set.map(operations=C.AutoContrast(cutoff=1, ignore=(0, 255)), input_columns="image")
    ds_orig = ds.MnistDataset(dataset_dir=MNIST_DATA_DIR, num_samples=2, shuffle=False)

    images = []
    images_trans = []
    labels = []
    for _, (data_orig, data_trans) in enumerate(zip(ds_orig, ds_auto_contrast_c)):
        image_orig, label_orig = data_orig
        image_trans, _ = data_trans
        images.append(image_orig.asnumpy())
        labels.append(label_orig.asnumpy())
        images_trans.append(image_trans.asnumpy())

    # Compare with expected md5 from images
    filename = "autocontrast_mnist_result_c.npz"
    save_and_check_md5(ds_auto_contrast_c, filename, generate_golden=GENERATE_GOLDEN)

    if plot:
        visualize_one_channel_dataset(images, images_trans, labels)


def test_auto_contrast_invalid_ignore_param_c():
    """
    Test AutoContrast C Op with invalid ignore parameter
    """
    logger.info("Test AutoContrast C Op with invalid ignore parameter")
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(operations=[C.Decode(),
                                            C.Resize((224, 224)),
                                            lambda img: np.array(img[:, :, 0])], input_columns=["image"])
        # invalid ignore
        data_set = data_set.map(operations=C.AutoContrast(ignore=255.5), input_columns="image")
    except TypeError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "Argument ignore with value 255.5 is not of type" in str(error)
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(operations=[C.Decode(), C.Resize((224, 224)),
                                            lambda img: np.array(img[:, :, 0])], input_columns=["image"])
        # invalid ignore
        data_set = data_set.map(operations=C.AutoContrast(ignore=(10, 100)), input_columns="image")
    except TypeError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "Argument ignore with value (10,100) is not of type" in str(error)


def test_auto_contrast_invalid_cutoff_param_c():
    """
    Test AutoContrast C Op with invalid cutoff parameter
    """
    logger.info("Test AutoContrast C Op with invalid cutoff parameter")
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(operations=[C.Decode(),
                                            C.Resize((224, 224)),
                                            lambda img: np.array(img[:, :, 0])], input_columns=["image"])
        # invalid ignore
        data_set = data_set.map(operations=C.AutoContrast(cutoff=-10.0), input_columns="image")
    except ValueError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "Input cutoff is not within the required interval of [0, 50)." in str(error)
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(operations=[C.Decode(),
                                            C.Resize((224, 224)),
                                            lambda img: np.array(img[:, :, 0])], input_columns=["image"])
        # invalid ignore
        data_set = data_set.map(operations=C.AutoContrast(cutoff=120.0), input_columns="image")
    except ValueError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "Input cutoff is not within the required interval of [0, 50)." in str(error)


def test_auto_contrast_invalid_ignore_param_py():
    """
    Test AutoContrast python Op with invalid ignore parameter
    """
    logger.info("Test AutoContrast python Op with invalid ignore parameter")
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(operations=[mindspore.dataset.transforms.py_transforms.Compose([F.Decode(),
                                                                                                F.Resize((224, 224)),
                                                                                                F.AutoContrast(
                                                                                                    ignore=255.5),
                                                                                                F.ToTensor()])],
                                input_columns=["image"])
    except TypeError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "Argument ignore with value 255.5 is not of type" in str(error)
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(operations=[mindspore.dataset.transforms.py_transforms.Compose([F.Decode(),
                                                                                                F.Resize((224, 224)),
                                                                                                F.AutoContrast(
                                                                                                    ignore=(10, 100)),
                                                                                                F.ToTensor()])],
                                input_columns=["image"])
    except TypeError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "Argument ignore with value (10,100) is not of type" in str(error)


def test_auto_contrast_invalid_cutoff_param_py():
    """
    Test AutoContrast python Op with invalid cutoff parameter
    """
    logger.info("Test AutoContrast python Op with invalid cutoff parameter")
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(operations=[mindspore.dataset.transforms.py_transforms.Compose([F.Decode(),
                                                                                                F.Resize((224, 224)),
                                                                                                F.AutoContrast(
                                                                                                    cutoff=-10.0),
                                                                                                F.ToTensor()])],
                                input_columns=["image"])
    except ValueError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "Input cutoff is not within the required interval of [0, 50)." in str(error)
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(
            operations=[mindspore.dataset.transforms.py_transforms.Compose([F.Decode(),
                                                                            F.Resize((224, 224)),
                                                                            F.AutoContrast(cutoff=120.0),
                                                                            F.ToTensor()])],
            input_columns=["image"])
    except ValueError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "Input cutoff is not within the required interval of [0, 50)." in str(error)


if __name__ == "__main__":
    test_auto_contrast_py(plot=True)
    test_auto_contrast_c(plot=True)
    test_auto_contrast_one_channel_c(plot=True)
    test_auto_contrast_mnist_c(plot=True)
    test_auto_contrast_invalid_ignore_param_c()
    test_auto_contrast_invalid_ignore_param_py()
    test_auto_contrast_invalid_cutoff_param_c()
    test_auto_contrast_invalid_cutoff_param_py()
