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
Testing Equalize op in DE
"""
import numpy as np

import mindspore.dataset.engine as de
import mindspore.dataset.transforms.vision.c_transforms as C
import mindspore.dataset.transforms.vision.py_transforms as F
from mindspore import log as logger
from util import visualize_list, visualize_one_channel_dataset, diff_mse, save_and_check_md5

DATA_DIR = "../data/dataset/testImageNetData/train/"
MNIST_DATA_DIR = "../data/dataset/testMnistData"

GENERATE_GOLDEN = False


def test_equalize_py(plot=False):
    """
    Test Equalize py op
    """
    logger.info("Test Equalize")

    # Original Images
    ds = de.ImageFolderDatasetV2(dataset_dir=DATA_DIR, shuffle=False)

    transforms_original = F.ComposeOp([F.Decode(),
                                       F.Resize((224, 224)),
                                       F.ToTensor()])

    ds_original = ds.map(input_columns="image",
                         operations=transforms_original())

    ds_original = ds_original.batch(512)

    for idx, (image, _) in enumerate(ds_original):
        if idx == 0:
            images_original = np.transpose(image, (0, 2, 3, 1))
        else:
            images_original = np.append(images_original,
                                        np.transpose(image, (0, 2, 3, 1)),
                                        axis=0)

            # Color Equalized Images
    ds = de.ImageFolderDatasetV2(dataset_dir=DATA_DIR, shuffle=False)

    transforms_equalize = F.ComposeOp([F.Decode(),
                                       F.Resize((224, 224)),
                                       F.Equalize(),
                                       F.ToTensor()])

    ds_equalize = ds.map(input_columns="image",
                         operations=transforms_equalize())

    ds_equalize = ds_equalize.batch(512)

    for idx, (image, _) in enumerate(ds_equalize):
        if idx == 0:
            images_equalize = np.transpose(image, (0, 2, 3, 1))
        else:
            images_equalize = np.append(images_equalize,
                                        np.transpose(image, (0, 2, 3, 1)),
                                        axis=0)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_equalize[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))

    if plot:
        visualize_list(images_original, images_equalize)


def test_equalize_c(plot=False):
    """
    Test Equalize Cpp op
    """
    logger.info("Test Equalize cpp op")

    # Original Images
    ds = de.ImageFolderDatasetV2(dataset_dir=DATA_DIR, shuffle=False)

    transforms_original = [C.Decode(), C.Resize(size=[224, 224])]

    ds_original = ds.map(input_columns="image",
                         operations=transforms_original)

    ds_original = ds_original.batch(512)

    for idx, (image, _) in enumerate(ds_original):
        if idx == 0:
            images_original = image
        else:
            images_original = np.append(images_original,
                                        image,
                                        axis=0)

    # Equalize Images
    ds = de.ImageFolderDatasetV2(dataset_dir=DATA_DIR, shuffle=False)

    transform_equalize = [C.Decode(), C.Resize(size=[224, 224]),
                          C.Equalize()]

    ds_equalize = ds.map(input_columns="image",
                         operations=transform_equalize)

    ds_equalize = ds_equalize.batch(512)

    for idx, (image, _) in enumerate(ds_equalize):
        if idx == 0:
            images_equalize = image
        else:
            images_equalize = np.append(images_equalize,
                                        image,
                                        axis=0)
    if plot:
        visualize_list(images_original, images_equalize)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_equalize[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))


def test_equalize_py_c(plot=False):
    """
    Test Equalize Cpp op and python op
    """
    logger.info("Test Equalize cpp and python op")

    # equalize Images in cpp
    ds = de.ImageFolderDatasetV2(dataset_dir=DATA_DIR, shuffle=False)
    ds = ds.map(input_columns=["image"],
                operations=[C.Decode(), C.Resize((224, 224))])

    ds_c_equalize = ds.map(input_columns="image",
                           operations=C.Equalize())

    ds_c_equalize = ds_c_equalize.batch(512)

    for idx, (image, _) in enumerate(ds_c_equalize):
        if idx == 0:
            images_c_equalize = image
        else:
            images_c_equalize = np.append(images_c_equalize,
                                          image,
                                          axis=0)

    # Equalize images in python
    ds = de.ImageFolderDatasetV2(dataset_dir=DATA_DIR, shuffle=False)
    ds = ds.map(input_columns=["image"],
                operations=[C.Decode(), C.Resize((224, 224))])

    transforms_p_equalize = F.ComposeOp([lambda img: img.astype(np.uint8),
                                         F.ToPIL(),
                                         F.Equalize(),
                                         np.array])

    ds_p_equalize = ds.map(input_columns="image",
                           operations=transforms_p_equalize())

    ds_p_equalize = ds_p_equalize.batch(512)

    for idx, (image, _) in enumerate(ds_p_equalize):
        if idx == 0:
            images_p_equalize = image
        else:
            images_p_equalize = np.append(images_p_equalize,
                                          image,
                                          axis=0)

    num_samples = images_c_equalize.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_p_equalize[i], images_c_equalize[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))

    if plot:
        visualize_list(images_c_equalize, images_p_equalize, visualize_mode=2)


def test_equalize_one_channel():
    """
     Test Equalize cpp op with one channel image
     """
    logger.info("Test Equalize C Op With One Channel Images")

    c_op = C.Equalize()

    try:
        ds = de.ImageFolderDatasetV2(dataset_dir=DATA_DIR, shuffle=False)
        ds = ds.map(input_columns=["image"],
                    operations=[C.Decode(),
                                C.Resize((224, 224)),
                                lambda img: np.array(img[:, :, 0])])

        ds.map(input_columns="image",
               operations=c_op)

    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "The shape" in str(e)


def test_equalize_mnist_c(plot=False):
    """
    Test Equalize C op with MNIST dataset (Grayscale images)
    """
    logger.info("Test Equalize C Op With MNIST Images")
    ds = de.MnistDataset(dataset_dir=MNIST_DATA_DIR, num_samples=2, shuffle=False)
    ds_equalize_c = ds.map(input_columns="image",
                           operations=C.Equalize())
    ds_orig = de.MnistDataset(dataset_dir=MNIST_DATA_DIR, num_samples=2, shuffle=False)

    images = []
    images_trans = []
    labels = []
    for _, (data_orig, data_trans) in enumerate(zip(ds_orig, ds_equalize_c)):
        image_orig, label_orig = data_orig
        image_trans, _ = data_trans
        images.append(image_orig)
        labels.append(label_orig)
        images_trans.append(image_trans)

    # Compare with expected md5 from images
    filename = "equalize_mnist_result_c.npz"
    save_and_check_md5(ds_equalize_c, filename, generate_golden=GENERATE_GOLDEN)

    if plot:
        visualize_one_channel_dataset(images, images_trans, labels)


def test_equalize_md5_py():
    """
    Test Equalize py op with md5 check
    """
    logger.info("Test Equalize")

    # First dataset
    data1 = de.ImageFolderDatasetV2(dataset_dir=DATA_DIR, shuffle=False)
    transforms = F.ComposeOp([F.Decode(),
                              F.Equalize(),
                              F.ToTensor()])

    data1 = data1.map(input_columns="image", operations=transforms())
    # Compare with expected md5 from images
    filename = "equalize_01_result.npz"
    save_and_check_md5(data1, filename, generate_golden=GENERATE_GOLDEN)


def test_equalize_md5_c():
    """
    Test Equalize cpp op with md5 check
    """
    logger.info("Test Equalize cpp op with md5 check")

    # Generate dataset
    ds = de.ImageFolderDatasetV2(dataset_dir=DATA_DIR, shuffle=False)

    transforms_equalize = [C.Decode(),
                           C.Resize(size=[224, 224]),
                           C.Equalize(),
                           F.ToTensor()]

    data = ds.map(input_columns="image", operations=transforms_equalize)
    # Compare with expected md5 from images
    filename = "equalize_01_result_c.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)


if __name__ == "__main__":
    test_equalize_py(plot=False)
    test_equalize_c(plot=False)
    test_equalize_py_c(plot=False)
    test_equalize_mnist_c(plot=True)
    test_equalize_one_channel()
    test_equalize_md5_py()
    test_equalize_md5_c()
