# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
Testing Invert op in DE
"""
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.transforms.py_transforms
import mindspore.dataset.vision.py_transforms as F
import mindspore.dataset.vision.c_transforms as C
from mindspore import log as logger
from util import visualize_list, save_and_check_md5, diff_mse

DATA_DIR = "../data/dataset/testImageNetData/train/"

GENERATE_GOLDEN = False


def test_invert_callable():
    """
    Test Invert is callable
    """
    logger.info("Test Invert callable")
    img = np.fromfile("../data/dataset/apple.jpg", dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img = C.Decode()(img)
    img = C.Invert()(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    assert img.shape == (2268, 4032, 3)


def test_invert_py(plot=False):
    """
    Test Invert python op
    """
    logger.info("Test Invert Python op")

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

    # Color Inverted Images
    data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transforms_invert = mindspore.dataset.transforms.py_transforms.Compose([F.Decode(),
                                                                            F.Resize((224, 224)),
                                                                            F.Invert(),
                                                                            F.ToTensor()])

    ds_invert = data_set.map(operations=transforms_invert, input_columns="image")

    ds_invert = ds_invert.batch(512)

    for idx, (image, _) in enumerate(ds_invert):
        if idx == 0:
            images_invert = np.transpose(image.asnumpy(), (0, 2, 3, 1))
        else:
            images_invert = np.append(images_invert,
                                      np.transpose(image.asnumpy(), (0, 2, 3, 1)),
                                      axis=0)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = np.mean((images_invert[i] - images_original[i]) ** 2)
    logger.info("MSE= {}".format(str(np.mean(mse))))

    if plot:
        visualize_list(images_original, images_invert)


def test_invert_c(plot=False):
    """
    Test Invert Cpp op
    """
    logger.info("Test Invert cpp op")

    # Original Images
    data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transforms_original = [C.Decode(), C.Resize(size=[224, 224])]

    ds_original = data_set.map(operations=transforms_original, input_columns="image")

    ds_original = ds_original.batch(512)

    for idx, (image, _) in enumerate(ds_original):
        if idx == 0:
            images_original = image.asnumpy()
        else:
            images_original = np.append(images_original,
                                        image.asnumpy(),
                                        axis=0)

    # Invert Images
    data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transform_invert = [C.Decode(), C.Resize(size=[224, 224]),
                        C.Invert()]

    ds_invert = data_set.map(operations=transform_invert, input_columns="image")

    ds_invert = ds_invert.batch(512)

    for idx, (image, _) in enumerate(ds_invert):
        if idx == 0:
            images_invert = image.asnumpy()
        else:
            images_invert = np.append(images_invert,
                                      image.asnumpy(),
                                      axis=0)
    if plot:
        visualize_list(images_original, images_invert)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_invert[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))


def test_invert_py_c(plot=False):
    """
    Test Invert Cpp op and python op
    """
    logger.info("Test Invert cpp and python op")

    # Invert Images in cpp
    data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
    data_set = data_set.map(operations=[C.Decode(), C.Resize((224, 224))], input_columns=["image"])

    ds_c_invert = data_set.map(operations=C.Invert(), input_columns="image")

    ds_c_invert = ds_c_invert.batch(512)

    for idx, (image, _) in enumerate(ds_c_invert):
        if idx == 0:
            images_c_invert = image.asnumpy()
        else:
            images_c_invert = np.append(images_c_invert,
                                        image.asnumpy(),
                                        axis=0)

    # invert images in python
    data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
    data_set = data_set.map(operations=[C.Decode(), C.Resize((224, 224))], input_columns=["image"])

    transforms_p_invert = mindspore.dataset.transforms.py_transforms.Compose([lambda img: img.astype(np.uint8),
                                                                              F.ToPIL(),
                                                                              F.Invert(),
                                                                              np.array])

    ds_p_invert = data_set.map(operations=transforms_p_invert, input_columns="image")

    ds_p_invert = ds_p_invert.batch(512)

    for idx, (image, _) in enumerate(ds_p_invert):
        if idx == 0:
            images_p_invert = image.asnumpy()
        else:
            images_p_invert = np.append(images_p_invert,
                                        image.asnumpy(),
                                        axis=0)

    num_samples = images_c_invert.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_p_invert[i], images_c_invert[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))

    if plot:
        visualize_list(images_c_invert, images_p_invert, visualize_mode=2)


def test_invert_one_channel():
    """
     Test Invert cpp op with one channel image
     """
    logger.info("Test Invert C Op With One Channel Images")

    c_op = C.Invert()

    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(operations=[C.Decode(), C.Resize((224, 224)),
                                            lambda img: np.array(img[:, :, 0])], input_columns=["image"])

        data_set.map(operations=c_op, input_columns="image")

    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "The shape" in str(e)


def test_invert_md5_py():
    """
    Test Invert python op with md5 check
    """
    logger.info("Test Invert python op with md5 check")

    # Generate dataset
    data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transforms_invert = mindspore.dataset.transforms.py_transforms.Compose([F.Decode(),
                                                                            F.Invert(),
                                                                            F.ToTensor()])

    data = data_set.map(operations=transforms_invert, input_columns="image")
    # Compare with expected md5 from images
    filename = "invert_01_result_py.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)


def test_invert_md5_c():
    """
    Test Invert cpp op with md5 check
    """
    logger.info("Test Invert cpp op with md5 check")

    # Generate dataset
    data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transforms_invert = [C.Decode(),
                         C.Resize(size=[224, 224]),
                         C.Invert(),
                         F.ToTensor()]

    data = data_set.map(operations=transforms_invert, input_columns="image")
    # Compare with expected md5 from images
    filename = "invert_01_result_c.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)


if __name__ == "__main__":
    test_invert_callable()
    test_invert_py(plot=False)
    test_invert_c(plot=False)
    test_invert_py_c(plot=False)
    test_invert_one_channel()
    test_invert_md5_py()
    test_invert_md5_c()
