# Copyright 2021 Huawei Technologies Co., Ltd
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
Testing AdjustGamma op in DE
"""
import numpy as np
from numpy.testing import assert_allclose
import PIL

import mindspore.dataset as ds
import mindspore.dataset.transforms.py_transforms
import mindspore.dataset.vision.py_transforms as F
import mindspore.dataset.vision.c_transforms as C
from mindspore import log as logger

DATA_DIR = "../data/dataset/testImageNetData/train/"
MNIST_DATA_DIR = "../data/dataset/testMnistData"

DATA_DIR_2 = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def generate_numpy_random_rgb(shape):
    """
    Only generate floating points that are fractions like n / 256, since they
    are RGB pixels. Some low-precision floating point types in this test can't
    handle arbitrary precision floating points well.
    """
    return np.random.randint(0, 256, shape) / 255.


def test_adjust_gamma_c_eager():
    # Eager 3-channel
    rgb_flat = generate_numpy_random_rgb((64, 3)).astype(np.float32)
    img_in = rgb_flat.reshape((8, 8, 3))

    adjustgamma_op = C.AdjustGamma(10, 1)
    img_out = adjustgamma_op(img_in)
    assert img_out is not None


def test_adjust_gamma_py_eager():
    # Eager 3-channel
    rgb_flat = generate_numpy_random_rgb((64, 3)).astype(np.uint8)
    img_in = PIL.Image.fromarray(rgb_flat.reshape((8, 8, 3)))

    adjustgamma_op = F.AdjustGamma(10, 1)
    img_out = adjustgamma_op(img_in)
    assert img_out is not None


def test_adjust_gamma_c_eager_gray():
    # Eager 3-channel
    rgb_flat = generate_numpy_random_rgb((64, 1)).astype(np.float32)
    img_in = rgb_flat.reshape((8, 8))

    adjustgamma_op = C.AdjustGamma(10, 1)
    img_out = adjustgamma_op(img_in)
    assert img_out is not None


def test_adjust_gamma_py_eager_gray():
    # Eager 3-channel
    rgb_flat = generate_numpy_random_rgb((64, 1)).astype(np.uint8)
    img_in = PIL.Image.fromarray(rgb_flat.reshape((8, 8)))

    adjustgamma_op = F.AdjustGamma(10, 1)
    img_out = adjustgamma_op(img_in)
    assert img_out is not None


def test_adjust_gamma_invalid_gamma_param_c():
    """
    Test AdjustGamma C Op with invalid ignore parameter
    """
    logger.info("Test AdjustGamma C Op with invalid ignore parameter")
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(operations=[C.Decode(), C.Resize((224, 224)), lambda img: np.array(img[:, :, 0])],
                                input_columns=["image"])
        # invalid gamma
        data_set = data_set.map(operations=C.AdjustGamma(gamma=-10.0, gain=1.0),
                                input_columns="image")
    except ValueError as error:
        logger.info("Got an exception in AdjustGamma: {}".format(str(error)))
        assert "Input is not within the required interval of " in str(error)
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(operations=[C.Decode(), C.Resize((224, 224)), lambda img: np.array(img[:, :, 0])],
                                input_columns=["image"])
        # invalid gamma
        data_set = data_set.map(operations=C.AdjustGamma(gamma=[1, 2], gain=1.0),
                                input_columns="image")
    except TypeError as error:
        logger.info("Got an exception in AdjustGamma: {}".format(str(error)))
        assert "is not of type [<class 'float'>, <class 'int'>], but got" in str(error)


def test_adjust_gamma_invalid_gamma_param_py():
    """
    Test AdjustGamma python Op with invalid ignore parameter
    """
    logger.info("Test AdjustGamma python Op with invalid ignore parameter")
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        trans = mindspore.dataset.transforms.py_transforms.Compose([
            F.Decode(),
            F.Resize((224, 224)),
            F.AdjustGamma(gamma=-10.0),
            F.ToTensor()
        ])
        data_set = data_set.map(operations=[trans], input_columns=["image"])
    except ValueError as error:
        logger.info("Got an exception in AdjustGamma: {}".format(str(error)))
        assert "Input is not within the required interval of " in str(error)
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        trans = mindspore.dataset.transforms.py_transforms.Compose([
            F.Decode(),
            F.Resize((224, 224)),
            F.AdjustGamma(gamma=[1, 2]),
            F.ToTensor()
        ])
        data_set = data_set.map(operations=[trans], input_columns=["image"])
    except TypeError as error:
        logger.info("Got an exception in AdjustGamma: {}".format(str(error)))
        assert "is not of type [<class 'float'>, <class 'int'>], but got" in str(error)


def test_adjust_gamma_invalid_gain_param_c():
    """
    Test AdjustGamma C Op with invalid gain parameter
    """
    logger.info("Test AdjustGamma C Op with invalid gain parameter")
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(operations=[C.Decode(), C.Resize((224, 224)), lambda img: np.array(img[:, :, 0])],
                                input_columns=["image"])
        # invalid gain
        data_set = data_set.map(operations=C.AdjustGamma(gamma=10.0, gain=[1, 10]),
                                input_columns="image")
    except TypeError as error:
        logger.info("Got an exception in AdjustGamma: {}".format(str(error)))
        assert "is not of type [<class 'float'>, <class 'int'>], but got " in str(error)


def test_adjust_gamma_invalid_gain_param_py():
    """
    Test AdjustGamma python Op with invalid gain parameter
    """
    logger.info("Test AdjustGamma python Op with invalid gain parameter")
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        trans = mindspore.dataset.transforms.py_transforms.Compose([
            F.Decode(),
            F.Resize((224, 224)),
            F.AdjustGamma(gamma=10.0, gain=[1, 10]),
            F.ToTensor()
        ])
        data_set = data_set.map(operations=[trans], input_columns=["image"])
    except TypeError as error:
        logger.info("Got an exception in AdjustGamma: {}".format(str(error)))
        assert "is not of type [<class 'float'>, <class 'int'>], but got " in str(error)


def test_adjust_gamma_pipeline_c():
    """
    Test AdjustGamma C Op Pipeline
    """
    # First dataset
    transforms1 = [C.Decode(), C.Resize([64, 64])]
    transforms1 = mindspore.dataset.transforms.py_transforms.Compose(
        transforms1)
    ds1 = ds.TFRecordDataset(DATA_DIR_2,
                             SCHEMA_DIR,
                             columns_list=["image"],
                             shuffle=False)
    ds1 = ds1.map(operations=transforms1, input_columns=["image"])

    # Second dataset
    transforms2 = [
        C.Decode(),
        C.Resize([64, 64]),
        C.AdjustGamma(1.0, 1.0)
    ]
    transform2 = mindspore.dataset.transforms.py_transforms.Compose(
        transforms2)
    ds2 = ds.TFRecordDataset(DATA_DIR_2,
                             SCHEMA_DIR,
                             columns_list=["image"],
                             shuffle=False)
    ds2 = ds2.map(operations=transform2, input_columns=["image"])

    num_iter = 0
    for data1, data2 in zip(ds1.create_dict_iterator(num_epochs=1),
                            ds2.create_dict_iterator(num_epochs=1)):
        num_iter += 1
        ori_img = data1["image"].asnumpy()
        cvt_img = data2["image"].asnumpy()
        assert_allclose(ori_img.flatten(),
                        cvt_img.flatten(),
                        rtol=1e-5,
                        atol=0)
        assert ori_img.shape == cvt_img.shape


def test_adjust_gamma_pipeline_py():
    """
    Test AdjustGamma python Op Pipeline
    """
    # First dataset
    transforms1 = [F.Decode(), F.Resize([64, 64]), F.ToTensor()]
    transforms1 = mindspore.dataset.transforms.py_transforms.Compose(
        transforms1)
    ds1 = ds.TFRecordDataset(DATA_DIR_2,
                             SCHEMA_DIR,
                             columns_list=["image"],
                             shuffle=False)
    ds1 = ds1.map(operations=transforms1, input_columns=["image"])

    # Second dataset
    transforms2 = [
        F.Decode(),
        F.Resize([64, 64]),
        F.AdjustGamma(1.0, 1.0),
        F.ToTensor()
    ]
    transform2 = mindspore.dataset.transforms.py_transforms.Compose(
        transforms2)
    ds2 = ds.TFRecordDataset(DATA_DIR_2,
                             SCHEMA_DIR,
                             columns_list=["image"],
                             shuffle=False)
    ds2 = ds2.map(operations=transform2, input_columns=["image"])

    num_iter = 0
    for data1, data2 in zip(ds1.create_dict_iterator(num_epochs=1),
                            ds2.create_dict_iterator(num_epochs=1)):
        num_iter += 1
        ori_img = data1["image"].asnumpy()
        cvt_img = data2["image"].asnumpy()
        assert_allclose(ori_img.flatten(),
                        cvt_img.flatten(),
                        rtol=1e-5,
                        atol=0)
        assert ori_img.shape == cvt_img.shape


def test_adjust_gamma_pipeline_py_gray():
    """
    Test AdjustGamma python Op Pipeline 1-channel
    """
    # First dataset
    transforms1 = [F.Decode(), F.Resize([64, 64]), F.Grayscale(), F.ToTensor()]
    transforms1 = mindspore.dataset.transforms.py_transforms.Compose(
        transforms1)
    ds1 = ds.TFRecordDataset(DATA_DIR_2,
                             SCHEMA_DIR,
                             columns_list=["image"],
                             shuffle=False)
    ds1 = ds1.map(operations=transforms1, input_columns=["image"])

    # Second dataset
    transforms2 = [
        F.Decode(),
        F.Resize([64, 64]),
        F.Grayscale(),
        F.AdjustGamma(1.0, 1.0),
        F.ToTensor()
    ]
    transform2 = mindspore.dataset.transforms.py_transforms.Compose(
        transforms2)
    ds2 = ds.TFRecordDataset(DATA_DIR_2,
                             SCHEMA_DIR,
                             columns_list=["image"],
                             shuffle=False)
    ds2 = ds2.map(operations=transform2, input_columns=["image"])

    num_iter = 0
    for data1, data2 in zip(ds1.create_dict_iterator(num_epochs=1),
                            ds2.create_dict_iterator(num_epochs=1)):
        num_iter += 1
        ori_img = data1["image"].asnumpy()
        cvt_img = data2["image"].asnumpy()
        assert_allclose(ori_img.flatten(),
                        cvt_img.flatten(),
                        rtol=1e-5,
                        atol=0)


if __name__ == "__main__":
    test_adjust_gamma_c_eager()
    test_adjust_gamma_py_eager()
    test_adjust_gamma_c_eager_gray()
    test_adjust_gamma_py_eager_gray()

    test_adjust_gamma_invalid_gamma_param_c()
    test_adjust_gamma_invalid_gamma_param_py()
    test_adjust_gamma_invalid_gain_param_c()
    test_adjust_gamma_invalid_gain_param_py()
    test_adjust_gamma_pipeline_c()
    test_adjust_gamma_pipeline_py()
    test_adjust_gamma_pipeline_py_gray()
