# Copyright 2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Testing FiveCrop in DE
"""
import numpy as np
import os
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.transforms
import mindspore.dataset.transforms.transforms as t_trans
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger
from util import visualize_list, save_and_check_md5_pil

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
IMAGENET_DIR = "../data/dataset/testPK/data"
TEST_DATA_DATASET_FUNC ="../data/dataset/"

GENERATE_GOLDEN = False


def test_five_crop_op(plot=False):
    """
    Feature: FiveCrop op
    Description: Test FiveCrop op basic usage
    Expectation: Output is the same as expected output
    """
    logger.info("test_five_crop")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms_1 = [
        vision.Decode(True),
        vision.ToTensor(),
    ]
    transform_1 = mindspore.dataset.transforms.Compose(transforms_1)
    data1 = data1.map(operations=transform_1, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms_2 = [
        vision.Decode(True),
        vision.FiveCrop(200),
        lambda *images: np.stack([vision.ToTensor()(image) for image in images])  # 4D stack of 5 images
    ]
    transform_2 = mindspore.dataset.transforms.Compose(transforms_2)
    data2 = data2.map(operations=transform_2, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        num_iter += 1
        image_1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_2 = item2["image"]

        logger.info("shape of image_1: {}".format(image_1.shape))
        logger.info("shape of image_2: {}".format(image_2.shape))

        logger.info("dtype of image_1: {}".format(image_1.dtype))
        logger.info("dtype of image_2: {}".format(image_2.dtype))
        if plot:
            visualize_list(np.array([image_1] * 5), (image_2 * 255).astype(np.uint8).transpose(0, 2, 3, 1))

        # The output data should be of a 4D tensor shape, a stack of 5 images.
        assert len(image_2.shape) == 4
        assert image_2.shape[0] == 5


def test_five_crop_multiprocessing():
    """
    Feature: Test FiveCrop operator with multiprocessing
    Description: Test FiveCrop operator with multiprocessing
    Expectation: Output is the same as expected output
    """
    logger.info("Test FiveCrop operator with multiprocessing")

    dataset = ds.ImageFolderDataset(IMAGENET_DIR, num_samples=20, num_parallel_workers=8, shuffle=False)
    transforms = [
        vision.Decode(to_pil=True),
        vision.FiveCrop(20),
        lambda *images: np.stack([vision.ToTensor()(image) for image in images])
    ]
    transform = mindspore.dataset.transforms.Compose(transforms)
    num_parallel_workers = 8
    data = dataset.map(input_columns=["image"], operations=transform, num_parallel_workers=num_parallel_workers,
                       python_multiprocessing=True)
    num_iter = 0
    for _ in data.create_dict_iterator(output_numpy=True):
        num_iter += 1
    assert num_iter == 20


def test_five_crop_error_msg():
    """
    Feature: FiveCrop op
    Description: Test FiveCrop op when the input image is not in the correct format.
    Expectation: Invalid input is detected
    """

    logger.info("test_five_crop_error_msg")

    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        vision.Decode(True),
        vision.FiveCrop(200),
        vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])

    with pytest.raises(RuntimeError) as info:
        for _ in data:
            pass
    error_msg = \
        "map operation: [ToTensor] failed. The op is OneToOne, can only accept one tensor as input."
    assert error_msg in str(info.value)


def test_five_crop_md5():
    """
    Feature: FiveCrop op
    Description: Test FiveCrop op with md5 check
    Expectation: Passes the md5 check test
    """
    logger.info("test_five_crop_md5")

    # First dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        vision.Decode(True),
        vision.FiveCrop(100),
        lambda *images: np.stack([vision.ToTensor()(image) for image in images])  # 4D stack of 5 images
    ]
    transform = mindspore.dataset.transforms.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])
    # Compare with expected md5 from images
    filename = "five_crop_01_result.npz"
    save_and_check_md5_pil(data, filename, generate_golden=GENERATE_GOLDEN)


def test_five_crop_operation_01():
    """
    Feature: FiveCrop operation
    Description: Testing the normal functionality of the FiveCrop operator
    Expectation: The Output is equal to the expected output
    """
    # FiveCrop operator: Test size is 10
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    size = 10
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir)
    transforms = [
        vision.Decode(),
        vision.ToTensor()
    ]
    transform = t_trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir)
    transforms1 = [
        vision.Decode(to_pil=True),
        vision.FiveCrop(size=size),
        lambda *images: np.stack([vision.ToTensor()(image) for image in images])
    ]
    transform1 = t_trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)
    for _, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        image_2 = data2["image"]
        # The output data should be of a 4D tensor shape, a stack of 5 images.
        assert len(image_2.shape) == 4
        assert image_2.shape[0] == 5

    # FiveCrop operator: Test size is 100
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    size = 100

    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir)
    transforms = [
        vision.Decode(),
        vision.ToTensor()
    ]
    transform = t_trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)

    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir)
    transforms1 = [
        vision.Decode(to_pil=True),
        vision.FiveCrop(size=size),
        lambda *images: np.stack([vision.ToTensor()(image) for image in images])
    ]
    transform1 = t_trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for _, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        image_2 = data2["image"]
        assert len(image_2.shape) == 4
        assert image_2.shape[0] == 5

    # FiveCrop operator: Test size is a tuple of length 2, height and width are equal
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    size = (20, 20)
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir)
    transforms = [
        vision.Decode(),
        vision.ToTensor()
    ]
    transform = t_trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)

    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir)
    transforms1 = [
        vision.Decode(to_pil=True),
        vision.FiveCrop(size=size),
        lambda *images: np.stack([vision.ToTensor()(image) for image in images])
    ]
    transform1 = t_trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for _, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        image_2 = data2["image"]
        # The output data should be of a 4D tensor shape, a stack of 5 images.
        assert len(image_2.shape) == 4
        assert image_2.shape[0] == 5


def test_five_crop_operation_02():
    """
    Feature: FiveCrop operation
    Description: Testing the normal functionality of the FiveCrop operator
    Expectation: The Output is equal to the expected output
    """
    # FiveCrop operator: Test size is a list of length 2, height and width are not equal
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    size = [20, 500]
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir)
    transforms = [
        vision.Decode(),
        vision.ToTensor()
    ]
    transform = t_trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir)
    transforms1 = [
        vision.Decode(to_pil=True),
        vision.FiveCrop(size=size),
        lambda *images: np.stack([vision.ToTensor()(image) for image in images])
    ]
    transform1 = t_trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for _, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        image_2 = data2["image"]
        assert len(image_2.shape) == 4
        assert image_2.shape[0] == 5

    # FiveCrop operator: Test size is a list of length 2, height and width are equal
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    size = [10, 10]
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir)
    transforms = [
        vision.Decode(),
        vision.ToTensor()
    ]
    transform = t_trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir)
    transforms1 = [
        vision.Decode(to_pil=True),
        vision.FiveCrop(size=size),
        lambda *images: np.stack([vision.ToTensor()(image) for image in images])
    ]
    transform1 = t_trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for _, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        image_2 = data2["image"]
        assert len(image_2.shape) == 4
        assert image_2.shape[0] == 5

    # FiveCrop operator: Test image is jpg
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        size = 100
        five_crop_op = vision.FiveCrop(size=size)
        out = five_crop_op(image)
        assert len(out) == 5

    # FiveCrop operator: Test image is bmp
    image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
    with Image.open(image_bmp) as image:
        size = (50, 100)
        five_crop_op = vision.FiveCrop(size=size)
        out = five_crop_op(image)
        assert len(out) == 5

    # FiveCrop operator: Test image is png
    image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    with Image.open(image_png) as image:
        size = [50, 50]
        five_crop_op = vision.FiveCrop(size=size)
        out = five_crop_op(image)
        assert len(out) == 5

    # FiveCrop operator: Test image is gif
    image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")
    with Image.open(image_gif) as image:
        size = 1
        five_crop_op = vision.FiveCrop(size=size)
        out = five_crop_op(image)
        assert len(out) == 5

    # FiveCrop operator: Test image.shape is size
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        size = [432, 576]
        five_crop_op = vision.FiveCrop(size=size)
        out = five_crop_op(image)
        assert len(out) == 5
        assert (np.array(image) == np.array(out[0])).all()
        assert (np.array(image) == np.array(out[1])).all()
        assert (np.array(image) == np.array(out[2])).all()
        assert (np.array(image) == np.array(out[3])).all()
        assert (np.array(image) == np.array(out[4])).all()


def test_five_crop_exception_01():
    """
    Feature: FiveCrop operation
    Description: Testing the FiveCrop Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # FiveCrop operator: Test no size
    with pytest.raises(TypeError, match="missing a required argument: 'size'"):
        vision.FiveCrop()

    # FiveCrop operator: Test size is 'wajda^% 12^*l'
    size = "wajda^% 12^*l"
    with pytest.raises(TypeError, match=(r"Argument size with value wajda\^\% 12\^\*l is not of "
                                         r"type \[<class 'int'>, <class 'list'>, <class 'tuple'>\]")):
        vision.FiveCrop(size=size)

    # FiveCrop operator: Test size is 0
    size = 0
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[1, 16777216\\]."):
        vision.FiveCrop(size=size)

    # FiveCrop operator: Test image is numpy array
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image_data:
        image = np.array(image_data)
        size = [100, 100]
        five_crop_op = vision.FiveCrop(size=size)
        with pytest.raises(TypeError, match="img should be PIL image. Got <class 'numpy.ndarray'>."):
            five_crop_op(image)

    # FiveCrop operator: Test image is numpy list
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image_data:
        image = np.array(image_data).tolist()
        size = [100, 100]
        five_crop_op = vision.FiveCrop(size=size)
        with pytest.raises(TypeError, match="img should be PIL image. Got <class 'list'>."):
            five_crop_op(image)

    # FiveCrop operator: Test no image
    size = [100, 100]
    five_crop_op = vision.FiveCrop(size=size)
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'img'"):
        five_crop_op()

    # Test size > image.shape
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        size = [433, 576]
        five_crop_op = vision.FiveCrop(size=size)
        with pytest.raises(ValueError,
                           match="Crop size \\[433, 576\\] is larger than input image size \\(432, 576\\)."):
            five_crop_op(image)

    # FiveCrop operator: Test size is float
    size = 50.0
    with pytest.raises(TypeError, match="Argument size with value 50.0 is not of "
                                        "type \\[<class 'int'>, <class 'list'>, <class 'tuple'>\\]."):
        vision.FiveCrop(size=size)

    # FiveCrop operator: Test size is 1-tuple
    size = (50, 60, 70)
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple \\(h, w\\) of length 2."):
        vision.FiveCrop(size=size)

    # FiveCrop operator: Test size is list
    size = [60]
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple \\(h, w\\) of length 2."):
        vision.FiveCrop(size=size)

    # FiveCrop operator: Test size is numpy
    size = np.array([10, 20])
    with pytest.raises(TypeError, match="Argument size with value \\[10 20\\] is not of "
                                        "type \\[<class 'int'>, <class 'list'>, <class 'tuple'>\\]."):
        vision.FiveCrop(size=size)


if __name__ == "__main__":
    test_five_crop_op(plot=True)
    test_five_crop_error_msg()
    test_five_crop_md5()
    test_five_crop_operation_01()
    test_five_crop_operation_02()
    test_five_crop_exception_01()
