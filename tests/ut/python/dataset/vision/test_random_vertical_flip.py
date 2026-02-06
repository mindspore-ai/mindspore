# Copyright 2025 Huawei Technologies Co., Ltd
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
Testing the random vertical flip op in DE
"""
import numpy as np
import pytest
import os
from PIL import Image

from mindspore import log as logger
import mindspore.dataset as ds
import mindspore.dataset.transforms as ops
import mindspore.dataset.vision.transforms as vision
from util import save_and_check_md5, save_and_check_md5_pil, visualize_list, visualize_image, diff_mse, \
    config_get_set_seed, config_get_set_num_parallel_workers

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
TEST_DATA_DATASET_FUNC ="../data/dataset/"


def generator_mc(maxid=3):
    """Multi-column generator function as callable input"""
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")

    with Image.open(image_jpg) as image_data:
        image = np.array(image_data)
    with Image.open(image_png) as image_data1:
        image1 = np.array(image_data1)
    with Image.open(image_gif) as image_data2:
        image2 = np.array(image_data2)
    image3 = np.random.randn(300, 300, 3)
    for _ in range(maxid):
        yield image, image, image1, image2, image3


def v_flip(image):
    """
    Apply the random_vertical
    """

    # with the seed provided in this test case, it will always flip.
    # that's why we flip here too
    image = image[::-1, :, :]
    return image


def test_random_vertical_op(plot=False):
    """
    Feature: RandomVerticalFlip op
    Description: Test RandomVerticalFlip with default probability
    Expectation: The dataset is processed as expected
    """
    logger.info("Test random_vertical")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    random_vertical_op = vision.RandomVerticalFlip(1.0)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=random_vertical_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=decode_op, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):

        # with the seed value, we can only guarantee the first number generated
        if num_iter > 0:
            break

        image_v_flipped = item1["image"]
        image = item2["image"]
        image_v_flipped_2 = v_flip(image)

        mse = diff_mse(image_v_flipped, image_v_flipped_2)
        assert mse == 0
        logger.info("image_{}, mse: {}".format(num_iter + 1, mse))
        num_iter += 1
        if plot:
            visualize_image(image, image_v_flipped, mse, image_v_flipped_2)


def test_random_vertical_valid_prob_c():
    """
    Feature: RandomVerticalFlip op
    Description: Test RandomVerticalFlip op with Cpp implementation using valid non-default input
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_vertical_valid_prob_c")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    random_horizontal_op = vision.RandomVerticalFlip(0.8)
    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=random_horizontal_op, input_columns=["image"])

    filename = "random_vertical_01_c_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_vertical_valid_prob_py():
    """
    Feature: RandomVerticalFlip op
    Description: Test RandomVerticalFlip op with Python implementation using valid non-default input
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_vertical_valid_prob_py")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        vision.Decode(True),
        vision.RandomVerticalFlip(0.8),
        vision.ToTensor()
    ]
    transform = ops.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])

    filename = "random_vertical_01_py_result.npz"
    save_and_check_md5_pil(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_vertical_invalid_prob_c():
    """
    Feature: RandomVerticalFlip op
    Description: Test RandomVerticalFlip op with Cpp implementation using invalid input
    Expectation: Error is raised as expected
    """
    logger.info("test_random_vertical_invalid_prob_c")

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    try:
        # Note: Valid range of prob should be [0.0, 1.0]
        random_horizontal_op = vision.RandomVerticalFlip(1.5)
        data = data.map(operations=decode_op, input_columns=["image"])
        data = data.map(operations=random_horizontal_op, input_columns=["image"])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert 'Input prob is not within the required interval of [0.0, 1.0].' in str(e)


def test_random_vertical_invalid_prob_py():
    """
    Feature: RandomVerticalFlip op
    Description: Test RandomVerticalFlip op with Python implementation using invalid input
    Expectation: Error is raised as expected
    """
    logger.info("test_random_vertical_invalid_prob_py")

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    try:
        transforms = [
            vision.Decode(True),
            # Note: Valid range of prob should be [0.0, 1.0]
            vision.RandomVerticalFlip(1.5),
            vision.ToTensor()
        ]
        transform = ops.Compose(transforms)
        data = data.map(operations=transform, input_columns=["image"])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert 'Input prob is not within the required interval of [0.0, 1.0].' in str(e)


def test_random_vertical_comp(plot=False):
    """
    Feature: RandomVerticalFlip op
    Description: Test RandomVerticalFlip and compare between Python and Cpp image augmentation ops
    Expectation: Image outputs from both implementation are the same
    """
    logger.info("test_random_vertical_comp")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    # Note: The image must be flipped if prob is set to be 1
    random_horizontal_op = vision.RandomVerticalFlip(1)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=random_horizontal_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        vision.Decode(True),
        # Note: The image must be flipped if prob is set to be 1
        vision.RandomVerticalFlip(1),
        vision.ToTensor()
    ]
    transform = ops.Compose(transforms)
    data2 = data2.map(operations=transform, input_columns=["image"])

    images_list_c = []
    images_list_py = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_c = item1["image"]
        image_py = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        images_list_c.append(image_c)
        images_list_py.append(image_py)

        # Check if the output images are the same
        mse = diff_mse(image_c, image_py)
        assert mse < 0.001
    if plot:
        visualize_list(images_list_c, images_list_py, visualize_mode=2)


def test_random_vertical_op_1():
    """
    Feature: RandomVerticalFlip op
    Description: Test RandomVerticalFlip with different fields
    Expectation: The dataset is processed as expected
    """
    logger.info("Test RandomVerticalFlip with different fields.")

    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data = data.map(operations=ops.Duplicate(), input_columns=["image"],
                    output_columns=["image", "image_copy"])
    random_vertical_op = vision.RandomVerticalFlip(1.0)
    decode_op = vision.Decode()

    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=decode_op, input_columns=["image_copy"])
    data = data.map(operations=random_vertical_op, input_columns=["image", "image_copy"])

    num_iter = 0
    for data1 in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = data1["image"]
        image_copy = data1["image_copy"]
        mse = diff_mse(image, image_copy)
        logger.info("image_{}, mse: {}".format(num_iter + 1, mse))
        assert mse == 0
        num_iter += 1


def test_random_vertical_flip_invalid_data():
    """
    Feature: RandomVerticalFlip
    Description: Test RandomVerticalFlip with invalid data
    Expectation: Error is raised as expected
    """

    invalid_type_img = np.random.random((32, 32, 3)).astype(np.str_)
    invalid_shape_img = np.random.random(32).astype(np.float32)
    random_vertical_flip = vision.RandomVerticalFlip(0.1)

    with pytest.raises(RuntimeError) as error_info:
        random_vertical_flip(invalid_type_img)
    assert "Currently unsupported data type: [uint32, int64, uint64, string]" in str(error_info.value)

    with pytest.raises(RuntimeError) as error_info:
        random_vertical_flip(invalid_shape_img)
    assert "input tensor is not in shape of <H,W> or <H,W,C>" in str(error_info.value)


def test_random_vertical_flip_operation_01():
    """
    Feature: RandomVerticalFlip operation
    Description: Testing the normal functionality of the RandomVerticalFlip operator
    Expectation: The Output is equal to the expected output
    """
    # RandomVerticalFlip operator:Test prob is 1
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "testImageNetData", "train")
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    prob = 1
    random_vertical_flip_op = vision.RandomVerticalFlip(prob=prob)
    dataset2 = dataset2.map(input_columns=["image"], operations=random_vertical_flip_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # RandomVerticalFlip operator:Test prob is 0
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "testImageNetData", "train")
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    prob = 0
    random_vertical_flip_op = vision.RandomVerticalFlip(prob=prob)
    dataset2 = dataset2.map(input_columns=["image"], operations=random_vertical_flip_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # RandomVerticalFlip operator:Test prob is 0.1
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    prob = 0.1
    ds2 = ds.ImageFolderDataset(data_dir, 1)
    transforms1 = [
        vision.Decode(),
        vision.RandomVerticalFlip(prob),
        vision.ToTensor()
    ]
    transform1 = ops.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for _ in ds2.create_dict_iterator(output_numpy=True):
        pass


def test_random_vertical_flip_operation_02():
    """
    Feature: RandomVerticalFlip operation
    Description: Testing the normal functionality of the RandomVerticalFlip operator
    Expectation: The Output is equal to the expected output
    """
    # RandomVerticalFlip operator:Test prob is 0.9
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    prob = 0.9
    ds2 = ds.ImageFolderDataset(data_dir, 1)
    transforms1 = [
        vision.Decode(to_pil=True),
        vision.RandomVerticalFlip(prob),
        vision.ToTensor()
    ]
    transform1 = ops.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for _ in ds2.create_dict_iterator(output_numpy=True):
        pass

    # RandomVerticalFlip operator:Test input is Multi-column dataset
    source = generator_mc
    column_names = ["image1", "image2", "image3", "image4", "image5"]
    dataset = ds.GeneratorDataset(source, column_names)
    prob = 0.8
    random_vertical_flip_op = vision.RandomVerticalFlip(prob=prob)
    dataset = dataset.map(input_columns=["image1", "image2", "image3", "image4", "image5"],
                          operations=random_vertical_flip_op)
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["image1"] == data["image2"]).all()

    # RandomVerticalFlip operator:Test input is two image
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")
    with Image.open(image_jpg) as image:
        with Image.open(image_gif) as image2:
            prob = 1
            random_vertical_flip_op = vision.RandomVerticalFlip(prob)
            _ = random_vertical_flip_op(image, image2)

    # RandomVerticalFlip operator:Test input is bmp image
    image_file3 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
    with Image.open(image_file3) as image:
        prob = 1
        transforms_op = vision.RandomVerticalFlip(prob)
        out = transforms_op(image)
        assert (np.flipud(np.array(out)) == (np.array(image))).all()

    # RandomVerticalFlip operator:Test input is png image
    image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    with  Image.open(image_png) as image:
        prob = 0
        random_vertical_flip_op = vision.RandomVerticalFlip(prob)
        out = random_vertical_flip_op(image)
        assert (np.array(image) == out).all()

    # RandomVerticalFlip operator:Test input is gif image
    image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")
    with Image.open(image_gif) as image:
        prob = 0
        random_vertical_flip_op = vision.RandomVerticalFlip(prob)
        out = random_vertical_flip_op(image)
        assert (np.array(image) == out).all()

    # RandomVerticalFlip operator:Test input is 3d numpy array
    image = np.random.randn(2500, 3000, 1)
    prob = 0.956
    random_vertical_flip_op = vision.RandomVerticalFlip(prob)
    _ = random_vertical_flip_op(image)

    # RandomVerticalFlip operator:Test 4 channel numpy array
    image = np.random.randn(32, 32, 4)
    prob = 0.01
    random_vertical_flip_op = vision.RandomVerticalFlip(prob)
    _ = random_vertical_flip_op(image)

    # RandomVerticalFlip operator:Test input is 2d numpy array
    image = np.random.randint(0, 255, (255, 255)).astype(np.uint8)
    random_vertical_flip_op = vision.RandomVerticalFlip()
    _ = random_vertical_flip_op(image)


def test_random_vertical_flip_exception_01():
    """
    Feature: RandomVerticalFlip operation
    Description: Testing the RandomVerticalFlip Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # RandomVerticalFlip operator:Test input is 3d tuple
    image = tuple(np.random.randint(0, 255, (255, 255, 3)).astype(np.uint8))
    random_vertical_flip_op = vision.RandomVerticalFlip()
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'tuple'>"):
        _ = random_vertical_flip_op(image)

    # RandomVerticalFlip operator:Test input is 3d list
    image = list(np.random.randint(0, 255, (52, 52, 3)).astype(np.uint8))
    random_vertical_flip_op = vision.RandomVerticalFlip()
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>"):
        _ = random_vertical_flip_op(image)

    # RandomVerticalFlip operator:Test prob > 1
    prob = 1.1
    with pytest.raises(ValueError, match="Input prob is not within the required interval"):
        vision.RandomVerticalFlip(prob=prob)

    # RandomVerticalFlip operator:Test prob < 0
    prob = -0.1
    with pytest.raises(ValueError, match="Input prob is not within the required interval"):
        vision.RandomVerticalFlip(prob=prob)

    # RandomVerticalFlip operator:Test prob is None
    prob = None
    with pytest.raises(TypeError,
                       match="Argument prob with value None is not of type \\[<class 'float'>, <class 'int'>\\]."):
        vision.RandomVerticalFlip(prob=prob)

    # RandomVerticalFlip operator:Test prob is bool
    prob = True
    with pytest.raises(TypeError,
                       match="Argument prob with value True is not of type \\(<class 'float'>, <class 'int'>\\)."):
        vision.RandomVerticalFlip(prob=prob)

    # RandomVerticalFlip operator:Test prob is str
    prob = "1"
    with pytest.raises(TypeError,
                       match="Argument prob with value 1 is not of type \\[<class 'float'>, <class 'int'>\\]."):
        vision.RandomVerticalFlip(prob=prob)

    # RandomVerticalFlip operator:Test prob is tuple
    prob = (0.5, 1)
    with pytest.raises(TypeError, match=r'is not of type \[\<class \'float\'\>, \<class \'int\'\>\].'):
        vision.RandomVerticalFlip(prob)

    # RandomVerticalFlip operator:Test prob is list
    prob = [0.5, 0.9]
    with pytest.raises(TypeError, match=r'is not of type \[\<class \'float\'\>, \<class \'int\'\>\].'):
        vision.RandomVerticalFlip(prob)

    # RandomVerticalFlip operator:Test input is not <H,W,C> or <H,W>
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    image = np.fromfile(image_jpg, dtype=np.uint8)
    random_vertical_flip_op = vision.RandomVerticalFlip(1)
    with pytest.raises(RuntimeError,
                       match="RandomVerticalFlip: input tensor is not in shape of " + \
                             "<H,W> or <H,W,C>, but got rank: 1. You may need to perform Decode first"):
        random_vertical_flip_op(image)

    # RandomVerticalFlip operator:Test input is 4d numpy array
    image = np.random.randint(0, 255, (255, 255, 3, 3)).astype(np.uint8)
    random_vertical_flip_op = vision.RandomVerticalFlip(1)
    with pytest.raises(RuntimeError, match="RandomVerticalFlip: input tensor is not in shape of " + \
                                           "<H,W> or <H,W,C>, but got rank: 4"):
        random_vertical_flip_op(image)

    # RandomVerticalFlip operator:Test input is numpy list
    image = np.random.randint(0, 255, (255, 255, 3)).astype(np.uint8).tolist()
    random_vertical_flip_op = vision.RandomVerticalFlip(1)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>"):
        random_vertical_flip_op(image)

    # RandomVerticalFlip operator:Test input 1d numpy array
    image = np.random.randint(0, 255, (50,)).astype(np.uint8)
    prob = 1
    random_vertical_flip_op = vision.RandomVerticalFlip(prob)
    with pytest.raises(RuntimeError,
                       match="RandomVerticalFlip: input tensor is not in shape of <H,W> or " + \
                             "<H,W,C>, but got rank: 1. You may need to perform Decode first"):
        random_vertical_flip_op(np.array(image))

    # RandomVerticalFlip operator:Test input a parameter that does not exist
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'test'"):
        vision.RandomVerticalFlip(test='test')


if __name__ == "__main__":
    test_random_vertical_op(plot=True)
    test_random_vertical_valid_prob_c()
    test_random_vertical_valid_prob_py()
    test_random_vertical_invalid_prob_c()
    test_random_vertical_invalid_prob_py()
    test_random_vertical_comp(plot=True)
    test_random_vertical_op_1()
    test_random_vertical_flip_invalid_data()
    test_random_vertical_flip_operation_01()
    test_random_vertical_flip_operation_02()
    test_random_vertical_flip_exception_01()
