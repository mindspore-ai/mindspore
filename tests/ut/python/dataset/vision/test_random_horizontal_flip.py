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
Testing the random horizontal flip op in DE
"""
import cv2
import numpy as np
import os
import pytest
from PIL import Image

from mindspore import log as logger
import mindspore.dataset as ds
import mindspore.dataset.transforms.transforms as trans
import mindspore.dataset.vision.transforms as vision
from util import save_and_check_md5, save_and_check_md5_pil, visualize_list, visualize_image, diff_mse, \
    config_get_set_seed, config_get_set_num_parallel_workers

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
FOUR_DIM_DATA = [[[[1, 2, 3], [3, 4, 3]], [[5, 6, 3], [7, 8, 3]]],
                 [[[9, 10, 3], [11, 12, 3]], [[13, 14, 3], [15, 16, 3]]]]
FIVE_DIM_DATA = [[[[[1, 2, 3], [3, 4, 3]], [[5, 6, 3], [7, 8, 3]]],
                  [[[9, 10, 3], [11, 12, 3]], [[13, 14, 3], [15, 16, 3]]]]]
FOUR_DIM_RES = [[[[3.0, 4.0, 3.0], [1.0, 2.0, 3.0]], [[7.0, 8.0, 3.0], [5.0, 6.0, 3.0]]],
                [[[11.0, 12.0, 3.0], [9.0, 10.0, 3.0]], [[15.0, 16.0, 3.0], [13.0, 14.0, 3.0]]]]
FIVE_DIM_RES = [[[[[3.0, 4.0, 3.0], [1.0, 2.0, 3.0]], [[7.0, 8.0, 3.0], [5.0, 6.0, 3.0]]],
                 [[[11.0, 12.0, 3.0], [9.0, 10.0, 3.0]], [[15.0, 16.0, 3.0], [13.0, 14.0, 3.0]]]]]

TEST_DATA_DATASET_FUNC ="../data/dataset/"

DATA_DIR_1 = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")


def generator_mc(maxid=3):
    """ Multi-column generator function as callable input """
    image_obj = Image.open(image_jpg)
    image1_obj = Image.open(image_bmp)
    image2_obj = Image.open(image_gif)
    image = np.array(image_obj)
    image1 = np.array(image1_obj)
    image2 = np.array(image2_obj)
    for _ in range(maxid):
        yield image, image, image1, image1, image2
    image_obj.close()
    image1_obj.close()
    image2_obj.close()


def h_flip(image):
    """
    Apply the random_horizontal
    """

    # with the seed provided in this test case, it will always flip.
    # that's why we flip here too
    image = image[:, ::-1, :]
    return image


def test_random_horizontal_op(plot=False):
    """
    Feature: RandomHorizontalFlip op
    Description: Test RandomHorizontalFlip with default probability
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_horizontal_op")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    decode_op = vision.Decode()
    random_horizontal_op = vision.RandomHorizontalFlip(1.0)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=random_horizontal_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    data2 = data2.map(operations=decode_op, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):

        # with the seed value, we can only guarantee the first number generated
        if num_iter > 0:
            break

        image_h_flipped = item1["image"]
        image = item2["image"]
        image_h_flipped_2 = h_flip(image)

        mse = diff_mse(image_h_flipped, image_h_flipped_2)
        assert mse == 0
        logger.info("image_{}, mse: {}".format(num_iter + 1, mse))
        num_iter += 1
        if plot:
            visualize_image(image, image_h_flipped, mse, image_h_flipped_2)


def test_random_horizontal_valid_prob_c():
    """
    Feature: RandomHorizontalFlip op
    Description: Test RandomHorizontalFlip with Cpp implementation using valid non-default input
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_horizontal_valid_prob_c")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    decode_op = vision.Decode()
    random_horizontal_op = vision.RandomHorizontalFlip(0.8)
    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=random_horizontal_op, input_columns=["image"])

    filename = "random_horizontal_01_c_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_horizontal_valid_prob_py():
    """
    Feature: RandomHorizontalFlip op
    Description: Test RandomHorizontalFlip with Python implementation using valid non-default input
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_horizontal_valid_prob_py")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    transforms = [
        vision.Decode(True),
        vision.RandomHorizontalFlip(0.8),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])

    filename = "random_horizontal_01_py_result.npz"
    save_and_check_md5_pil(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_horizontal_invalid_prob_c():
    """
    Feature: RandomHorizontalFlip op
    Description: Test RandomHorizontalFlip with Cpp implementation using invalid input
    Expectation: Error is raised as expected
    """
    logger.info("test_random_horizontal_invalid_prob_c")

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    decode_op = vision.Decode()
    try:
        # Note: Valid range of prob should be [0.0, 1.0]
        random_horizontal_op = vision.RandomHorizontalFlip(1.5)
        data = data.map(operations=decode_op, input_columns=["image"])
        data = data.map(operations=random_horizontal_op,
                        input_columns=["image"])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input prob is not within the required interval of [0.0, 1.0]." in str(
            e)


def test_random_horizontal_invalid_prob_py():
    """
    Feature: RandomHorizontalFlip op
    Description: Test RandomHorizontalFlip with Python implementation using invalid input
    Expectation: Error is raised as expected
    """
    logger.info("test_random_horizontal_invalid_prob_py")

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)

    try:
        transforms = [
            vision.Decode(True),
            # Note: Valid range of prob should be [0.0, 1.0]
            vision.RandomHorizontalFlip(1.5),
            vision.ToTensor()
        ]
        transform = trans.Compose(transforms)
        data = data.map(operations=transform, input_columns=["image"])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input prob is not within the required interval of [0.0, 1.0]." in str(
            e)


def test_random_horizontal_comp(plot=False):
    """
    Feature: RandomHorizontalFlip op
    Description: Test RandomHorizontalFlip op and compare between Python and Cpp image augmentation ops
    Expectation: Resulting datasets from the ops are expected to be the same
    """
    logger.info("test_random_horizontal_comp")
    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    decode_op = vision.Decode()
    # Note: The image must be flipped if prob is set to be 1
    random_horizontal_op = vision.RandomHorizontalFlip(1)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=random_horizontal_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    transforms = [
        vision.Decode(True),
        # Note: The image must be flipped if prob is set to be 1
        vision.RandomHorizontalFlip(1),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
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


def test_random_horizontal_op_1():
    """
    Feature: RandomHorizontalFlip op
    Description: Test RandomHorizontalFlip with different fields
    Expectation: The dataset is processed as expected
    """
    logger.info("Test RandomHorizontalFlip with different fields.")

    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    data = data.map(operations=trans.Duplicate(), input_columns=["image"],
                    output_columns=["image", "image_copy"])
    random_horizontal_op = vision.RandomHorizontalFlip(1.0)
    decode_op = vision.Decode()

    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=decode_op, input_columns=["image_copy"])
    data = data.map(operations=random_horizontal_op,
                    input_columns=["image", "image_copy"])

    num_iter = 0
    for data1 in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = data1["image"]
        image_copy = data1["image_copy"]
        mse = diff_mse(image, image_copy)
        logger.info("image_{}, mse: {}".format(num_iter + 1, mse))
        assert mse == 0
        num_iter += 1


def test_random_horizontal_flip_invalid_data():
    """
    Feature: RandomHorizontalFlip op
    Description: Test RandomHorizontalFlip with invalid data
    Expectation: Error is raised as expected
    """

    invalid_type_img = np.random.random((32, 32, 3)).astype(np.str_)
    invalid_shape_img = np.random.random(32).astype(np.float32)
    random_horizontal_flip = vision.RandomHorizontalFlip(0.1)

    with pytest.raises(RuntimeError) as error_info:
        random_horizontal_flip(invalid_type_img)
    assert "RandomHorizontalFlip: the data type of image tensor does not match the requirement of operator." \
           in str(error_info.value)

    with pytest.raises(RuntimeError) as error_info:
        random_horizontal_flip(invalid_shape_img)
    assert "RandomHorizontalFlip: the image tensor should have at least two dimensions. You may need to perform " \
           "Decode first." in str(error_info.value)


def test_random_horizontal_flip_video_op_1d_c():
    """
    Feature: RandomHorizontalFlip op
    Description: Test RandomHorizontalFlip op by processing tensor with dim 1
    Expectation: Error is raised as expected
    """
    logger.info("Test RandomHorizontalFlip with 1 dimension input")
    data = [1]
    input_mindspore = np.array(data).astype(np.uint8)
    random_horizontal_op = vision.RandomHorizontalFlip(1.0)
    try:
        random_horizontal_op(input_mindspore)
    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "RandomHorizontalFlip: the image tensor should have at least two dimensions. You may need to perform " \
               "Decode first." in str(e)


def test_random_horizontal_flip_video_op_4d_c():
    """
    Feature: RandomHorizontalFlip op
    Description: Test RandomHorizontalFlip op by processing tensor with dim more than 3 (dim 4)
    Expectation: The dataset is processed successfully
    """
    logger.info("Test RandomHorizontalFlip with 4 dimension input")
    input_4_dim = np.array(FOUR_DIM_DATA).astype(np.uint8)
    input_4_shape = input_4_dim.shape
    n_num = input_4_dim.size // (input_4_shape[-2] * input_4_shape[-1])
    input_3_dim = input_4_dim.reshape([n_num, input_4_shape[-2], input_4_shape[-1]])

    random_horizontal_op = vision.RandomHorizontalFlip(1.0)
    out_4_dim = random_horizontal_op(input_4_dim)
    out_3_dim = random_horizontal_op(input_3_dim)
    out_3_dim = out_3_dim.reshape(input_4_shape)

    mse = diff_mse(out_4_dim, out_3_dim)
    assert mse < 0.001


def test_random_horizontal_flip_video_op_5d_c():
    """
    Feature: RandomHorizontalFlip op
    Description: Test RandomHorizontalFlip op by processing tensor with dim more than 3 (dim 5)
    Expectation: The dataset is processed successfully
    """
    logger.info("Test RandomHorizontalFlip with 5 dimension input")
    input_5_dim = np.array(FIVE_DIM_DATA).astype(np.uint8)
    input_5_shape = input_5_dim.shape
    n_num = input_5_dim.size // (input_5_shape[-2] * input_5_shape[-1])
    input_3_dim = input_5_dim.reshape([n_num, input_5_shape[-2], input_5_shape[-1]])

    random_horizontal_op = vision.RandomHorizontalFlip(1.0)
    out_5_dim = random_horizontal_op(input_5_dim)
    out_3_dim = random_horizontal_op(input_3_dim)
    out_3_dim = out_3_dim.reshape(input_5_shape)

    mse = diff_mse(out_5_dim, out_3_dim)
    assert mse < 0.001


def test_random_horizontal_flip_video_op_precision_eager_c():
    """
    Feature: RandomHorizontalFlip op
    Description: Test RandomHorizontalFlip op by processing tensor with dim more than 3 (dim 4) in eager mode
    Expectation: The dataset is processed successfully
    """
    logger.info("Test RandomHorizontalFlip eager with 4 dimension input")
    input_mindspore = np.array(FOUR_DIM_DATA).astype(np.uint8)

    random_horizontal_op = vision.RandomHorizontalFlip(1.0)
    out_mindspore = random_horizontal_op(input_mindspore)
    mse = diff_mse(out_mindspore, np.array(FOUR_DIM_RES).astype(np.uint8))
    assert mse < 0.001


def test_random_horizontal_flip_video_op_precision_pipeline_c():
    """
    Feature: RandomHorizontalFlip op
    Description: Test RandomHorizontalFlip op by processing tensor with dim more than 3 (dim 5) in eager mode
    Expectation: The dataset is processed successfully
    """
    logger.info("Test RandomHorizontalFlip pipeline with 5 dimension input")
    data = np.array(FIVE_DIM_DATA).astype(np.uint8)
    expand_data = np.expand_dims(data, axis=0)

    dataset = ds.NumpySlicesDataset(expand_data, column_names=["col1"], shuffle=False)
    random_horizontal_op = vision.RandomHorizontalFlip(1.0)
    dataset = dataset.map(input_columns=["col1"], operations=random_horizontal_op)
    for item in dataset.create_dict_iterator(output_numpy=True):
        mse = diff_mse(item["col1"], np.array(FIVE_DIM_RES).astype(np.uint8))
        assert mse < 0.001


def test_random_horizontal_flip_operation_01():
    """
    Feature: RandomHorizontalFlip operation
    Description: Testing the normal functionality of the RandomHorizontalFlip operator
    Expectation: The Output is equal to the expected output
    """
    # When parameter prob is 0.0, RandomHorizontalFlip interface is successfully called
    prob = 0.0
    ds2 = ds.ImageFolderDataset(DATA_DIR_1, 1)
    transforms1 = [
        vision.Decode(),
        vision.RandomHorizontalFlip(prob),
        vision.ToTensor()
    ]
    transform1 = trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)
    for _ in ds2.create_dict_iterator(output_numpy=True):
        pass

    # When parameter prob is 0, RandomHorizontalFlip interface is successfully called
    dataset2 = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    prob = 0
    random_horizontal_flip_op = vision.RandomHorizontalFlip(prob=prob)
    dataset2 = dataset2.map(input_columns=["image"], operations=random_horizontal_flip_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # When parameter prob is 0.8, RandomHorizontalFlip interface is successfully called
    source = generator_mc
    column_names = ["image1", "image2", "image3", "image4", "image5"]
    dataset = ds.GeneratorDataset(source, column_names)
    prob = 0.8
    horizontal_flip_op = vision.RandomHorizontalFlip(prob=prob)
    dataset = dataset.map(input_columns=["image1", "image2", "image3", "image4", "image5"],
                          operations=horizontal_flip_op)
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["image1"] == data["image2"]).all()
        assert (data["image3"] == data["image4"]).all()

    # When no parameters are set, RandomHorizontalFlip interface is successfully called
    source = generator_mc
    column_names = ["image1", "image2", "image3", "image4", "image5"]
    dataset = ds.GeneratorDataset(source, column_names)
    horizontal_flip_op = vision.RandomHorizontalFlip()
    dataset = dataset.map(input_columns=["image1", "image2", "image5"], operations=horizontal_flip_op)
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["image1"] == data["image2"]).all()

    # When setting extra parameters, RandomHorizontalFlip interface call fails
    image = cv2.imread(image_jpg)
    with Image.open(image_bmp) as image2:
        prob = 1
        random_horizontal_flip_op = vision.RandomHorizontalFlip(prob)
        _ = random_horizontal_flip_op(image, image2)

    # When input image is bmp, RandomHorizontalFlip interface is successfully called
    with Image.open(image_bmp) as image:
        prob = 0
        random_horizontal_flip_op = vision.RandomHorizontalFlip(prob)
        _ = random_horizontal_flip_op(image)


def test_random_horizontal_flip_operation_02():
    """
    Feature: RandomHorizontalFlip operation
    Description: Testing the normal functionality of the RandomHorizontalFlip operator
    Expectation: The Output is equal to the expected output
    """
    # Call RandomHorizontalFlip interface, output data out[0]/out[1] are the same
    with Image.open(image_gif) as image:
        image2 = np.random.randint(0, 255, (256, 256, 3)).astype(np.uint8)
        random_horizontal_flip_op = vision.RandomHorizontalFlip()
        out = random_horizontal_flip_op(image, image, image2)
        assert (out[0] == out[1]).all()

    # When parameter prob is 0.50689, RandomHorizontalFlip interface is successfully called
    with Image.open(image_png) as image:
        prob = 0.50689
        random_horizontal_flip_op = vision.RandomHorizontalFlip(prob)
        _ = random_horizontal_flip_op(image)

    # When input data is 3-dimensional, RandomHorizontalFlip interface is successfully called
    image = np.random.randint(0, 255, (256, 256, 3)).astype(np.uint8)
    prob = 0.0856
    random_horizontal_flip_op = vision.RandomHorizontalFlip(prob)
    _ = random_horizontal_flip_op(image)

    # When input channel is 1, RandomHorizontalFlip interface is successfully called
    image = np.random.randint(0, 255, (256, 256, 1)).astype(np.uint8)
    prob = 0.999999
    random_horizontal_flip_op = vision.RandomHorizontalFlip(prob)
    random_horizontal_flip_op(image)

    # When input data is 2-dimensional, RandomHorizontalFlip interface is successfully called
    image = np.random.randint(0, 255, (256, 512)).astype(np.uint8)
    prob = 0.35
    random_horizontal_flip_op = vision.RandomHorizontalFlip(prob)
    _ = random_horizontal_flip_op(image)


def test_random_horizontal_flip_exception_01():
    """
    Feature: RandomHorizontalFlip operation
    Description: Testing the RandomHorizontalFlip Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When parameter prob is greater than 1, RandomHorizontalFlip interface call fails
    dataset2 = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    prob = 1.1
    with pytest.raises(ValueError, match="Input prob is not within the required interval"):
        random_horizontal_flip_op = vision.RandomHorizontalFlip(prob=prob)
        dataset2 = dataset2.map(input_columns=["image"], operations=random_horizontal_flip_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # When parameter prob is negative, RandomHorizontalFlip interface call fails
    dataset2 = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    prob = -1
    with pytest.raises(ValueError, match="Input prob is not within the required interval"):
        random_horizontal_flip_op = vision.RandomHorizontalFlip(prob=prob)
        dataset2 = dataset2.map(input_columns=["image"], operations=random_horizontal_flip_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # When parameter prob is string, RandomHorizontalFlip interface call fails
    dataset2 = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    prob = ""
    with pytest.raises(TypeError, match="Argument prob"):
        random_horizontal_flip_op = vision.RandomHorizontalFlip(prob=prob)
        dataset2 = dataset2.map(input_columns=["image"], operations=random_horizontal_flip_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # When setting extra parameters, RandomHorizontalFlip interface call fails
    dataset2 = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    prob = 1
    more_para = None
    with pytest.raises(TypeError, match="too many positional arguments"):
        random_horizontal_flip_op = vision.RandomHorizontalFlip(prob, more_para)
        dataset2 = dataset2.map(input_columns=["image"], operations=random_horizontal_flip_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # When input data is 1-dimensional, RandomHorizontalFlip interface call fails
    image = np.random.randint(0, 255, (512,)).astype(np.uint8)
    prob = 0.999999
    random_horizontal_flip_op = vision.RandomHorizontalFlip(prob)
    with pytest.raises(RuntimeError,
                       match="RandomHorizontalFlip: the image tensor should have at least "
                             "two dimensions. You may need to perform Decode first."):
        random_horizontal_flip_op(image)

    # When parameter prob is -0.1, RandomHorizontalFlip interface call fails
    prob = -0.1
    with pytest.raises(ValueError, match="Input prob is not within the required interval of \\[0.0, 1.0\\]."):
        vision.RandomHorizontalFlip(prob)

    # When parameter prob is 1.01, RandomHorizontalFlip interface call fails
    prob = 1.01
    with pytest.raises(ValueError, match="Input prob is not within the required interval of \\[0.0, 1.0\\]."):
        vision.RandomHorizontalFlip(prob)

    # When parameter prob is list, RandomHorizontalFlip interface call fails
    prob = [0.5]
    with pytest.raises(TypeError,
                       match="Argument prob with value \\[0.5\\] is not of type \\[<class 'float'>, <class 'int'>\\]."):
        vision.RandomHorizontalFlip(prob)

    # When parameter prob is 1-tuple, RandomHorizontalFlip interface call fails
    prob = (0.5,)
    with pytest.raises(TypeError,
                       match="Argument prob with value \\(0.5,\\) is not of type \\[<class 'float'>, <class 'int'>\\]"):
        vision.RandomHorizontalFlip(prob)

    # When input data is empty, RandomHorizontalFlip interface call fails
    prob = 1
    random_horizontal_flip_op = vision.RandomHorizontalFlip(prob)
    with pytest.raises(RuntimeError, match="Input Tensor is not valid"):
        random_horizontal_flip_op()

    # When input data is list, RandomHorizontalFlip interface call fails
    image = np.random.randint(0, 255, (52, 52, 3)).astype(np.uint8)
    image = image.tolist()
    prob = 1
    random_horizontal_flip_op = vision.RandomHorizontalFlip(prob)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        random_horizontal_flip_op(image)


def test_random_horizontal_flip_exception_02():
    """
    Feature: RandomHorizontalFlip operation
    Description: Testing the RandomHorizontalFlip Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When input data has np array, RandomHorizontalFlip interface call fails
    image = np.random.randint(0, 255, (50, 50, 3)).astype(np.uint8)
    prob = 1
    random_horizontal_flip_op = vision.RandomHorizontalFlip(prob)
    with pytest.raises(RuntimeError):
        random_horizontal_flip_op(np.array(10), image)

    # When input data is int64, RandomHorizontalFlip interface call fails
    image = np.random.randint(0, 255, (50, 50, 3)).astype(np.int64)
    random_horizontal_flip_op = vision.RandomHorizontalFlip(0.8)
    with pytest.raises(RuntimeError, match=r"RandomHorizontalFlip: the data type of image "
                                           r"tensor does not match the requirement of operator. Expecting "
                                           r"tensor in type of \(bool, int8, uint8, int16, uint16, int32, "
                                           r"float16, float32, float64\). But got type int64."):
        random_horizontal_flip_op(image)

    # Test normal："prob", [(0.5,), [0.5, 0.9]]
    prob = (0.5, 1)
    with pytest.raises(TypeError, match=r'is not of type \[\<class \'float\'\>, \<class \'int\'\>\].'):
        vision.RandomHorizontalFlip(prob)


if __name__ == "__main__":
    test_random_horizontal_op(plot=True)
    test_random_horizontal_valid_prob_c()
    test_random_horizontal_valid_prob_py()
    test_random_horizontal_invalid_prob_c()
    test_random_horizontal_invalid_prob_py()
    test_random_horizontal_comp(plot=True)
    test_random_horizontal_op_1()
    test_random_horizontal_flip_invalid_data()
    test_random_horizontal_flip_video_op_1d_c()
    test_random_horizontal_flip_video_op_4d_c()
    test_random_horizontal_flip_video_op_5d_c()
    test_random_horizontal_flip_video_op_precision_eager_c()
    test_random_horizontal_flip_video_op_precision_pipeline_c()
    test_random_horizontal_flip_operation_01()
    test_random_horizontal_flip_operation_02()
    test_random_horizontal_flip_exception_01()
    test_random_horizontal_flip_exception_02()
