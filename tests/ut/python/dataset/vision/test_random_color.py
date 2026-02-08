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
Testing RandomColor op in DE
"""
import numpy as np
import os
import pytest
from PIL import Image

import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.transforms.transforms as t_trans
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger
from util import visualize_list, diff_mse, save_and_check_md5, save_and_check_md5_pil, \
    config_get_set_seed, config_get_set_num_parallel_workers

DATA_DIR = "../data/dataset/testImageNetData/train/"
C_DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
C_SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
MNIST_DATA_DIR = "../data/dataset/testMnistData"
TEST_DATA_DATASET_FUNC ="../data/dataset/"

image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")

GENERATE_GOLDEN = False


def test_random_color_py(degrees=(0.1, 1.9), plot=False):
    """
    Feature: RandomColor op
    Description: Test RandomColor op with Python implementation
    Expectation: The dataset is processed as expected
    """
    logger.info("Test RandomColor")

    # Original Images
    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transforms_original = t_trans.Compose([vision.Decode(True), vision.Resize((224, 224)), vision.ToTensor()])

    ds_original = data.map(operations=transforms_original, input_columns="image")

    ds_original = ds_original.batch(512)

    for idx, (image, _) in enumerate(ds_original):
        if idx == 0:
            images_original = np.transpose(image.asnumpy(), (0, 2, 3, 1))
        else:
            images_original = np.append(images_original,
                                        np.transpose(image.asnumpy(), (0, 2, 3, 1)),
                                        axis=0)

            # Random Color Adjusted Images
    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transforms_random_color = t_trans.Compose([vision.Decode(True),
                                               vision.Resize((224, 224)),
                                               vision.RandomColor(degrees=degrees),
                                               vision.ToTensor()])

    ds_random_color = data.map(operations=transforms_random_color, input_columns="image")

    ds_random_color = ds_random_color.batch(512)

    for idx, (image, _) in enumerate(ds_random_color):
        if idx == 0:
            images_random_color = np.transpose(image.asnumpy(), (0, 2, 3, 1))
        else:
            images_random_color = np.append(images_random_color,
                                            np.transpose(image.asnumpy(), (0, 2, 3, 1)),
                                            axis=0)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_random_color[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))

    if plot:
        visualize_list(images_original, images_random_color)


def test_random_color_c(degrees=(0.1, 1.9), plot=False, run_golden=True):
    """
    Feature: RandomColor op
    Description: Test RandomColor op with Cpp implementation
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_color_op")

    original_seed = config_get_set_seed(10)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Decode with rgb format set to True
    data1 = ds.TFRecordDataset(C_DATA_DIR, C_SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = ds.TFRecordDataset(C_DATA_DIR, C_SCHEMA_DIR, columns_list=["image"], shuffle=False)

    # Serialize and Load dataset requires using vision.Decode instead of vision.Decode().
    if degrees is None:
        c_op = vision.RandomColor()
    else:
        c_op = vision.RandomColor(degrees)

    data1 = data1.map(operations=[vision.Decode()], input_columns=["image"])
    data2 = data2.map(operations=[vision.Decode(), c_op], input_columns=["image"])

    image_random_color_op = []
    image = []

    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        actual = item1["image"]
        expected = item2["image"]
        image.append(actual)
        image_random_color_op.append(expected)

    if run_golden:
        # Compare with expected md5 from images
        filename = "random_color_op_02_result.npz"
        save_and_check_md5(data2, filename, generate_golden=GENERATE_GOLDEN)

    if plot:
        visualize_list(image, image_random_color_op)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers((original_num_parallel_workers))


def test_random_color_py_md5():
    """
    Feature: RandomColor op
    Description: Test RandomColor op with Python implementation with md5 check
    Expectation: Passes the md5 check test
    """
    logger.info("Test RandomColor with md5 check")
    original_seed = config_get_set_seed(10)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transforms = t_trans.Compose([vision.Decode(True),
                                  vision.RandomColor((2.0, 2.5)),
                                  vision.ToTensor()])

    data = data.map(operations=transforms, input_columns="image")
    # Compare with expected md5 from images
    filename = "random_color_01_result.npz"
    save_and_check_md5_pil(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers((original_num_parallel_workers))


def test_compare_random_color_op(degrees=None, plot=False):
    """
    Feature: RandomColor op
    Description: Test RandomColor op and compare between Python and Cpp implementation
    Expectation: Resulting datasets from both operations are expected to be the same
    """

    logger.info("test_random_color_op")

    original_seed = config_get_set_seed(5)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Decode with rgb format set to True
    data1 = ds.TFRecordDataset(C_DATA_DIR, C_SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = ds.TFRecordDataset(C_DATA_DIR, C_SCHEMA_DIR, columns_list=["image"], shuffle=False)

    if degrees is None:
        c_op = vision.RandomColor()
        p_op = vision.RandomColor()
    else:
        c_op = vision.RandomColor(degrees)
        p_op = vision.RandomColor(degrees)

    transforms_random_color_py = t_trans.Compose(
        [lambda img: img.astype(np.uint8), vision.ToPIL(),
         p_op, np.array])

    data1 = data1.map(operations=[vision.Decode(), c_op], input_columns=["image"])
    data2 = data2.map(operations=[vision.Decode()], input_columns=["image"])
    data2 = data2.map(operations=transforms_random_color_py, input_columns=["image"])

    image_random_color_op = []
    image = []

    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        actual = item1["image"]
        expected = item2["image"]
        image_random_color_op.append(actual)
        image.append(expected)
        assert actual.shape == expected.shape
        mse = diff_mse(actual, expected)
        logger.info("MSE= {}".format(str(np.mean(mse))))

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)

    if plot:
        visualize_list(image, image_random_color_op)


def test_random_color_c_errors():
    """
    Feature: RandomColor op
    Description: Test error cases RandomColor op with Cpp implementation
    Expectation: Correct error is thrown as expected
    """
    with pytest.raises(TypeError) as error_info:
        vision.RandomColor((12))
    assert "degrees must be either a tuple or a list." in str(error_info.value)

    with pytest.raises(TypeError) as error_info:
        vision.RandomColor(("col", 3))
    assert "Argument degrees[0] with value col is not of type [<class 'int'>, <class 'float'>]" in str(
        error_info.value)

    with pytest.raises(ValueError) as error_info:
        vision.RandomColor((0.9, 0.1))
    assert "degrees should be in (min,max) format. Got (max,min)." in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        vision.RandomColor((0.9,))
    assert "degrees must be a sequence with length 2." in str(error_info.value)

    # RandomColor Cpp Op will fail with one channel input
    mnist_ds = ds.MnistDataset(dataset_dir=MNIST_DATA_DIR, num_samples=2, shuffle=False)
    mnist_ds = mnist_ds.map(operations=vision.RandomColor(), input_columns="image")

    with pytest.raises(RuntimeError) as error_info:
        for _ in enumerate(mnist_ds):
            pass
    assert "image shape is not <H,W,C> or channel is not 3" in str(error_info.value)


def test_random_color_operation_01():
    """
    Feature: RandomColor operation
    Description: Testing the normal functionality of the RandomColor operator
    Expectation: The Output is equal to the expected output
    """
    # When the parameter degrees is within the range (0, 1.5), the RandomColor interface call succeeds.
    degrees = (0, 1.5)
    dataset2 = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=True)
    random_color_op = vision.RandomColor(degrees=degrees)
    dataset2 = dataset2.map(input_columns=["image"], operations=random_color_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # When no parameter value is set, the RandomColor interface call succeeds.
    dataset = ds.ImageFolderDataset(DATA_DIR)
    transforms1 = [
        vision.Decode(),
        vision.RandomColor(),
        vision.ToTensor()
    ]
    transform1 = t_trans.Compose(transforms1)
    dataset = dataset.map(input_columns=["image"], operations=transform1)

    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When the parameter degrees is within the range (0.1, 1), the RandomColor interface call succeeds.
    degrees = (0.1, 1)
    dataset = ds.ImageFolderDataset(DATA_DIR)
    transforms1 = [
        vision.Decode(),
        vision.RandomColor(degrees=degrees),
        vision.ToTensor()
    ]
    transform1 = t_trans.Compose(transforms1)
    dataset = dataset.map(input_columns=["image"], operations=transform1)

    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When the parameter degrees is within the range [0.5, 1.5], the RandomColor interface call succeeds.
    degrees = [0.5, 1.5]
    dataset = ds.ImageFolderDataset(DATA_DIR)
    transforms1 = [
        vision.Decode(),
        vision.RandomColor(degrees=degrees),
        vision.ToTensor()
    ]
    transform1 = t_trans.Compose(transforms1)
    dataset = dataset.map(input_columns=["image"], operations=transform1)

    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass


def test_random_color_operation_02():
    """
    Feature: RandomColor operation
    Description: Testing the normal functionality of the RandomColor operator
    Expectation: The Output is equal to the expected output
    """
    # When the parameter degrees is (0, 0), the RandomColor interface call succeeds.
    degrees = (0, 0)
    dataset = ds.ImageFolderDataset(DATA_DIR)
    transforms1 = [
        vision.Decode(),
        vision.RandomColor(degrees=degrees),
        vision.ToTensor()
    ]
    transform1 = t_trans.Compose(transforms1)
    dataset = dataset.map(input_columns=["image"], operations=transform1)

    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When the parameter degrees is [1, 1], the RandomColor interface call succeeds.
    degrees = [1, 1]
    dataset2 = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=True)
    random_color_op = vision.RandomColor(degrees=degrees)
    dataset2 = dataset2.map(input_columns=["image"], operations=random_color_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # When the input image format is JPG, the RandomColor interface call succeeds.
    with Image.open(image_jpg) as image:
        degrees = (0.3, 18.6)
        random_color_op = vision.RandomColor(degrees=degrees)
        _ = random_color_op(image)

    # When the input image format is BMP, the RandomColor interface call succeeds.
    with Image.open(image_bmp) as image:
        degrees = (16777216, 16777216)
        random_color_op = vision.RandomColor(degrees=degrees)
        _ = random_color_op(image)

    # When the input image format is PNG, the RandomColor interface call succeeds.
    with Image.open(image_png) as image:
        degrees = [0, 16777216]
        random_color_op = vision.RandomColor(degrees=degrees)
        _ = random_color_op(image)

    # When the input shape is (128, 256, 3), the RandomColor interface call succeeds.
    image = np.random.randint(0, 255, (128, 256, 3)).astype(np.uint8)
    image = vision.ToPIL()(image)
    degrees = [10.3, 12.6]
    random_color_op = vision.RandomColor(degrees=degrees)
    _ = random_color_op(image)

    # When the input shape is (1024, 2048), the RandomColor interface call succeeds.
    image = np.random.randint(0, 255, (1024, 2048)).astype(np.uint8)
    image = vision.ToPIL()(image)
    degrees = [0.1, 0.6]
    random_color_op = vision.RandomColor(degrees=degrees)
    random_color_op(image)

    # When input data is a numpy array, the RandomColor interface call succeeds.
    image = np.random.randint(0, 255, (128, 256, 3)).astype(np.uint8)
    degrees = [1, 1]
    random_color_op = vision.RandomColor(degrees=degrees)
    _ = random_color_op(image)


def test_random_color_exception_01():
    """
    Feature: RandomColor operation
    Description: Testing the RandomColor Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When the degrees parameter is negative, the RandomColor interface call fails.
    degrees = (-0.5, 1.5)
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        vision.RandomColor(degrees=degrees)

    # When the parameter degrees is in the range [-1, 2), the RandomColor interface call fails.
    degrees = [-1, 2]
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        vision.RandomColor(degrees=degrees)

    # When the parameter "degrees" is an integer, the RandomColor interface call fails.
    degrees = 2
    with pytest.raises(TypeError, match="degrees must be either a tuple or a list."):
        vision.RandomColor(degrees=degrees)

    # When the parameter "degrees" is a 3-tuple, the RandomColor interface call fails.
    degrees = (0, 0.5, 1.5)
    with pytest.raises(ValueError, match="degrees must be a sequence with length 2"):
        vision.RandomColor(degrees=degrees)

    # When the input image format is GIF, the RandomColor interface call fails.
    with Image.open(image_gif) as image:
        degrees = [1, 1]
        random_color_op = vision.RandomColor(degrees=degrees)
        with pytest.raises(ValueError):
            random_color_op(image)

    # When input data is a list, the RandomColor interface call fails.
    image = np.random.randint(0, 255, (128, 256, 3)).astype(np.uint8).tolist()
    degrees = [1, 1]
    random_color_op = vision.RandomColor(degrees=degrees)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        random_color_op(image)

    # When the degrees parameter is a numpy array, the RandomColor interface call fails.
    degrees = np.array([10.3, 12.6])
    with pytest.raises(TypeError, match="degrees must be either a tuple or a list."):
        vision.RandomColor(degrees=degrees)

    # When the parameter "degrees" has a length of 1, the RandomColor interface call fails.
    degrees = [10.3]
    with pytest.raises(ValueError, match="degrees must be a sequence with length 2."):
        vision.RandomColor(degrees=degrees)

    # When the parameter "degrees" is set, the RandomColor interface call fails.
    degrees = {10.3, 10.5}
    with pytest.raises(TypeError, match="degrees must be either a tuple or a list."):
        vision.RandomColor(degrees=degrees)

    # The first value of the parameter degrees is greater than the second value.
    degrees = [10.3, 10.2]
    with pytest.raises(ValueError, match="degrees should be in \\(min,max\\) format. Got \\(max,min\\)."):
        vision.RandomColor(degrees=degrees)

    # When the parameter degrees exceeds 16777216, the RandomColor interface call fails.
    degrees = [10.3, 16777216.1]
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[0, 16777216\\]."):
        vision.RandomColor(degrees=degrees)

    # When the input data channel is set to 4, the RandomColor interface call fails.
    image = np.random.randn(128, 128, 4)
    degrees = (0.2, 0.3)
    random_color_op = vision.RandomColor(degrees=degrees)
    with pytest.raises(RuntimeError, match="image shape is not <H,W,C> or channel is not 3."):
        random_color_op(image)

    # When input data is 4-dimensional, the RandomColor interface call fails.
    image = np.random.randn(128, 128, 3, 3)
    degrees = (0.2, 0.3)
    random_color_op = vision.RandomColor(degrees=degrees)
    with pytest.raises(RuntimeError, match="image shape is not <H,W,C> or channel is not 3."):
        random_color_op(image)

    # When input data is 1-dimensional, the RandomColor interface call fails.
    image = np.random.randn(128,)
    degrees = (0.2, 0.3)
    random_color_op = vision.RandomColor(degrees=degrees)
    with pytest.raises(RuntimeError, match="image shape is not <H,W,C> or channel is not 3."):
        random_color_op(image)

    # When input data is a tuple, the RandomColor interface call fails.
    image = tuple(np.random.randn(128, 128, 3))
    degrees = (0.2, 0.3)
    random_color_op = vision.RandomColor(degrees=degrees)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'tuple'>."):
        random_color_op(image)

    # When input data is a Tensor, the RandomColor interface call fails.
    image = ms.Tensor(np.random.randn(128, 128, 3))
    degrees = (0.2, 0.3)
    random_color_op = vision.RandomColor(degrees=degrees)
    with pytest.raises(TypeError,
                       match="Input should be NumPy or PIL image, got <class 'mindspore.common.tensor.Tensor'>."):
        random_color_op(image)

    # When input data is two-dimensional, the RandomColor interface call fails.
    image = np.random.randn(128, 128)
    degrees = (0.2, 0.3)
    random_color_op = vision.RandomColor(degrees=degrees)
    with pytest.raises(RuntimeError, match="image shape is not <H,W,C> or channel is not 3."):
        random_color_op(image)


if __name__ == "__main__":
    test_random_color_py()
    test_random_color_py(plot=True)
    test_random_color_py(degrees=(2.0, 2.5), plot=True)  # Test with degree values that show more obvious transformation
    test_random_color_py_md5()

    test_random_color_c()
    test_random_color_c(plot=True)
    test_random_color_c(degrees=(2.0, 2.5), plot=True,
                        run_golden=False)  # Test with degree values that show more obvious transformation
    test_random_color_c(degrees=(0.1, 0.1), plot=True, run_golden=False)
    test_compare_random_color_op(plot=True)
    test_random_color_c_errors()
    test_random_color_operation_01()
    test_random_color_operation_02()
    test_random_color_exception_01()
