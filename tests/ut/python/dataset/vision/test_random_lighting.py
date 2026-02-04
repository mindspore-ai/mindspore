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

Testing RandomLighting op in DE
"""
import numpy as np
import os
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.transforms
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger
from util import visualize_list, diff_mse, save_and_check_md5_pil, \
    config_get_set_seed, config_get_set_num_parallel_workers

DATA_DIR = "../data/dataset/testImageNetData/train/"
MNIST_DATA_DIR = "../data/dataset/testMnistData"
TEST_DATA_DATASET_FUNC ="../data/dataset/"

GENERATE_GOLDEN = False


def test_random_lighting_py(alpha=1, plot=False):
    """
    Feature: RandomLighting
    Description: Test RandomLighting Python implementation
    Expectation: Equal results
    """
    logger.info("Test RandomLighting Python implementation")

    # Original Images
    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transforms_original = mindspore.dataset.transforms.Compose([vision.Decode(True),
                                                                vision.Resize((224, 224)),
                                                                vision.ToTensor()])

    ds_original = data.map(
        operations=transforms_original, input_columns="image")

    ds_original = ds_original.batch(512)

    for idx, (image, _) in enumerate(ds_original.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        if idx == 0:
            images_original = np.transpose(image, (0, 2, 3, 1))
        else:
            images_original = np.append(
                images_original, np.transpose(image, (0, 2, 3, 1)), axis=0)

    # Random Lighting Adjusted Images
    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    alpha = alpha if alpha is not None else 0.05
    py_op = vision.RandomLighting(alpha)

    transforms_random_lighting = mindspore.dataset.transforms.Compose([vision.Decode(True),
                                                                       vision.Resize((224, 224)),
                                                                       py_op,
                                                                       vision.ToTensor()])
    ds_random_lighting = data.map(
        operations=transforms_random_lighting, input_columns="image")

    ds_random_lighting = ds_random_lighting.batch(512)

    for idx, (image, _) in enumerate(ds_random_lighting.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        if idx == 0:
            images_random_lighting = np.transpose(image, (0, 2, 3, 1))
        else:
            images_random_lighting = np.append(
                images_random_lighting, np.transpose(image, (0, 2, 3, 1)), axis=0)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_random_lighting[i], images_original[i])

    logger.info("MSE= {}".format(str(np.mean(mse))))

    if plot:
        visualize_list(images_original, images_random_lighting)


def test_random_lighting_py_md5():
    """
    Feature: RandomLighting
    Description: Test RandomLighting Python implementation with md5 comparison
    Expectation: Same MD5
    """
    logger.info("Test RandomLighting Python implementation with md5 comparison")
    original_seed = config_get_set_seed(140)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # define map operations
    transforms = [
        vision.Decode(True),
        vision.Resize((224, 224)),
        vision.RandomLighting(1),
        vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.Compose(transforms)

    #  Generate dataset
    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
    data = data.map(operations=transform, input_columns=["image"])

    # check results with md5 comparison
    filename = "random_lighting_py_01_result.npz"
    save_and_check_md5_pil(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_lighting_c(alpha=1, plot=False):
    """
    Feature: RandomLighting
    Description: Test RandomLighting cpp op
    Expectation: Equal results from Mindspore and benchmark
    """
    logger.info("Test RandomLighting cpp op")
    # Original Images
    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transforms_original = [vision.Decode(), vision.Resize((224, 224))]

    ds_original = data.map(
        operations=transforms_original, input_columns="image")

    ds_original = ds_original.batch(512)

    for idx, (image, _) in enumerate(ds_original.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        if idx == 0:
            images_original = image
        else:
            images_original = np.append(images_original, image, axis=0)

    # Random Lighting Adjusted Images
    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    alpha = alpha if alpha is not None else 0.05
    c_op = vision.RandomLighting(alpha)

    transforms_random_lighting = [
        vision.Decode(), vision.Resize((224, 224)), c_op]

    ds_random_lighting = data.map(
        operations=transforms_random_lighting, input_columns="image")

    ds_random_lighting = ds_random_lighting.batch(512)

    for idx, (image, _) in enumerate(ds_random_lighting.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        if idx == 0:
            images_random_lighting = image
        else:
            images_random_lighting = np.append(
                images_random_lighting, image, axis=0)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_random_lighting[i], images_original[i])

    logger.info("MSE= {}".format(str(np.mean(mse))))

    if plot:
        visualize_list(images_original, images_random_lighting)


def test_random_lighting_c_py(alpha=1, plot=False):
    """
    Feature: RandomLighting
    Description: Test Random Lighting Cpp and Python Op
    Expectation: Equal results from Cpp and Python
    """
    logger.info("Test RandomLighting Cpp and python Op")

    # RandomLighting Images
    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
    data = data.map(operations=[vision.Decode(), vision.Resize(
        (200, 300))], input_columns=["image"])

    python_op = vision.RandomLighting(alpha)
    c_op = vision.RandomLighting(alpha)

    transforms_op = mindspore.dataset.transforms.Compose([lambda img: vision.ToPIL()(img.astype(np.uint8)),
                                                          python_op,
                                                          np.array])

    ds_random_lighting_py = data.map(
        operations=transforms_op, input_columns="image")

    ds_random_lighting_py = ds_random_lighting_py.batch(512)

    for idx, (image, _) in enumerate(ds_random_lighting_py.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        if idx == 0:
            images_random_lighting_py = image

        else:
            images_random_lighting_py = np.append(
                images_random_lighting_py, image, axis=0)

    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
    data = data.map(operations=[vision.Decode(), vision.Resize(
        (200, 300))], input_columns=["image"])

    ds_images_random_lighting_c = data.map(
        operations=c_op, input_columns="image")

    ds_random_lighting_c = ds_images_random_lighting_c.batch(512)

    for idx, (image, _) in enumerate(ds_random_lighting_c.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        if idx == 0:
            images_random_lighting_c = image
        else:
            images_random_lighting_c = np.append(
                images_random_lighting_c, image, axis=0)

    num_samples = images_random_lighting_c.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_random_lighting_c[i],
                          images_random_lighting_py[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))
    if plot:
        visualize_list(images_random_lighting_c,
                       images_random_lighting_py, visualize_mode=2)


def test_random_lighting_invalid_params():
    """
    Feature: RandomLighting
    Description: Test RandomLighting with invalid input parameters
    Expectation: Throw correct error and message
    """
    logger.info("Test RandomLighting with invalid input parameters.")
    with pytest.raises(ValueError) as error_info:
        data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data = data.map(operations=[vision.Decode(), vision.Resize((224, 224)),
                                    vision.RandomLighting(-2)], input_columns=["image"])
    assert "Input alpha is not within the required interval of [0, 16777216]." in str(
        error_info.value)

    with pytest.raises(TypeError) as error_info:
        data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data = data.map(operations=[vision.Decode(), vision.Resize((224, 224)),
                                    vision.RandomLighting('1')], input_columns=["image"])
    err_msg = "Argument alpha with value 1 is not of type [<class 'float'>, <class 'int'>], but got <class 'str'>."
    assert err_msg in str(error_info.value)


def test_random_lighting_operation_01():
    """
    Feature: RandomLighting operation
    Description: Testing the normal functionality of the RandomLighting operator
    Expectation: The Output is equal to the expected output
    """
    # Test RandomLighting pipeline func,input Numpy data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    random_lighting_op = vision.RandomLighting(10)
    dataset2 = dataset2.map(input_columns=["image"], operations=random_lighting_op)
    for _ in dataset2.create_dict_iterator(output_numpy=False):
        pass

    # Test RandomLighting pipeline func,input PIL data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    op = [
        vision.Decode(to_pil=True),
        vision.RandomLighting(10)
    ]
    dataset2 = dataset2.map(input_columns=["image"], operations=op)

    for _ in dataset2.create_dict_iterator(output_numpy=False):
        pass

    # Test RandomLighting eager func,input Numpy data
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    image = np.fromfile(image_jpg, dtype=np.float32)
    random_lighting_op = vision.RandomLighting(1)
    image = vision.Decode()(image)
    out = random_lighting_op(image)
    assert (out == 255 - image).all

    # Test RandomLighting func,input PIL data
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        random_lighting_op = vision.RandomLighting(0.01)
        _ = random_lighting_op(image)

    # Test RandomLighting func, input .bmp img data
    image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
    with Image.open(image_bmp) as image:
        random_lighting_op = vision.RandomLighting(0.68)
        _ = random_lighting_op(image)

    # Test RandomLighting func, input three channel img
    image = np.random.randint(0, 255, (20, 48, 3)).astype(np.uint8)
    random_lighting_op = vision.RandomLighting()
    _ = random_lighting_op(image)


def test_random_lighting_exception_01():
    """
    Feature: RandomLighting operation
    Description: Testing the RandomLighting Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Test RandomLighting func input list data（tolist）
    image = np.random.randint(0, 255, (10, 10, 3)).astype(np.uint8).tolist()
    random_lighting_op = vision.RandomLighting(1)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        random_lighting_op(image)

    # Test RandomLighting func input shape 0 data
    image = np.array(0).astype(np.uint8)
    random_lighting_op = vision.RandomLighting(1)
    with pytest.raises(RuntimeError, match="the shape of input tensor does not match the requirement of operator. "
                                           "Expecting tensor in shape of <height, width, ...>. "
                                           "But got tensor with dimension 0."):
        random_lighting_op(image)

    # Test RandomLighting func,input four channel img
    image = np.random.randint(0, 255, (10, 10, 4)).astype(np.uint8)
    random_lighting_op = vision.RandomLighting(1)
    with pytest.raises(RuntimeError, match="RandomLighting: input tensor is not in shape of "
                                           "\\<H,W,C\\> or channel is not 3, got rank: 3, and shape: <10,10,4>"):
        random_lighting_op(image)

    # Test RandomLighting func input 2-D data
    image = np.random.randint(0, 255, (20, 48)).astype(np.uint8)
    random_lighting_op = vision.RandomLighting(1)
    with pytest.raises(RuntimeError, match="RandomLighting: input tensor is not in shape of <H,W,C> "
                                           "or channel is not 3, got rank: 2"):
        random_lighting_op(image)

    # Test RandomLighting func input 4-D data
    image = np.random.randint(0, 255, (10, 10, 3, 3)).astype(np.uint8)
    random_lighting_op = vision.RandomLighting(1)
    with pytest.raises(RuntimeError, match="RandomLighting: input tensor is not in shape of "
                                           "<H,W,C> or channel is not 3, got rank: 4, and shape: <10,10,3,3>"):
        random_lighting_op(image)

    # Test RandomLighting array type is str
    image = np.array([[["a", "b", "c"], ["a", "b", "c"]]])
    random_lighting_op = vision.RandomLighting(1)
    with pytest.raises(RuntimeError, match="RandomLighting: the data type of input tensor does not match "
                                           "the requirement of operator. Expecting tensor in type of "
                                           "\\[int, float, double\\]. But got type string."):
        random_lighting_op(image)

    # Test RandomLighting array type is int64
    image = np.random.randint(0, 255, (10, 10, 3)).astype(np.int64)
    random_lighting_op = vision.RandomLighting(1)
    with pytest.raises(RuntimeError, match="RandomLighting: Cannot convert from OpenCV type, unknown "
                                           "CV type. Currently supported data type: \\[int8, uint8, int16, uint16, "
                                           "int32, float16, float32, float64\\]."):
        random_lighting_op(image)

    # Test RandomLighting func alpha is list
    with pytest.raises(TypeError, match="Argument alpha with value \\[1\\] is not of type \\[<class 'float'>, "
                                        "<class 'int'>\\], but got <class 'list'>."):
        vision.RandomLighting([1])

    # Test RandomLighting func alpha is -0.5
    with pytest.raises(ValueError, match="Input alpha is not within the required interval of \\[0, 16777216\\]"):
        vision.RandomLighting(-0.5)

    # Test RandomLighting func alpha is bool
    with pytest.raises(TypeError, match="Argument alpha with value True is not of type \\(<class 'float'>, "
                                        "<class 'int'>\\), but got <class 'bool'>."):
        vision.RandomLighting(True)

    # Test RandomLighting func alpha is str
    with pytest.raises(TypeError, match="Argument alpha with value 1 is not of type \\[<class 'float'>, "
                                        "<class 'int'>\\], but got <class 'str'>."):
        vision.RandomLighting("1")

    # Test RandomLighting func input two arguments
    with pytest.raises(TypeError, match="too many positional arguments"):
        vision.RandomLighting(1, 2)


if __name__ == "__main__":
    test_random_lighting_py()
    test_random_lighting_py_md5()
    test_random_lighting_c()
    test_random_lighting_c_py()
    test_random_lighting_invalid_params()
    test_random_lighting_operation_01()
    test_random_lighting_exception_01()
