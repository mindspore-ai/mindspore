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
Testing RandomSolarizeOp op in DE
"""
import numpy as np
import os
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger
from util import visualize_list, save_and_check_md5, config_get_set_seed, config_get_set_num_parallel_workers, \
    visualize_one_channel_dataset

GENERATE_GOLDEN = False

MNIST_DATA_DIR = "../data/dataset/testMnistData"
DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
TEST_DATA_DATASET_FUNC ="../data/dataset/"
DATA_DIR_1 = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")


def test_random_solarize_op(threshold=(10, 150), plot=False, run_golden=True):
    """
    Feature: RandomSolarize op
    Description: Test RandomSolarize op on TFRecordDataset
    Expectation: The dataset is processed as expected
    """
    logger.info("Test RandomSolarize")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()

    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    if threshold is None:
        solarize_op = vision.RandomSolarize()
    else:
        solarize_op = vision.RandomSolarize(threshold)

    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=solarize_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=decode_op, input_columns=["image"])

    if run_golden:
        filename = "random_solarize_01_result.npz"
        save_and_check_md5(data1, filename, generate_golden=GENERATE_GOLDEN)

    image_solarized = []
    image = []

    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_solarized.append(item1["image"].copy())
        image.append(item2["image"].copy())
    if plot:
        visualize_list(image, image_solarized)

    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_solarize_mnist(plot=False, run_golden=True):
    """
    Feature: RandomSolarize op
    Description: Test RandomSolarize op on MnistDataset (grayscale images)
    Expectation: The dataset is processed as expected
    """
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    mnist_1 = ds.MnistDataset(dataset_dir=MNIST_DATA_DIR, num_samples=2, shuffle=False)
    mnist_2 = ds.MnistDataset(dataset_dir=MNIST_DATA_DIR, num_samples=2, shuffle=False)
    mnist_2 = mnist_2.map(operations=vision.RandomSolarize((0, 255)), input_columns="image")

    images = []
    images_trans = []
    labels = []

    for _, (data_orig, data_trans) in enumerate(zip(mnist_1, mnist_2)):
        image_orig, label_orig = data_orig
        image_trans, _ = data_trans
        images.append(image_orig.asnumpy())
        labels.append(label_orig.asnumpy())
        images_trans.append(image_trans.asnumpy())

    if plot:
        visualize_one_channel_dataset(images, images_trans, labels)

    if run_golden:
        filename = "random_solarize_02_result.npz"
        save_and_check_md5(mnist_2, filename, generate_golden=GENERATE_GOLDEN)

    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_solarize_errors():
    """
    Feature: RandomSolarize op
    Description: Test RandomSolarize with bad inputs
    Expectation: Correct error is thrown as expected
    """
    with pytest.raises(ValueError) as error_info:
        vision.RandomSolarize((12, 1))
    assert "threshold must be in min max format numbers" in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        vision.RandomSolarize((12, 1000))
    assert "Input is not within the required interval of [0, 255]." in str(error_info.value)

    with pytest.raises(TypeError) as error_info:
        vision.RandomSolarize((122.1, 140))
    assert "Argument threshold[0] with value 122.1 is not of type [<class 'int'>]" in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        vision.RandomSolarize((122, 100, 30))
    assert "threshold must be a sequence of two numbers" in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        vision.RandomSolarize((120,))
    assert "threshold must be a sequence of two numbers" in str(error_info.value)


def test_random_solarize_operation_01():
    """
    Feature: RandomSolarize operation
    Description: Testing the normal functionality of the RandomSolarize operator
    Expectation: The Output is equal to the expected output
    """
    # Test randomsolarize no para
    dataset2 = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    random_solarize_op = vision.RandomSolarize()
    dataset2 = dataset2.map(input_columns=["image"], operations=random_solarize_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # Test randomsolarize threshold = (0, 150)
    threshold = (0, 150)
    image = Image.open(image_jpg)
    random_solarize_op = vision.RandomSolarize(threshold=threshold)
    _ = random_solarize_op(image)

    # Test randomsolarize threshold = (0, 255)
    threshold = (0, 255)
    image = Image.open(image_bmp)
    random_solarize_op = vision.RandomSolarize(threshold=threshold)
    _ = random_solarize_op(image)

    # Test randomsolarize image = gif
    image = Image.open(image_gif)
    random_solarize_op = vision.RandomSolarize()
    _ = random_solarize_op(image)

    # Test randomsolarize threshold = (200, 201)
    threshold = (200, 201)
    image = np.random.randint(0, 199, (256, 382, 3)).astype(np.uint8)
    random_solarize_op = vision.RandomSolarize(threshold=threshold)
    out = random_solarize_op(image)
    assert (np.array(image) == out).all()

    # Test randomsolarize threshold = (10, 82)
    threshold = (10, 82)
    image = np.random.randint(0, 255, (253, 300, 1)).astype(np.uint8)
    random_solarize_op = vision.RandomSolarize(threshold=threshold)
    random_solarize_op(image)

    # Test randomsolarize threshold = (0, 0)
    threshold = (0, 0)
    image = np.random.randint(0, 255, (500, 500)).astype(np.uint8)
    random_solarize_op = vision.RandomSolarize(threshold=threshold)
    _ = random_solarize_op(image)

    # Test randomsolarize threshold = (0, 0) and (0, 255)
    threshold = (0, 0)
    threshold2 = (0, 255)
    image = np.random.randint(0, 255, (256, 256)).astype(np.uint8)
    random_solarize_op = vision.RandomSolarize(threshold=threshold)
    random_solarize_op2 = vision.RandomSolarize(threshold=threshold2)
    out = random_solarize_op(image)
    out2 = random_solarize_op2(image)
    assert (out != out2).any()


def test_random_solarize_exception_01():
    """
    Feature: RandomSolarize operation
    Description: Testing the RandomSolarize Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Test randomsolarize threshold = [0, 255]
    threshold = [0, 255]
    with pytest.raises(TypeError,
                       match="Argument threshold with value \\[0, 255\\] is not of type \\[<class 'tuple'>\\]."):
        vision.RandomSolarize(threshold=threshold)

    # Test randomsolarize threshold = (-1, 255)
    threshold = (-1, 255)
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        vision.RandomSolarize(threshold=threshold)

    # Test randomsolarize threshold = (0, 256)
    threshold = (0, 256)
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[0, 255\\]."):
        vision.RandomSolarize(threshold=threshold)

    # Test randomsolarize threshold = (128, 1)
    threshold = (128, 1)
    with pytest.raises(ValueError, match="threshold must be in min max format numbers"):
        vision.RandomSolarize(threshold=threshold)

    # Test randomsolarize threshold = (0, 128.5)
    threshold = (0, 128.5)
    with pytest.raises(TypeError,
                       match="Argument threshold\\[1\\] with value 128.5 is not of type \\[<class 'int'>\\]."):
        vision.RandomSolarize(threshold=threshold)

    # Test randomsolarize threshold = ()
    threshold = ()
    with pytest.raises(ValueError, match="threshold must be a sequence of two numbers"):
        vision.RandomSolarize(threshold=threshold)

    # Test randomsolarize threshold = (0,1,128)
    threshold = (0, 1, 128)
    with pytest.raises(ValueError, match="threshold must be a sequence of two numbers"):
        vision.RandomSolarize(threshold=threshold)

    # Test randomsolarize threshold = 0
    threshold = 0
    with pytest.raises(TypeError, match="Argument threshold with value 0 is not of type \\[<class 'tuple'>\\]."):
        vision.RandomSolarize(threshold=threshold)

    # Test randomsolarize threshold =
    threshold = ""
    with pytest.raises(TypeError, match="Argument threshold"):
        vision.RandomSolarize(threshold=threshold)

    # Test randomsolarize image.shape = (500, 500, 4)
    threshold = (125, 255)
    image = np.random.randint(0, 255, (500, 500, 4)).astype(np.uint8)
    random_solarize_op = vision.RandomSolarize(threshold=threshold)
    with pytest.raises(RuntimeError, match=r"Solarize: the channel of image tensor does not "
                                           r"match the requirement of operator. Expecting tensor in channel of "
                                           r"\(1, 3\). But got channel 4."):
        random_solarize_op(image)

    # Test randomsolarize image = 1d
    threshold = (0, 125)
    image = np.fromfile(image_jpg, dtype=np.uint8)
    random_solarize_op = vision.RandomSolarize(threshold=threshold)
    with pytest.raises(RuntimeError, match=r"Solarize: the dimension of image tensor does "
                                           r"not match the requirement of operator. Expecting tensor in "
                                           r"dimension of \(2, 3\), in shape of <H, W> or <H, W, C>. But got "
                                           r"dimension 1. You may need to perform Decode first."):
        random_solarize_op(image)

    # Test randomsolarize image = list
    threshold = (0, 125)
    image = np.random.randint(0, 255, (500, 500, 3)).astype(np.uint8).tolist()
    random_solarize_op = vision.RandomSolarize(threshold=threshold)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        random_solarize_op(image)

    # Test randomsolarize image = tuple
    threshold = (0, 125)
    image = tuple(np.random.randint(0, 255, (500, 500, 3)).astype(np.uint8).tolist())
    random_solarize_op = vision.RandomSolarize(threshold=threshold)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'tuple'>."):
        random_solarize_op(image)

    # Test randomsolarize image.shape = (500, 500, 3, 3)
    threshold = (0, 125)
    image = np.random.randint(0, 255, (500, 500, 3, 3)).astype(np.uint8)
    random_solarize_op = vision.RandomSolarize(threshold=threshold)
    with pytest.raises(RuntimeError, match=r"Solarize: the dimension of image tensor does "
                                           r"not match the requirement of operator. Expecting tensor in "
                                           r"dimension of \(2, 3\), in shape of <H, W> or <H, W, C>. "
                                           r"But got dimension 4."):
        random_solarize_op(image)

    # Test randomsolarize threshold = (10,)
    threshold = (10,)
    with pytest.raises(ValueError, match="threshold must be a sequence of two numbers."):
        vision.RandomSolarize(threshold=threshold)

    # Test randomsolarize threshold = {10, 50}
    threshold = {10, 50}
    with pytest.raises(TypeError,
                       match="Argument threshold with value {10, 50} is not of type \\[<class 'tuple'>\\]."):
        vision.RandomSolarize(threshold=threshold)

    # Test randomsolarize threshold = np.array([10, 50])
    threshold = np.array([10, 50])
    with pytest.raises(TypeError,
                       match="Argument threshold with value \\[10 50\\] is not of type \\[<class 'tuple'>\\]."):
        vision.RandomSolarize(threshold=threshold)

    # Test randomsolarize threshold = (True, 20)
    threshold = (True, 20)
    with pytest.raises(TypeError,
                       match="Argument threshold\\[0\\] with value True is not of type \\(<class 'int'>,\\)."):
        vision.RandomSolarize(threshold=threshold)


if __name__ == "__main__":
    test_random_solarize_op((10, 150), plot=True, run_golden=True)
    test_random_solarize_op((12, 120), plot=True, run_golden=False)
    test_random_solarize_op(plot=True, run_golden=False)
    test_random_solarize_mnist(plot=True, run_golden=True)
    test_random_solarize_errors()
    test_random_solarize_operation_01()
    test_random_solarize_exception_01()
