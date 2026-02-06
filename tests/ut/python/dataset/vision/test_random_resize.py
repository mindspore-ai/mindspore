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
Testing RandomResize op in DE
"""
import numpy as np
import os
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.transforms as ops
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger
from util import visualize_list, save_and_check_md5, diff_mse, \
    config_get_set_seed, config_get_set_num_parallel_workers

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
TEST_DATA_DATASET_FUNC ="../data/dataset/"

GENERATE_GOLDEN = False


def generator_mc(maxid=3):
    """ return five image """
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
    image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")
    with Image.open(image_jpg) as image:
        image = np.array(image)
    with Image.open(image_bmp) as image1:
        image1 = np.array(image1)
    with Image.open(image_gif) as image2:
        image2 = np.array(image2)
    for _ in range(maxid):
        yield image, image, image1, image1, image2


def test_random_resize_op(plot=False):
    """
    Feature: RandomResize op
    Description: Test RandomResize op basic usage
    Expectation: The dataset is processed as expected
    """
    logger.info("Test resize")
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

    # define map operations
    decode_op = vision.Decode()
    resize_op = vision.RandomResize(10)

    # apply map operations on images
    data1 = data1.map(operations=decode_op, input_columns=["image"])

    data2 = data1.map(operations=resize_op, input_columns=["image"])
    image_original = []
    image_resized = []
    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_1 = item1["image"]
        image_2 = item2["image"]
        image_original.append(image_1)
        image_resized.append(image_2)
        num_iter += 1
    if plot:
        visualize_list(image_original, image_resized)


def test_random_resize_md5():
    """
    Feature: RandomResize op
    Description: Test RandomResize op with md5 check
    Expectation: The dataset is processed as expected
    """
    logger.info("Test RandomResize with md5 check")
    original_seed = config_get_set_seed(5)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    resize_op = vision.RandomResize(10)
    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=resize_op, input_columns=["image"])
    # Compare with expected md5 from images
    filename = "random_resize_01_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)

def test_random_resize_op_1():
    """
    Feature: RandomResize op
    Description: Test RandomResize op with different fields
    Expectation: The dataset is processed as expected
    """
    logger.info("Test RandomResize with different fields.")

    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data = data.map(operations=ops.Duplicate(), input_columns=["image"],
                    output_columns=["image", "image_copy"])
    resize_op = vision.RandomResize(10)
    decode_op = vision.Decode()

    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=decode_op, input_columns=["image_copy"])
    data = data.map(operations=resize_op, input_columns=["image", "image_copy"])

    num_iter = 0
    for data1 in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = data1["image"]
        image_copy = data1["image_copy"]
        mse = diff_mse(image, image_copy)
        logger.info("image_{}, mse: {}".format(num_iter + 1, mse))
        assert mse == 0
        num_iter += 1


def test_random_resize_operation_01():
    """
    Feature: RandomResize operation
    Description: Testing the normal functionality of the RandomResize operator
    Expectation: The Output is equal to the expected output
    """
    # Test RandomResize func size is 1
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = 1
    random_resize_op = vision.RandomResize(size=size)
    dataset2 = dataset2.map(input_columns=["image"], operations=random_resize_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomResize func size is [500, 520]
    source = generator_mc
    column_names = ["image1", "image2", "image3", "image4", "image5"]
    dataset = ds.GeneratorDataset(source, column_names)
    size = [500, 520]
    random_resize_op = vision.RandomResize(size=size)
    dataset = dataset.map(input_columns=["image1", "image2", "image3", "image4", "image5"], operations=random_resize_op)
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["image1"] == data["image2"]).all()
        assert (data["image3"] == data["image4"]).all()

    # Test RandomResize func all para
    source = generator_mc
    column_names = ["image1", "image2", "image3", "image4", "image5"]
    dataset = ds.GeneratorDataset(source, column_names)
    size = (500, 520)
    random_resize_op = vision.RandomResize(size=size)
    dataset = dataset.map(input_columns=["image1", "image2", "image5"], operations=random_resize_op)
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["image1"] == data["image2"]).all()

    # Test RandomResize func size is (800, 1000)
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        size = (800, 1000)
        random_resize_op = vision.RandomResize(size)
        out = random_resize_op(image)
        assert out.shape[0] == 800
        assert out.shape[1] == 1000

    # Test RandomResize func size is (100, 200)
    image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    image = Image.open(image_png)
    size = (100, 200)
    random_resize_op = vision.RandomResize(size)
    out = random_resize_op(image, image)
    assert np.array(out).shape[0] == 2
    assert np.array(out).shape[1] == 100
    assert np.array(out).shape[2] == 200
    assert (out[0] == out[1]).all()
    image.close()

    # Test RandomResize func size is (200, 355)
    image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
    image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")
    image = Image.open(image_gif)
    image1 = Image.open(image_bmp)
    image2 = np.random.randint(0, 255, (256, 382, 3)).astype(np.uint8)
    size = (200, 355)
    random_resize_op = vision.RandomResize(size)
    out = random_resize_op(image, image1, image2)
    assert np.array(out[0]).shape[0] == 200
    assert np.array(out[0]).shape[1] == 355
    image.close()
    image1.close()

    # Test RandomResize func size is 400
    image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
    image = Image.open(image_bmp)
    size = 400
    random_resize_op = vision.RandomResize(size)
    out = random_resize_op(image)
    assert out.shape[0] == 400
    assert out.shape[1] == out.shape[1] / out.shape[0] * 400
    image.close()

    # Test RandomResize func size is [400, 400]
    image = np.random.randint(0, 255, (300, 500, 1)).astype(np.uint8)
    size = [400, 400]
    random_resize_op = vision.RandomResize(size)
    out = random_resize_op(image)
    assert out.shape[0] == 400
    assert out.shape[1] == 400


def test_random_resize_operation_02():
    """
    Feature: RandomResize operation
    Description: Testing the normal functionality of the RandomResize operator
    Expectation: The Output is equal to the expected output
    """
    # Test size is (600, 400)
    image = np.random.randn(300, 500)
    size = (600, 400)
    random_resize_op = vision.RandomResize(size)
    out = random_resize_op(image)
    assert out.shape[0] == 600
    assert out.shape[1] == 400

    # Test size is 1
    image = np.random.randn(300, 600, 3)
    size = 1
    random_resize_op = vision.RandomResize(size)
    out = random_resize_op(image)
    assert out.shape[0] == 1
    assert out.shape[1] == 2

    # Test size is 3000
    image = np.random.randn(3, 6, 3)
    size = 3000
    random_resize_op = vision.RandomResize(size)
    out = random_resize_op(image)
    assert out.shape[0] == 3000
    assert out.shape[1] == 6000

    # 输入 4d numpy array，维度已扩展
    image = np.random.randn(300, 300, 3, 3)
    size = (500, 500)
    random_resize_op = vision.RandomResize(size)
    output = random_resize_op(image)
    assert output.shape == (300, 500, 500, 3)

    # Test input PIL data,size is (600, 400)
    image = np.random.randn(300, 500)
    image = vision.ToPIL()(image)
    size = (600, 400)
    random_resize_op = vision.RandomResize(size)
    out = random_resize_op(image)
    assert out.shape[0] == 600
    assert out.shape[1] == 400


def test_random_resize_exception_01():
    """
    Feature: RandomResize operation
    Description: Testing the RandomResize Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Test RandomResize func size is 0
    size = 0
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        vision.RandomResize(size=size)

    # Test RandomResize func size is 16777217
    size = 16777217
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        vision.RandomResize(size=size)

    # Test RandomResize func size is 500.5
    size = 500.5
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple"):
        vision.RandomResize(size=size)

    # Test RandomResize func size is (500, 500, 520)
    size = (500, 500, 520)
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple"):
        vision.RandomResize(size=size)

    # Test RandomResize func size is ""
    size = ""
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple"):
        vision.RandomResize(size=size)

    # Test RandomResize func no para
    with pytest.raises(TypeError, match="missing a required argument"):
        vision.RandomResize()

    # Test RandomResize func more para
    size = (500, 520)
    more_para = None
    with pytest.raises(TypeError, match="too many positional arguments"):
        vision.RandomResize(size, more_para)

    # Test RandomResize func size is 16777216
    image = np.random.randint(0, 255, (256, 382, 3)).astype(np.uint8)
    size = 16777216
    random_resize_op = vision.RandomResize(size)
    with pytest.raises(RuntimeError, match="Resize: the resizing width or height is too big, "
                                           "it's 1000 times bigger than the original image"):
        random_resize_op(image)

    # Test input is 1d
    image = np.random.randn(300,)
    size = 500
    random_resize_op = vision.RandomResize(size)
    with pytest.raises(RuntimeError,
                       match="Resize: the image tensor should have at least two dimensions. "
                             "You may need to perform Decode first."):
        random_resize_op(image)

    # Test input is list
    image = np.random.randn(300, 300, 3).tolist()
    size = 500
    random_resize_op = vision.RandomResize(size)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        random_resize_op(image)

    # Test size is (500, 2147483648)
    size = (500, 2147483648)
    with pytest.raises(ValueError,
                       match="Input size at dim 1 is not within the required interval of \\[1, 2147483647\\]."):
        vision.RandomResize(size)

    # Test size is {500, 1670}
    size = {500, 1670}
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple \\(h, w\\) of length 2."):
        vision.RandomResize(size)

    # Test size is [500]
    size = [500]
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple \\(h, w\\) of length 2."):
        vision.RandomResize(size)

    # Test size is np
    size = np.array([200, 200])
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple \\(h, w\\) of length 2."):
        vision.RandomResize(size)

    # Test input is 1d
    image = np.random.randint(0, 255, (52, 52, 3)).astype(np.uint8)
    random_resize_op = vision.RandomResize(30)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        random_resize_op(list(image))

    # Test input is 1d
    image = np.random.randint(0, 255, (50, 50, 3)).astype(np.uint8)
    random_resize_op = vision.RandomResize(30)
    with pytest.raises(RuntimeError):
        random_resize_op(np.array(10), image)


if __name__ == "__main__":
    test_random_resize_op(plot=True)
    test_random_resize_md5()
    test_random_resize_op_1()
    test_random_resize_operation_01()
    test_random_resize_operation_02()
    test_random_resize_exception_01()
