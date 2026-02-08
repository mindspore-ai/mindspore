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
Testing RandomPerspective op in DE
"""
import numpy as np
import os
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.transforms
import mindspore.dataset.transforms.transforms as t_trans
import mindspore.dataset.vision.transforms as vision
from mindspore.dataset.vision.utils import Inter
from mindspore import log as logger
from util import visualize_list, save_and_check_md5_pil, \
    config_get_set_seed, config_get_set_num_parallel_workers

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
TEST_DATA_DATASET_FUNC ="../data/dataset/"


def test_random_perspective_op(plot=False):
    """
    Feature: RandomPerspective op
    Description: Test RandomPerspective in Python transformations
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_perspective_op")
    # define map operations
    transforms1 = [
        vision.Decode(True),
        vision.RandomPerspective(),
        vision.ToTensor()
    ]
    transform1 = mindspore.dataset.transforms.Compose(transforms1)

    transforms2 = [
        vision.Decode(True),
        vision.ToTensor()
    ]
    transform2 = mindspore.dataset.transforms.Compose(transforms2)

    #  First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=transform1, input_columns=["image"])
    #  Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=transform2, input_columns=["image"])

    image_perspective = []
    image_original = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image2 = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_perspective.append(image1)
        image_original.append(image2)
    if plot:
        visualize_list(image_original, image_perspective)


def skip_test_random_perspective_md5():
    """
    Feature: RandomPerspective op
    Description: Test RandomPerspective with md5 comparison
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_perspective_md5")
    original_seed = config_get_set_seed(5)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # define map operations
    transforms = [
        vision.Decode(True),
        vision.RandomPerspective(distortion_scale=0.3, prob=0.7,
                                 interpolation=Inter.BILINEAR),
        vision.Resize(1450),  # resize to a smaller size to prevent round-off error
        vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.Compose(transforms)

    #  Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data = data.map(operations=transform, input_columns=["image"])

    # check results with md5 comparison
    filename = "random_perspective_01_result.npz"
    save_and_check_md5_pil(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers((original_num_parallel_workers))


def test_random_perspective_exception_distortion_scale_range():
    """
    Feature: RandomPerspective op
    Description: Test RandomPerspective where distortion_scale is not in [0, 1]
    Expectation: Error is raised as expected
    """
    logger.info("test_random_perspective_exception_distortion_scale_range")
    try:
        _ = vision.RandomPerspective(distortion_scale=1.5)
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input distortion_scale is not within the required interval of [0.0, 1.0]."


def test_random_perspective_exception_prob_range():
    """
    Feature: RandomPerspective op
    Description: Test RandomPerspective where prob is not in [0, 1]
    Expectation: Error is raised as expected
    """
    logger.info("test_random_perspective_exception_prob_range")
    try:
        _ = vision.RandomPerspective(prob=1.2)
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input prob is not within the required interval of [0.0, 1.0]."


def test_random_perspective_operation_01():
    """
    Feature: RandomPerspective operation
    Description: Testing the normal functionality of the RandomPerspective operator
    Expectation: The Output is equal to the expected output
    """
    # Test RandomPerspective func with Inter mode isv_Inter.NEAREST
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, 1)
    transforms1 = [
        vision.Decode(to_pil=True),
        vision.RandomPerspective(0.5, 0.5, Inter.BICUBIC),
        vision.ToTensor()
    ]
    transform1 = t_trans.Compose(transforms1)
    dataset = dataset.map(input_columns=["image"], operations=transform1)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomPerspective func interpolation is Inter.BICUBIC
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        distortion_scale = 0.2
        prob = 0.6
        interpolation = Inter.BICUBIC
        random_perspective_op = vision.RandomPerspective(distortion_scale=distortion_scale, prob=prob,
                                                         interpolation=interpolation)
        _ = random_perspective_op(image)

    # Test RandomPerspective func interpolation is Inter.BILINEAR
    image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    with Image.open(image_png) as image:
        distortion_scale = 0.01
        prob = 0.8
        interpolation = Inter.BILINEAR
        random_perspective_op = vision.RandomPerspective(distortion_scale=distortion_scale, prob=prob,
                                                         interpolation=interpolation)
        _ = random_perspective_op(image)

    # Test RandomPerspective func interpolation is Inter.NEAREST
    image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")
    with Image.open(image_gif) as image:
        distortion_scale = 0.5
        prob = 0.999
        interpolation = Inter.NEAREST
        random_perspective_op = vision.RandomPerspective(distortion_scale=distortion_scale, prob=prob,
                                                         interpolation=interpolation)
        _ = random_perspective_op(image)

    # Test RandomPerspective func default value
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        random_perspective_op = vision.RandomPerspective()
        _ = random_perspective_op(image)


def test_random_perspective_exception_01():
    """
    Feature: RandomPerspective operation
    Description: Testing the RandomPerspective Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Test RandomPerspective func with Numpy data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, 1)
    transforms1 = [
        vision.Decode(to_pil=False),
        vision.RandomPerspective(),
        vision.ToTensor()
    ]
    transform1 = t_trans.Compose(transforms1)
    dataset = dataset.map(input_columns=["image"], operations=transform1)
    with pytest.raises(RuntimeError, match="Input image should be a Pillow image"):
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # Test RandomPerspective func input .bmp img
    image = np.random.randn(3, 4, 3)
    distortion_scale = 0.9999
    prob = 0.4
    interpolation = Inter.BILINEAR
    with pytest.raises(ValueError, match="Input image should be a Pillow image."):
        random_perspective_op = vision.RandomPerspective(distortion_scale=distortion_scale, prob=prob,
                                                         interpolation=interpolation)
        random_perspective_op(image)

    # Test RandomPerspective func normal: (0.5, 1.0)
    distortion_scale = (0.5, 1.0)
    with pytest.raises(TypeError, match=r'is not of type \[\<class \'float\'\>\].'):
        vision.RandomPerspective(distortion_scale)

    # Test RandomPerspective func normal: [0.5, 0.9]
    distortion_scale = [0.5, 0.9]
    with pytest.raises(TypeError, match=r'is not of type \[\<class \'float\'\>\].'):
        vision.RandomPerspective(distortion_scale)

    # Test RandomPerspective func normal:-0.5, ValueError
    distortion_scale = -0.5
    with pytest.raises(ValueError, match=r'Input distortion_scale is not within the required interval of \[0.0, 1.0\]'):
        vision.RandomPerspective(distortion_scale)

    # Test RandomPerspective func normal:1.3 ValueError
    distortion_scale = 1.3
    with pytest.raises(ValueError, match=r'Input distortion_scale is not within the required interval of \[0.0, 1.0\]'):
        vision.RandomPerspective(distortion_scale)

    # Test RandomPerspective func normal:(0.5, 1)
    prob = (0.5, 1)
    with pytest.raises(TypeError, match=r'is not of type \[\<class \'float\'\>\]'):
        vision.RandomPerspective(prob)

    # Test RandomPerspective func normal:[0.5, 0.9]
    prob = [0.5, 0.9]
    with pytest.raises(TypeError, match=r'is not of type \[\<class \'float\'\>\]'):
        vision.RandomPerspective(prob)

    # Test RandomPerspective func normal:-0.5 ValueError
    prob = -0.5
    with pytest.raises(ValueError, match=r'Input prob is not within the required interval of \[0.0, 1.0\]'):
        vision.RandomPerspective(prob=prob)

    # Test RandomPerspective func normal:1.3 ValueError
    prob = 1.3
    with pytest.raises(ValueError, match=r'Input prob is not within the required interval of \[0.0, 1.0\]'):
        vision.RandomPerspective(prob=prob)

    # Test RandomPerspective func normal:interpolation is 2
    interpolation = 2
    distortion_scale = 0.1
    prob = 0.1
    with pytest.raises(TypeError, match="Argument interpolation with value 2 is not of type \\[<enum 'Inter'>\\]."):
        vision.RandomPerspective(distortion_scale, prob, interpolation)


if __name__ == "__main__":
    test_random_perspective_op(plot=True)
    skip_test_random_perspective_md5()
    test_random_perspective_exception_distortion_scale_range()
    test_random_perspective_exception_prob_range()
    test_random_perspective_operation_01()
    test_random_perspective_exception_01()
