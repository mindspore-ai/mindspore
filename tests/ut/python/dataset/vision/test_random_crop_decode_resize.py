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
Testing RandomCropDecodeResize op in DE
"""
import numpy as np
import os
import pytest
from PIL import Image

import mindspore as ms
from mindspore import log as logger
import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as vision
from mindspore.dataset.vision import Inter as v_Inter
from util import diff_mse, visualize_image, save_and_check_md5, config_get_set_seed, config_get_set_num_parallel_workers

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
TEST_DATA_DATASET_FUNC ="../data/dataset/"

DATA_DIR_1 = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")

GENERATE_GOLDEN = False


def test_random_crop_decode_resize_op(plot=False):
    """
    Feature: RandomCropDecodeResize op
    Description: Test RandomCropDecodeResize op basic usgae
    Expectation: Passes the mse equality check
    """
    logger.info("test_random_decode_resize_op")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    random_crop_decode_resize_op = vision.RandomCropDecodeResize((256, 512), (1, 1), (0.5, 0.5))
    data1 = data1.map(operations=random_crop_decode_resize_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    random_crop_resize_op = vision.RandomResizedCrop((256, 512), (1, 1), (0.5, 0.5))
    data2 = data2.map(operations=decode_op, input_columns=["image"])
    data2 = data2.map(operations=random_crop_resize_op, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        if num_iter > 0:
            break
        image1 = item1["image"]
        image2 = item2["image"]
        mse = diff_mse(image1, image2)
        assert mse == 0
        logger.info("random_crop_decode_resize_op_{}, mse: {}".format(num_iter + 1, mse))
        if plot:
            visualize_image(image1, image2, mse)
        num_iter += 1


def test_random_crop_decode_resize_md5():
    """
    Feature: RandomCropDecodeResize op
    Description: Test RandomCropDecodeResize op with md5 check
    Expectation: Passes the md5 check test
    """
    logger.info("Test RandomCropDecodeResize with md5 check")
    original_seed = config_get_set_seed(10)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    random_crop_decode_resize_op = vision.RandomCropDecodeResize((256, 512), (1, 1), (0.5, 0.5))
    data = data.map(operations=random_crop_decode_resize_op, input_columns=["image"])
    # Compare with expected md5 from images
    filename = "random_crop_decode_resize_01_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers((original_num_parallel_workers))


def test_random_crop_decode_resize_invalid():
    """
    Feature: RandomCropDecodeResize
    Description: Test RandomCropDecodeResize with invalid input
    Expectation: Error is raised as expected
    """
    with pytest.raises(ValueError) as error_info:
        vision.RandomCropDecodeResize((256, 512), (1, 1), (0.5, -1))
    assert "not within the required interval of (0, 16777216]" in str(error_info.value)

    with pytest.raises(TypeError) as error_info:
        vision.RandomCropDecodeResize((256, 512), (1, 1), (0.5, 0.5), max_attempts=True)
    assert "not of type (<class 'int'>,)" in str(error_info.value)


def test_random_crop_decode_resize_operation_01():
    """
    Feature: RandomCropDecodeResize operation
    Description: Testing the normal functionality of the RandomCropDecodeResize operator
    Expectation: The Output is equal to the expected output
    """
    # When parameter size is 1, RandomCropDecodeResize interface is called successfully
    dataset = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False)
    size = 1
    random_crop_decode_resize_op = vision.RandomCropDecodeResize(size=size)
    dataset = dataset.map(input_columns=["image"], operations=random_crop_decode_resize_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When parameter size is a list, RandomCropDecodeResize interface is called successfully
    dataset = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False)
    size = [500, 520]
    random_crop_decode_resize_op = vision.RandomCropDecodeResize(size=size)
    dataset = dataset.map(input_columns=["image"], operations=random_crop_decode_resize_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When parameter max_attempts is 16777216, RandomCropDecodeResize interface is called successfully
    dataset = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False)
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = v_Inter.BILINEAR
    max_attempts = 16777216
    random_crop_decode_resize_op = vision.RandomCropDecodeResize(size=size, scale=scale, ratio=ratio,
                                                              interpolation=interpolation, max_attempts=max_attempts)
    dataset = dataset.map(input_columns=["image"], operations=random_crop_decode_resize_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When all parameters are set, RandomCropDecodeResize interface is called successfully
    dataset = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False)
    size = (500, 520)
    scale = [0.5, 1.0]
    ratio = (0.5, 1.0)
    interpolation = v_Inter.BILINEAR
    max_attempts = 1
    random_crop_decode_resize_op = vision.RandomCropDecodeResize(size=size, scale=scale, ratio=ratio,
                                                              interpolation=interpolation, max_attempts=max_attempts)
    dataset = dataset.map(input_columns=["image"], operations=random_crop_decode_resize_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When interpolation is set to AREA, RandomCropDecodeResize interface is called successfully
    dataset = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False)
    size = (500, 520)
    ratio = (0.5, 1.0)
    interpolation = v_Inter.AREA
    max_attempts = 8
    random_crop_decode_resize_op = vision.RandomCropDecodeResize(size=size, ratio=ratio, interpolation=interpolation,
                                                              max_attempts=max_attempts)
    dataset = dataset.map(input_columns=["image"], operations=random_crop_decode_resize_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When interpolation is set to PILCUBIC, RandomCropDecodeResize interface is called successfully
    dataset = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False)
    size = (500, 520)
    scale = [0.5, 1.0]
    interpolation = v_Inter.PILCUBIC
    max_attempts = 321
    random_crop_decode_resize_op = vision.RandomCropDecodeResize(size=size, scale=scale, interpolation=interpolation,
                                                              max_attempts=max_attempts)
    dataset = dataset.map(input_columns=["image"], operations=random_crop_decode_resize_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass


def test_random_crop_decode_resize_operation_02():
    """
    Feature: RandomCropDecodeResize operation
    Description: Testing the normal functionality of the RandomCropDecodeResize operator
    Expectation: The Output is equal to the expected output
    """
    # When interpolation is not set, RandomCropDecodeResize interface is called successfully
    dataset = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False)
    size = (500, 520)
    scale = [0.5, 1.0]
    ratio = [0.5, 1.0]
    max_attempts = 100
    random_crop_decode_resize_op = vision.RandomCropDecodeResize(size=size, scale=scale, ratio=ratio,
                                                              max_attempts=max_attempts)
    dataset = dataset.map(input_columns=["image"], operations=random_crop_decode_resize_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When interpolation is set to NEAREST, RandomCropDecodeResize interface is called successfully
    dataset = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False)
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = v_Inter.NEAREST
    random_crop_decode_resize_op = vision.RandomCropDecodeResize(size=size, scale=scale, ratio=ratio,
                                                              interpolation=interpolation)
    dataset = dataset.map(input_columns=["image"], operations=random_crop_decode_resize_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When input image format is jpg, RandomCropDecodeResize interface is called successfully
    image = np.fromfile(image_jpg, dtype=np.uint8)
    size = (500, 520)
    scale = (0, 10.0)
    ratio = (0.5, 0.5)
    interpolation = v_Inter.BILINEAR
    max_attempts = 1
    vc_randomcropdecoderesize = vision.RandomCropDecodeResize(size, scale, ratio, interpolation, max_attempts)
    _ = vc_randomcropdecoderesize(image)

    # When input image format is bmp, RandomCropDecodeResize interface is called successfully
    image = np.fromfile(image_bmp, dtype=np.uint8)
    size = 600
    scale = [0.5, 0.5]
    ratio = (0.1, 0.9)
    interpolation = v_Inter.NEAREST
    max_attempts = 10
    vc_randomcropdecoderesize = vision.RandomCropDecodeResize(size, scale, ratio, interpolation, max_attempts)
    _ = vc_randomcropdecoderesize(image)

    # When eager mode max_attempts is 16777216, RandomCropDecodeResize interface is called successfully
    image = np.fromfile(image_png, dtype=np.uint8)
    size = [180, 800]
    scale = (0, 167)
    ratio = [1.2, 10.0]
    interpolation = v_Inter.BICUBIC
    max_attempts = 16777216
    vc_randomcropdecoderesize = vision.RandomCropDecodeResize(size, scale, ratio, interpolation, max_attempts)
    _ = vc_randomcropdecoderesize(image)

    # When eager mode size is 6000, RandomCropDecodeResize interface is called successfully
    image = np.fromfile(image_jpg, dtype=np.uint8)
    size = 6000
    scale = [1, 1]
    ratio = [0.5, 1.5]
    interpolation = v_Inter.NEAREST
    max_attempts = 2
    vc_randomcropdecoderesize = vision.RandomCropDecodeResize(size, scale, ratio, interpolation, max_attempts)
    _ = vc_randomcropdecoderesize(image)

    # When parameter size is [200, 50], RandomCropDecodeResize interface is called successfully
    dataset = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=False)
    size = [200, 50]
    scale = (0.2, 2.0)
    ratio = [0.5, 1.0]
    interpolation = v_Inter.AREA
    vc_randomcropdecoderesize = vision.RandomCropDecodeResize(size=size, scale=scale, ratio=ratio,
                                                              interpolation=interpolation)
    dataset = dataset.map(input_columns=["image"], operations=vc_randomcropdecoderesize)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When parameter size is 300, RandomCropDecodeResize interface is called successfully
    dataset = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=False)
    size = 300
    scale = (1.2, 2.0)
    interpolation = v_Inter.PILCUBIC
    vc_randomcropdecoderesize = vision.RandomCropDecodeResize(size=size, scale=scale, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image"], operations=vc_randomcropdecoderesize)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass


def test_random_crop_decode_resize_exception_01():
    """
    Feature: RandomCropDecodeResize operation
    Description: Testing the RandomCropDecodeResize Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When parameter size is 0, RandomCropDecodeResize interface call fails
    size = 0
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        vision.RandomCropDecodeResize(size=size)

    # When parameter size is 16777217, RandomCropDecodeResize interface call fails
    size = 16777217
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        vision.RandomCropDecodeResize(size=size)

    # When parameter size is float, RandomCropDecodeResize interface call fails
    size = 500.5
    with pytest.raises(TypeError, match="Argument size with value 500.5 is not of type \\[<class 'int'>,"
                                        " <class 'list'>, <class 'tuple'>\\], but got <class 'float'>"):
        vision.RandomCropDecodeResize(size=size)

    # When parameter size is 3-tuple, RandomCropDecodeResize interface call fails
    size = (500, 500, 520)
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple"):
        vision.RandomCropDecodeResize(size=size)

    # When parameter size is empty, RandomCropDecodeResize interface call fails
    size = ""
    with pytest.raises(TypeError, match="Argument size with value .*, but got <class \'str\'>."):
        vision.RandomCropDecodeResize(size=size)

    # When parameter scale is negative, RandomCropDecodeResize interface call fails
    size = (500, 520)
    scale = (-0.5, 1.5)
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        vision.RandomCropDecodeResize(size=size, scale=scale)

    # When the first value of parameter scale is greater than the second value, RandomCropDecodeResize interface call fails
    size = (500, 520)
    scale = (1.5, 0.5)
    with pytest.raises(ValueError, match="scale should be in \\(min,max\\) format. Got \\(max,min\\)."):
        vision.RandomCropDecodeResize(size=size, scale=scale)

    # When parameter scale is empty, RandomCropDecodeResize interface call fails
    size = (500, 520)
    scale = ("", "")
    with pytest.raises(TypeError, match="Argument scale\\[0\\] with value"):
        vision.RandomCropDecodeResize(size=size, scale=scale)

    # When parameter scale is ms.Tensor, RandomCropDecodeResize interface call fails
    size = (500, 520)
    scale = ms.Tensor([0.5, 1.0])
    with pytest.raises(TypeError) as e:
        vision.RandomCropDecodeResize(size=size, scale=scale)
    assert "Argument scale with value {} is not of type [<class 'tuple'>, <class 'list'>]".format(scale) in str(e)

    # When parameter scale is string, RandomCropDecodeResize interface call fails
    size = (500, 520)
    scale = ""
    with pytest.raises(TypeError, match="Argument scale with value.*, but got <class \'str\'>"):
        vision.RandomCropDecodeResize(size=size, scale=scale)

    # When parameter ratio is negative, RandomCropDecodeResize interface call fails
    size = (500, 520)
    scale = (0.5, 1)
    ratio = (-0.5, 1.0)
    with pytest.raises(ValueError, match="Input ratio\\[0\\] is not within the required interval of"
                                         " \\(0, 16777216\\]."):
        vision.RandomCropDecodeResize(size=size, scale=scale, ratio=ratio)

    # When the second value of parameter ratio is less than the first value, RandomCropDecodeResize interface call fails
    size = (500, 520)
    scale = (0.5, 1)
    ratio = (1.5, 0.5)
    with pytest.raises(ValueError, match="ratio should be in \\(min,max\\) format. Got \\(max,min\\)."):
        vision.RandomCropDecodeResize(size=size, scale=scale, ratio=ratio)

    # When parameter ratio is empty, RandomCropDecodeResize interface call fails
    size = (500, 520)
    scale = (0.5, 1)
    ratio = ("", "")
    with pytest.raises(TypeError, match="Argument ratio\\[0\\] with value"):
        vision.RandomCropDecodeResize(size=size, scale=scale, ratio=ratio)

    # When parameter ratio is ms.Tensor, RandomCropDecodeResize interface call fails
    size = (500, 520)
    scale = (0.5, 1)
    ratio = ms.Tensor([0.5, 1.0])
    with pytest.raises(TypeError) as e :
        vision.RandomCropDecodeResize(size=size, scale=scale, ratio=ratio)
    assert "Argument ratio with value {} is not of type [<class 'tuple'>, <class 'list'>]".format(ratio) in str(e)

    # When parameter ratio is string, RandomCropDecodeResize interface call fails
    size = (500, 520)
    scale = (0.5, 1)
    ratio = ""
    with pytest.raises(TypeError, match="Argument ratio with value.*, but got <class \'str\'>"):
        vision.RandomCropDecodeResize(size=size, scale=scale, ratio=ratio)


def test_random_crop_decode_resize_exception_02():
    """
    Feature: RandomCropDecodeResize operation
    Description: Testing the RandomCropDecodeResize Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When parameter interpolation is string, RandomCropDecodeResize interface call fails
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = ""
    with pytest.raises(TypeError, match="Argument interpolation with value.*, but got <class \'str\'>"):
        vision.RandomCropDecodeResize(size=size, scale=scale, ratio=ratio, interpolation=interpolation)

    # When parameter max_attempts is 0, RandomCropDecodeResize interface call fails
    size = (500, 520)
    scale = [0.5, 1.0]
    ratio = [0.5, 1.0]
    interpolation = v_Inter.BILINEAR
    max_attempts = 0
    with pytest.raises(ValueError, match="Input max_attempts is not within the required interval of"
                                         " \\[1, 2147483647\\]."):
        vision.RandomCropDecodeResize(size=size, scale=scale, ratio=ratio, interpolation=interpolation,
                                      max_attempts=max_attempts)

    # When parameter max_attempts is float, RandomCropDecodeResize interface call fails
    dataset = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False)
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = v_Inter.BILINEAR
    max_attempts = 1.5
    with pytest.raises(TypeError, match="Argument max_attempts with value 1.5 is not of type \\[<class 'int'>\\],"
                                        " but got <class 'float'>."):
        random_crop_decode_resize_op = vision.RandomCropDecodeResize(size=size, scale=scale, ratio=ratio,
                                                                  interpolation=interpolation,
                                                                  max_attempts=max_attempts)
        dataset = dataset.map(input_columns=["image"], operations=random_crop_decode_resize_op)
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # When parameter max_attempts is empty, RandomCropDecodeResize interface call fails
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = v_Inter.BILINEAR
    max_attempts = ""
    with pytest.raises(TypeError, match="Argument max_attempts with value \"\" is not of type \\[<class 'int'>\\],"
                                        " but got <class 'str'>."):
        vision.RandomCropDecodeResize(size=size, scale=scale, ratio=ratio, interpolation=interpolation,
                                      max_attempts=max_attempts)

    # When no parameters are provided, RandomCropDecodeResize interface call fails
    with pytest.raises(TypeError, match="missing a required argument: 'size'"):
        vision.RandomCropDecodeResize()

    # When no size parameter is provided, RandomCropDecodeResize interface call fails
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = v_Inter.BILINEAR
    max_attempts = 10
    with pytest.raises(TypeError, match="missing a required argument"):
        vision.RandomCropDecodeResize(scale=scale, ratio=ratio, interpolation=interpolation, max_attempts=max_attempts)

    # When extra parameters are set, RandomCropDecodeResize interface call fails
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = v_Inter.BILINEAR
    max_attempts = 10
    more_para = None
    with pytest.raises(TypeError, match="too many positional arguments"):
        vision.RandomCropDecodeResize(size, scale, ratio, interpolation, max_attempts, more_para)

    # When input image format is gif, RandomCropDecodeResize interface call fails
    image = np.fromfile(image_gif, dtype=np.uint8)
    size = 600
    vc_randomcropdecoderesize = vision.RandomCropDecodeResize(size)
    with pytest.raises(RuntimeError,
                       match="Exception thrown from dataset pipeline. Refer to 'Dataset Pipeline Error Message'."):
        vc_randomcropdecoderesize(image)

    # When input is numpy, RandomCropDecodeResize interface call fails
    image = np.random.randint(0, 255, (500,)).astype(np.uint8)
    size = 600
    vc_randomcropdecoderesize = vision.RandomCropDecodeResize(size)
    with pytest.raises(RuntimeError, match="Decode: image decode failed."):
        vc_randomcropdecoderesize(image)

    # When input is non-1D numpy, RandomCropDecodeResize interface call fails
    with Image.open(image_jpg) as image:
        size = 600
        vc_randomcropdecoderesize = vision.RandomCropDecodeResize(size)
        with pytest.raises(TypeError, match="Input should be an encoded image in 1-D NumPy format, "
                                            "got <class 'PIL.JpegImagePlugin.JpegImageFile'>."):
            vc_randomcropdecoderesize(image)


def test_random_crop_decode_resize_exception_03():
    """
    Feature: RandomCropDecodeResize operation
    Description: Testing the RandomCropDecodeResize Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When input is list, RandomCropDecodeResize interface call fails
    image = np.fromfile(image_jpg, dtype=np.uint8).tolist()
    size = 600
    vc_randomcropdecoderesize = vision.RandomCropDecodeResize(size)
    with pytest.raises(TypeError, match="Input should be an encoded image in 1-D NumPy format, got <class 'list'>."):
        vc_randomcropdecoderesize(image)

    # When input is tuple, RandomCropDecodeResize interface call fails
    image = tuple(np.fromfile(image_jpg, dtype=np.uint8))
    size = 600
    vc_randomcropdecoderesize = vision.RandomCropDecodeResize(size)
    with pytest.raises(TypeError, match="Input should be an encoded image in 1-D NumPy format, got <class 'tuple'>."):
        vc_randomcropdecoderesize(image)

    # When no image is provided, RandomCropDecodeResize interface call fails
    size = 600
    vc_randomcropdecoderesize = vision.RandomCropDecodeResize(size)
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'img'"):
        vc_randomcropdecoderesize()

    # When parameter size is numpy array, RandomCropDecodeResize interface call fails
    size = np.array([50, 50])
    with pytest.raises(TypeError, match="Argument size with value \\[50 50\\] is not of "
                                        "type \\[<class 'int'>, <class 'list'>, <class 'tuple'>\\]."):
        vision.RandomCropDecodeResize(size=size)

    # When parameter size is set, RandomCropDecodeResize interface call fails
    size = {100}
    with pytest.raises(TypeError, match="Argument size with value \\{100\\} is not of type \\["
                                        "<class 'int'>, <class 'list'>, <class 'tuple'>\\]."):
        vision.RandomCropDecodeResize(size=size)

    # When parameter size is list, RandomCropDecodeResize interface call fails
    size = [100]
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple \\(h, w\\) of length 2."):
        vision.RandomCropDecodeResize(size=size)

    # When parameter size is bool, RandomCropDecodeResize interface call fails
    size = True
    with pytest.raises(TypeError, match="Argument size with value True is not of "
                                        "type \\(<class 'int'>, <class 'list'>, <class 'tuple'>\\)."):
        vision.RandomCropDecodeResize(size=size)

    # When parameter scale is 3-tuple, RandomCropDecodeResize interface call fails
    image = np.fromfile(image_bmp, dtype=np.uint8)
    size = 600
    scale = (0.5, 0.6, 0.8)
    with pytest.raises(TypeError, match="scale should be a list/tuple of length 2."):
        vc_randomcropdecoderesize = vision.RandomCropDecodeResize(size, scale)
        vc_randomcropdecoderesize(image)

    # When parameter scale is 0, RandomCropDecodeResize interface call fails
    size = [100, 100]
    scale = (0, 0)
    with pytest.raises(ValueError, match="Input scale\\[1\\] must be greater than 0."):
        vision.RandomCropDecodeResize(size=size, scale=scale)

    # When parameter scale is 1-tuple, RandomCropDecodeResize interface call fails
    size = [100, 100]
    scale = (0.5,)
    with pytest.raises(TypeError, match="scale should be a list/tuple of length 2."):
        vision.RandomCropDecodeResize(size=size, scale=scale)

    # When parameter scale is greater than 16777216, RandomCropDecodeResize interface call fails
    size = [100, 100]
    scale = (0.5, 16777216.1)
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[0, 16777216\\]."):
        vision.RandomCropDecodeResize(size=size, scale=scale)

    # When parameter scale is numpy array, RandomCropDecodeResize interface call fails
    size = [100, 100]
    scale = np.array([0.5, 0.8])
    with pytest.raises(TypeError, match="Argument scale with value \\[0.5 0.8\\] is "
                                        "not of type \\[<class 'tuple'>, <class 'list'>\\]"):
        vision.RandomCropDecodeResize(size=size, scale=scale)

    # When parameter scale is 3, RandomCropDecodeResize interface call fails
    size = [100, 100]
    scale = 3
    with pytest.raises(TypeError,
                       match="Argument scale with value 3 is not of type \\[<class 'tuple'>, <class 'list'>\\]."):
        vision.RandomCropDecodeResize(size=size, scale=scale)

    # When parameter ratio is 3-tuple, RandomCropDecodeResize interface call fails
    image = np.fromfile(image_bmp, dtype=np.uint8)
    size = 600
    ratio = (0.5, 0.6, 0.8)
    with pytest.raises(TypeError, match="ratio should be a list/tuple of length 2."):
        vc_randomcropdecoderesize = vision.RandomCropDecodeResize(size=size, ratio=ratio)
        vc_randomcropdecoderesize(image)

    # When parameter ratio is 0, RandomCropDecodeResize interface call fails
    size = [100, 100]
    ratio = (0, 0.5)
    with pytest.raises(ValueError, match="Input ratio\\[0\\] is not within the required interval of"
                                         " \\(0, 16777216\\]."):
        vision.RandomCropDecodeResize(size=size, ratio=ratio)


def test_random_crop_decode_resize_exception_04():
    """
    Feature: RandomCropDecodeResize operation
    Description: Testing the RandomCropDecodeResize Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When parameter ratio is 1-tuple, RandomCropDecodeResize interface call fails
    size = [100, 100]
    ratio = (0.5,)
    with pytest.raises(TypeError, match="ratio should be a list/tuple of length 2."):
        vision.RandomCropDecodeResize(size=size, ratio=ratio)

    # When parameter ratio is greater than 16777216, RandomCropDecodeResize interface call fails
    size = [100, 100]
    ratio = (0.5, 16777216.1)
    with pytest.raises(ValueError, match="Input ratio\\[1\\] is not within the required interval of"
                                         " \\(0, 16777216\\]."):
        vision.RandomCropDecodeResize(size=size, ratio=ratio)

    # When parameter ratio is numpy array, RandomCropDecodeResize interface call fails
    size = [100, 100]
    ratio = np.array([0.5, 0.8])
    with pytest.raises(TypeError, match="Argument ratio with value \\[0.5 0.8\\] is "
                                        "not of type \\[<class 'tuple'>, <class 'list'>\\]."):
        vision.RandomCropDecodeResize(size=size, ratio=ratio)

    # When parameter ratio is 3, RandomCropDecodeResize interface call fails
    size = [100, 100]
    ratio = 3
    with pytest.raises(TypeError,
                       match="Argument ratio with value 3 is not of type \\[<class 'tuple'>, <class 'list'>\\]."):
        vision.RandomCropDecodeResize(size=size, ratio=ratio)

    # When parameter interpolation is Inter, RandomCropDecodeResize interface call fails
    size = [100, 100]
    interpolation = v_Inter
    with pytest.raises(TypeError,
                       match="Argument interpolation with value <enum 'Inter'> is not of type \\[<enum 'Inter'>\\]."):
        vision.RandomCropDecodeResize(size=size, interpolation=interpolation)

    # When parameter interpolation is int, RandomCropDecodeResize interface call fails
    size = [100, 100]
    interpolation = 3
    with pytest.raises(TypeError, match="Argument interpolation with value 3 is not of type \\[<enum 'Inter'>\\]."):
        vision.RandomCropDecodeResize(size=size, interpolation=interpolation)

    # When parameter interpolation is list, RandomCropDecodeResize interface call fails
    size = [100, 100]
    interpolation = [v_Inter.BICUBIC]
    with pytest.raises(TypeError, match="Argument interpolation with value \\[<Inter.BICUBIC: 3>\\] is "
                                        "not of type \\[<enum 'Inter'>\\]."):
        vision.RandomCropDecodeResize(size=size, interpolation=interpolation)

    # When parameter max_attempts is list, RandomCropDecodeResize interface call fails
    size = [100, 100]
    max_attempts = [1]
    with pytest.raises(TypeError, match="Argument max_attempts with value \\[1\\] is not of type \\[<class 'int'>\\],"
                                        " but got <class 'list'>."):
        vision.RandomCropDecodeResize(size=size, max_attempts=max_attempts)

    # When parameter max_attempts is bool, RandomCropDecodeResize interface call fails
    image = np.fromfile(image_jpg, dtype=np.uint8)
    size = (500, 520)
    max_attempts = True
    with pytest.raises(TypeError, match="Argument max_attempts with value True is not of type \\(<class 'int'>,\\),"
                                        " but got <class 'bool'>."):
        vc_randomcropdecoderesize = vision.RandomCropDecodeResize(size=size, max_attempts=max_attempts)
        vc_randomcropdecoderesize(image)

    # When parameter max_attempts is greater than maximum value, RandomCropDecodeResize interface call fails
    size = [100, 100]
    max_attempts = 2147483648
    with pytest.raises(ValueError, match="Input max_attempts is not within the required interval of"
                                         " \\[1, 2147483647\\]."):
        vision.RandomCropDecodeResize(size=size, max_attempts=max_attempts)

    # When parameter max_attempts is float, RandomCropDecodeResize interface call fails
    image = np.fromfile(image_jpg, dtype=np.uint8)
    size = (500, 520)
    max_attempts = 2.0
    with pytest.raises(TypeError, match="Argument max_attempts with value 2.0 is not of type \\[<class 'int'>\\],"
                                        " but got <class 'float'>."):
        vc_randomcropdecoderesize = vision.RandomCropDecodeResize(size=size, max_attempts=max_attempts)
        vc_randomcropdecoderesize(image)

    # When ratio[0] in parameter ratio is greater than ratio[1], RandomCropDecodeResize interface call fails
    size = [100, 100]
    ratio = (0.1, 0.05)
    with pytest.raises(ValueError, match="ratio should be in \\(min,max\\) format. Got \\(max,min\\)."):
        vision.RandomCropDecodeResize(size=size, ratio=ratio)

    # When scale[0] in parameter scale is greater than scale[1], RandomCropDecodeResize interface call fails
    size = [100, 100]
    scale = (2, 1)
    with pytest.raises(ValueError, match="scale should be in \\(min,max\\) format. Got \\(max,min\\)."):
        vision.RandomCropDecodeResize(size=size, scale=scale)


if __name__ == "__main__":
    test_random_crop_decode_resize_op(plot=True)
    test_random_crop_decode_resize_md5()
    test_random_crop_decode_resize_invalid()
    test_random_crop_decode_resize_operation_01()
    test_random_crop_decode_resize_operation_02()
    test_random_crop_decode_resize_exception_01()
    test_random_crop_decode_resize_exception_02()
    test_random_crop_decode_resize_exception_03()
    test_random_crop_decode_resize_exception_04()
