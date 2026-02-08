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
Testing Decode op in DE
"""
import glob
import os

import cv2
import numpy as np
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.transforms.transforms as t_trans
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger
from mindspore.common.tensor import Tensor
from util import diff_mse

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
TEST_DATA_DATASET_FUNC ="../data/dataset/"


def dir_data():
    """Obtain the dataset"""
    data_list = []
    data_dir1 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "testImageNetData", "train")
    data_dir3 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    data_dir4 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
    data_dir5 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    data_dir6 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")
    data_list.append(data_dir1)
    data_list.append(data_dir3)
    data_list.append(data_dir4)
    data_list.append(data_dir5)
    data_list.append(data_dir6)
    return data_list


def test_decode_op():
    """
    Feature: Decode Op
    Description: Test C++ implementation
    Expectation: Dataset pipeline runs successfully and results are verified
    """
    logger.info("test_decode_op")

    # Serialize and Load dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

    # Decode with rgb format set to True
    data1 = data1.map(operations=[vision.Decode()], input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        actual = item1["image"]
        expected = cv2.imdecode(item2["image"], cv2.IMREAD_COLOR)
        expected = cv2.cvtColor(expected, cv2.COLOR_BGR2RGB)
        assert actual.shape == expected.shape
        mse = diff_mse(actual, expected)
        assert mse == 0


def test_decode_op_support_format():
    """
    Feature: Decode Op
    Description: Test support format of decode op
    Expectation: decode image successfully
    """
    c_decode = vision.Decode(to_pil=False)
    p_decode = vision.Decode(to_pil=True)

    # jpeg: Opencv[√] Pillow[√]
    jpg_image = np.fromfile("../data/dataset/testFormats/apple.jpg", np.uint8)
    c_decode(jpg_image)
    p_decode(jpg_image)

    # bmp: Opencv[√] Pillow[√]
    bmp_image = np.fromfile("../data/dataset/testFormats/apple.bmp", np.uint8)
    c_decode(bmp_image)
    p_decode(bmp_image)

    # png: Opencv[√] Pillow[√]
    png_image = np.fromfile("../data/dataset/testFormats/apple.png", np.uint8)
    c_decode(png_image)
    p_decode(png_image)

    # tiff: Opencv[√] Pillow[√]
    tiff_image = np.fromfile("../data/dataset/testFormats/apple.tiff", np.uint8)
    c_decode(tiff_image)
    p_decode(tiff_image)

    unsupported_list = glob.glob('unsupported_image*')
    for item in unsupported_list:
        os.remove(item)

    # gif: Opencv[×] Pillow[√]
    gif_image = np.fromfile("../data/dataset/testFormats/apple.gif", np.uint8)
    with pytest.raises(RuntimeError, match="Unsupported image type"):
        c_decode(gif_image)
    p_decode(gif_image)

    assert len(glob.glob('unsupported_image.gif')) == 1
    # delete the dump file which is not supported
    os.remove(glob.glob('unsupported_image.gif')[0])

    # webp: Opencv[×] Pillow[√]
    webp_image = np.fromfile("../data/dataset/testFormats/apple.webp", np.uint8)
    with pytest.raises(RuntimeError, match="Unsupported image type"):
        c_decode(webp_image)
    p_decode(webp_image)

    abnormal_list = glob.glob('abnormal_image*')
    for item in abnormal_list:
        os.remove(item)

    assert len(glob.glob('unsupported_image.webp')) == 1
    # delete the dump file which is not supported
    os.remove(glob.glob('unsupported_image.webp')[0])

    # abnormal image: Opencv[x] Pillow[x]
    abnormal_image = np.fromfile("../data/dataset/testFormats/abnormal_apple.jpg", np.uint8)
    with pytest.raises(RuntimeError, match="Dump the abnormal image to"):
        c_decode(abnormal_image)
    with pytest.raises(ValueError, match="image file is truncated"):
        p_decode(abnormal_image)

    assert len(glob.glob('abnormal_image.jpg')) == 1
    # delete the dump file which is abnormal
    os.remove(glob.glob('abnormal_image.jpg')[0])


class ImageDataset:
    """Custom class to generate and read image dataset"""

    def __init__(self, data_path, data_type="numpy"):
        self.data = [data_path]
        self.label = np.random.sample((1, 1))
        self.data_type = data_type

    def __getitem__(self, index):
        # use file open and read method
        with open(self.data[index], 'rb') as f:
            img_bytes = [f.read()]
        if self.data_type == "numpy":
            img_bytes = np.array(img_bytes)

        # Return bytes directly
        return img_bytes, self.label[index]

    def __len__(self):
        return len(self.data)


def test_read_image_decode_op():
    """
    Feature: Decode Op
    Description: Test Python implementation
    Expectation: Dataset pipeline runs successfully and results are verified
    """
    data_path = "../data/dataset/testPK/data/class1/0.jpg"
    dataset1 = ds.GeneratorDataset(ImageDataset(data_path, data_type="numpy"), ["data", "label"])
    dataset2 = ds.GeneratorDataset(ImageDataset(data_path, data_type="bytes"), ["data", "label"])
    decode_op = vision.Decode(to_pil=True)
    to_tensor = vision.ToTensor(output_type=np.int32)
    dataset1 = dataset1.map(operations=[decode_op, to_tensor], input_columns=["data"])
    dataset2 = dataset2.map(operations=[decode_op, to_tensor], input_columns=["data"])

    for item1, item2 in zip(dataset1, dataset2):
        np.allclose(item1[0].asnumpy(), item2[0].asnumpy())


def test_decode_operation_01():
    """
    Feature: Decode operation
    Description: Testing the normal functionality of the Decode operator
    Expectation: The Output is equal to the expected output
    """
    # Decode Normal Scenarios: Test input is jpg, as type is uint8
    image = np.fromfile(dir_data()[1], dtype=np.uint8)
    decode_op = vision.Decode()
    _ = decode_op(image)

    # Decode Normal Scenarios: Test input is jpg, as type is int32
    image = np.fromfile(dir_data()[1], dtype=np.int32)
    decode_op = vision.Decode()
    decode_op(image)
    _ = decode_op(image)

    # Decode Normal Scenarios: Test parameters to_pil is False
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    decode_op = vision.Decode()
    dataset2 = dataset2.map(input_columns=["image"], operations=decode_op)
    for _ in dataset2.create_dict_iterator():
        pass

    # Decode Normal Scenarios: Test parameters to_pil is True
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    decode_op = vision.Decode(to_pil=True)
    dataset2 = dataset2.map(input_columns=["image"], operations=decode_op)
    for _ in dataset2.create_dict_iterator():
        pass

    # Decode Normal Scenarios: Test parameters to_pil is 1
    with pytest.raises(TypeError,
                       match=r"Argument to_pil with value 1 is not of "
                             r"type \[<class 'bool'>\], but got <class 'int'>."):
        vision.Decode(to_pil=1)

    # Decode Normal Scenarios: Test parameters to_pil is 'a'
    with pytest.raises(TypeError,
                       match=r"Argument to_pil with value a is not of "
                             r"type \[<class 'bool'>\], but got <class 'str'>."):
        vision.Decode(to_pil='a')

    # Decode Normal Scenarios: Test parameters to_pil is None
    with pytest.raises(TypeError,
                       match=r"Argument to_pil with value None is not of "
                             r"type \[<class 'bool'>\], but got <class 'NoneType'>."):
        vision.Decode(to_pil=None)

    # Decode Normal Scenarios: The input data has been decoded.
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False, decode=True)
    with pytest.raises(RuntimeError,
                       match=r"invalid input shape, only support 1D input, got rank: 3"):
        decode_op = vision.Decode()
        dataset2 = dataset2.map(input_columns=["image"], operations=decode_op)
        for _ in dataset2.create_dict_iterator():
            pass

    # Decode Normal Scenarios: Test normal.
    ds1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    transforms = [
        vision.Decode(to_pil=True),
        vision.ToTensor()
    ]
    transform = t_trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)

    for _ in ds1.create_dict_iterator(output_numpy=True):
        pass

    # Decode Normal Scenarios: to_pil is True,input is np(jpg)
    image = np.fromfile(dir_data()[1], dtype=np.uint8)
    decode_op = vision.Decode(to_pil=True)
    _ = decode_op(image)

    # Decode Normal Scenarios: to_pil is True,input is np(png)
    image = np.fromfile(dir_data()[3], dtype=np.uint8)
    decode_op = vision.Decode(to_pil=True)
    _ = decode_op(image)

    # Decode Normal Scenarios: to_pil is True,input is np(bmg)
    image = np.fromfile(dir_data()[2], dtype=np.uint8)
    decode_op = vision.Decode(to_pil=True)
    _ = decode_op(image)

    # Decode Normal Scenarios: to_pil is True,input is np(gif)
    image = np.fromfile(dir_data()[4], dtype=np.uint8)
    decode_op = vision.Decode(to_pil=True)
    _ = decode_op(image)

    # Decode Normal Scenarios: Test to_pil is 1.1
    with pytest.raises(TypeError, match=r"Argument to_pil with value 1.1 is not of "
                                        r"type \[<class 'bool'>\], but got <class 'float'>"):
        vision.Decode(to_pil=1.1)


def test_decode_exception_01():
    """
    Feature: Decode operation
    Description: Testing the Decode Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Decode Exception Scenarios: Test error decode
    dataset2 = ds.ImageFolderDataset(dir_data()[0], 1, shuffle=False, decode=True)
    with pytest.raises(RuntimeError, match=" Decode: invalid input shape, only support 1D input"):
        decode_op = vision.Decode()
        dataset2 = dataset2.map(input_columns=["image"], operations=decode_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # Decode Exception Scenarios: Test input is list
    image = np.fromfile(dir_data()[1], dtype=np.float16).tolist()
    decode_op = vision.Decode()
    with pytest.raises(TypeError, match="The type of the encoded image should be <class 'numpy.ndarray'>, "
                                        "but got <class 'list'>."):
        decode_op(image)

    # Decode Exception Scenarios: Test input is Tensor
    image = Tensor(np.fromfile(dir_data()[1], dtype=np.float16))
    decode_op = vision.Decode()
    with pytest.raises(TypeError, match="The type of the encoded image should be <class 'numpy.ndarray'>, "
                                        "but got <class 'mindspore.common.tensor.Tensor'>."):
        decode_op(image)

    # Decode Exception Scenarios: Test input is 2d
    image = np.random.randn(64, 3)
    decode_op = vision.Decode()
    with pytest.raises(TypeError, match="The number of array dimensions of the encoded image should be 1, but got 2."):
        decode_op(image)

    # Decode Exception Scenarios: Do not transmit input data
    decode_op = vision.Decode()
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'img'"):
        decode_op()

    # Decode Exception Scenarios: Input data passed in 1
    decode_op = vision.Decode()
    with pytest.raises(TypeError, match="The type of the encoded image should be <class 'numpy.ndarray'>, "
                                        "but got <class 'int'>."):
        decode_op(1)

    # Decode Exception Scenarios: to_pil is True,input is np
    ds1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False, decode=True)
    with pytest.raises(RuntimeError,
                       match="The number of array dimensions of the encoded image should be 1, but got 3."):
        transforms = [
            vision.Decode(to_pil=True),
            vision.ToTensor()
        ]
        transform = t_trans.Compose(transforms)
        ds1 = ds1.map(input_columns=["image"], operations=transform)

        for _ in ds1.create_dict_iterator(output_numpy=True):
            pass

    # Decode Exception Scenarios: Test One more parameters
    ds1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    with pytest.raises(TypeError, match="too many positional arguments"):
        transforms = [
            vision.Decode(True, True),
            vision.ToTensor()
        ]
        transform = t_trans.Compose(transforms)
        ds1 = ds1.map(input_columns=["image"], operations=transform)

        for _ in ds1.create_dict_iterator(output_numpy=True):
            pass

    # Decode Exception Scenarios: Test input is numpy
    image = np.random.randint(0, 255, (128,)).astype(np.uint8)
    decode_op = vision.Decode(to_pil=True)
    with pytest.raises(ValueError, match="cannot identify image file"):
        decode_op(image)

    # Decode Exception Scenarios: Test input is PIL image
    with Image.open(dir_data()[1]) as image:
        decode_op = vision.Decode(to_pil=True)
        with pytest.raises(TypeError, match="The type of the encoded image should be <class 'numpy.ndarray'>, "
                                            "but got <class 'PIL.JpegImagePlugin.JpegImageFile'>."):
            decode_op(image)

    # Decode Exception Scenarios: Test don't have input
    decode_op = vision.Decode(to_pil=True)
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'img'"):
        decode_op()

    # Decode Exception Scenarios: Test input is int
    decode_op = vision.Decode(to_pil=True)
    with pytest.raises(TypeError, match="The type of the encoded image should be <class 'numpy.ndarray'>, "
                                        "but got <class 'int'>."):
        decode_op(10)

    # Decode Exception Scenarios: Test input is list
    image = np.fromfile(dir_data()[1], dtype=np.uint8).tolist()
    decode_op = vision.Decode(to_pil=True)
    with pytest.raises(TypeError, match="The type of the encoded image should be <class 'numpy.ndarray'>, "
                                        "but got <class 'list'>."):
        decode_op(image)


if __name__ == "__main__":
    test_decode_op()
    test_decode_op_support_format()
    test_read_image_decode_op()
    test_decode_operation_01()
    test_decode_exception_01()
