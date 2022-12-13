# Copyright 2019-2022 Huawei Technologies Co., Ltd
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

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import log as logger
from util import diff_mse

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


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


if __name__ == "__main__":
    test_decode_op()
    test_decode_op_support_format()
    test_read_image_decode_op()
