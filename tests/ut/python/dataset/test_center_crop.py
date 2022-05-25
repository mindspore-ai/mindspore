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
Testing CenterCrop op in DE
"""
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.transforms
import mindspore.dataset.vision as vision
from mindspore import log as logger
from util import diff_mse, visualize_list, save_and_check_md5

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_center_crop_op(height=375, width=375, plot=False):
    """
    Feature: CenterCrop op
    Description: Test CenterCrop op basic usage
    Expectation: Runs successfully
    """
    logger.info("Test CenterCrop")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"])
    decode_op = vision.Decode()
    # 3 images [375, 500] [600, 500] [512, 512]
    center_crop_op = vision.CenterCrop([height, width])
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=center_crop_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"])
    data2 = data2.map(operations=decode_op, input_columns=["image"])

    image_cropped = []
    image = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_cropped.append(item1["image"].copy())
        image.append(item2["image"].copy())
    if plot:
        visualize_list(image, image_cropped)


def test_center_crop_md5(height=375, width=375):
    """
    Feature: CenterCrop op
    Description: Test CenterCrop using md5 check test
    Expectation: Passes the md5 check test
    """
    logger.info("Test CenterCrop")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    decode_op = vision.Decode()
    # 3 images [375, 500] [600, 500] [512, 512]
    center_crop_op = vision.CenterCrop([height, width])
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=center_crop_op, input_columns=["image"])
    # Compare with expected md5 from images
    filename = "center_crop_01_result.npz"
    save_and_check_md5(data1, filename, generate_golden=GENERATE_GOLDEN)


def test_center_crop_comp(height=375, width=375, plot=False):
    """
    Feature: CenterCrop op
    Description: Test CenterCrop between Python and Cpp image augmentation
    Expectation: Resulting outputs from both operations are expected to be equal
    """
    logger.info("Test CenterCrop")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    decode_op = vision.Decode()
    center_crop_op = vision.CenterCrop([height, width])
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=center_crop_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    transforms = [
        vision.Decode(True),
        vision.CenterCrop([height, width]),
        vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.Compose(transforms)
    data2 = data2.map(operations=transform, input_columns=["image"])

    image_c_cropped = []
    image_py_cropped = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        c_image = item1["image"]
        py_image = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        # Note: The images aren't exactly the same due to rounding error
        assert diff_mse(py_image, c_image) < 0.001
        image_c_cropped.append(c_image.copy())
        image_py_cropped.append(py_image.copy())
    if plot:
        visualize_list(image_c_cropped, image_py_cropped, visualize_mode=2)


def test_crop_grayscale(height=375, width=375):
    """
    Feature: CenterCrop op
    Description: Test CenterCrop works with pad and grayscale images
    Expectation: Runs successfully
    """

    # Note: image.transpose performs channel swap to allow py transforms to
    # work with c transforms
    transforms = [
        vision.Decode(True),
        vision.Grayscale(1),
        vision.ToTensor(),
        (lambda image: (image.transpose(1, 2, 0) * 255).astype(np.uint8))
    ]

    transform = mindspore.dataset.transforms.Compose(transforms)
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    data1 = data1.map(operations=transform, input_columns=["image"])

    # If input is grayscale, the output dimensions should be single channel
    crop_gray = vision.CenterCrop([height, width])
    data1 = data1.map(operations=crop_gray, input_columns=["image"])

    for item1 in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        c_image = item1["image"]

        # Check that the image is grayscale
        assert (c_image.ndim == 3 and c_image.shape[2] == 1)


def test_center_crop_errors():
    """
    Feature: CenterCrop op
    Description: Test CenterCrop with bad inputs
    Expectation: Error is raised as expected
    """
    try:
        test_center_crop_op(16777216, 16777216)
    except RuntimeError as e:
        assert "CenterCropOp padding size is more than 3 times the original size" in \
               str(e)


if __name__ == "__main__":
    test_center_crop_op(600, 600, plot=True)
    test_center_crop_op(300, 600)
    test_center_crop_op(600, 300)
    test_center_crop_md5()
    test_center_crop_comp(plot=True)
    test_crop_grayscale()
