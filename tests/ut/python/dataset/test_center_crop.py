# Copyright 2019 Huawei Technologies Co., Ltd
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
import mindspore.dataset.transforms.vision.c_transforms as vision
import mindspore.dataset.transforms.vision.py_transforms as py_vision
import numpy as np
import matplotlib.pyplot as plt
import mindspore.dataset as ds
from mindspore import log as logger
from util import diff_mse, visualize, save_and_check_md5

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_center_crop_op(height=375, width=375, plot=False):
    """
    Test CenterCrop
    """
    logger.info("Test CenterCrop")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"])
    decode_op = vision.Decode()
    # 3 images [375, 500] [600, 500] [512, 512]
    center_crop_op = vision.CenterCrop([height, width])
    data1 = data1.map(input_columns=["image"], operations=decode_op)
    data1 = data1.map(input_columns=["image"], operations=center_crop_op)

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"])
    data2 = data2.map(input_columns=["image"], operations=decode_op)

    image_cropped = []
    image = []
    for item1, item2 in zip(data1.create_dict_iterator(), data2.create_dict_iterator()):
        image_cropped.append(item1["image"].copy())
        image.append(item2["image"].copy())
    if plot:
        visualize(image, image_cropped)


def test_center_crop_md5(height=375, width=375):
    """
    Test CenterCrop
    """
    logger.info("Test CenterCrop")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle =False)
    decode_op = vision.Decode()
    # 3 images [375, 500] [600, 500] [512, 512]
    center_crop_op = vision.CenterCrop([height, width])
    data1 = data1.map(input_columns=["image"], operations=decode_op)
    data1 = data1.map(input_columns=["image"], operations=center_crop_op)
    # expected md5 from images 

    filename = "test_center_crop_01_result.npz"
    save_and_check_md5(data1, filename, generate_golden=GENERATE_GOLDEN)


def test_center_crop_comp(height=375, width=375, plot=False):
    """
    Test CenterCrop between python and c image augmentation
    """
    logger.info("Test CenterCrop")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    center_crop_op = vision.CenterCrop([height, width])
    data1 = data1.map(input_columns=["image"], operations=decode_op)
    data1 = data1.map(input_columns=["image"], operations=center_crop_op)

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        py_vision.Decode(),
        py_vision.CenterCrop([height, width]), 
        py_vision.ToTensor()
    ]
    transform = py_vision.ComposeOp(transforms)
    data2 = data2.map(input_columns=["image"], operations=transform())

    image_cropped = []
    image = []
    for item1, item2 in zip(data1.create_dict_iterator(), data2.create_dict_iterator()):
        c_image = item1["image"]
        py_image = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        # the images aren't exactly the same due to rouding error 
        assert (diff_mse(py_image, c_image) < 0.001)
        image_cropped.append(item1["image"].copy())
        image.append(item2["image"].copy())
    if plot:
        visualize(image, image_cropped)


if __name__ == "__main__":
    test_center_crop_op(600, 600)
    test_center_crop_op(300, 600)
    test_center_crop_op(600, 300)
    test_center_crop_md5(600, 600)
    test_center_crop_comp()
