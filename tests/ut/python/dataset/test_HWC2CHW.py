# Copyright 2020 Huawei Technologies Co., Ltd
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
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.c_transforms as c_vision
import mindspore.dataset.transforms.vision.py_transforms as py_vision
from mindspore import log as logger
from util import diff_mse, visualize, save_and_check_md5

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_HWC2CHW(plot=False):
    """
    Test HWC2CHW
    """
    logger.info("Test HWC2CHW")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = c_vision.Decode()
    hwc2chw_op = c_vision.HWC2CHW()
    data1 = data1.map(input_columns=["image"], operations=decode_op)
    data1 = data1.map(input_columns=["image"], operations=hwc2chw_op)

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(input_columns=["image"], operations=decode_op)

    image_transposed = []
    image = []
    for item1, item2 in zip(data1.create_dict_iterator(), data2.create_dict_iterator()):
        transposed_item = item1["image"].copy()
        original_item = item2["image"].copy()
        image_transposed.append(transposed_item.transpose(1, 2, 0))
        image.append(original_item)

        # check if the shape of data is transposed correctly
        # transpose the original image from shape (H,W,C) to (C,H,W)
        mse = diff_mse(transposed_item, original_item.transpose(2, 0, 1))
        assert mse == 0
    if plot:
        visualize(image, image_transposed)


def test_HWC2CHW_md5():
    """
    Test HWC2CHW(md5)
    """
    logger.info("Test HWC2CHW with md5 comparison")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = c_vision.Decode()
    hwc2chw_op = c_vision.HWC2CHW()
    data1 = data1.map(input_columns=["image"], operations=decode_op)
    data1 = data1.map(input_columns=["image"], operations=hwc2chw_op)

    # Compare with expected md5 from images
    filename = "HWC2CHW_01_result.npz"
    save_and_check_md5(data1, filename, generate_golden=GENERATE_GOLDEN)


def test_HWC2CHW_comp(plot=False):
    """
    Test HWC2CHW between python and c image augmentation
    """
    logger.info("Test HWC2CHW with c_transform and py_transform comparison")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = c_vision.Decode()
    hwc2chw_op = c_vision.HWC2CHW()
    data1 = data1.map(input_columns=["image"], operations=decode_op)
    data1 = data1.map(input_columns=["image"], operations=hwc2chw_op)

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        py_vision.Decode(),
        py_vision.ToTensor(),
        py_vision.HWC2CHW()
    ]
    transform = py_vision.ComposeOp(transforms)
    data2 = data2.map(input_columns=["image"], operations=transform())

    image_c_transposed = []
    image_py_transposed = []
    for item1, item2 in zip(data1.create_dict_iterator(), data2.create_dict_iterator()):
        c_image = item1["image"]
        py_image = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)

        # Compare images between that applying c_transform and py_transform
        mse = diff_mse(py_image, c_image)
        # Note: The images aren't exactly the same due to rounding error
        assert mse < 0.001
        image_c_transposed.append(c_image.transpose(1, 2, 0))
        image_py_transposed.append(py_image.transpose(1, 2, 0))
    if plot:
        visualize(image_c_transposed, image_py_transposed)


if __name__ == '__main__':
    test_HWC2CHW(True)
    test_HWC2CHW_md5()
    test_HWC2CHW_comp(True)
