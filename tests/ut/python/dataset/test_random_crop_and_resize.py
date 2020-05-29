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
"""
Testing RandomCropAndResize op in DE
"""
import numpy as np
import cv2

import mindspore.dataset.transforms.vision.c_transforms as c_vision
import mindspore.dataset.transforms.vision.py_transforms as py_vision
import mindspore.dataset.transforms.vision.utils as mode
import mindspore.dataset as ds
from mindspore import log as logger
from util import diff_mse, save_and_check_md5, visualize

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

GENERATE_GOLDEN = False


def test_random_crop_and_resize_op(plot=False):
    """
    Test RandomCropAndResize op
    """
    logger.info("test_random_crop_and_resize_op")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = c_vision.Decode()
    random_crop_and_resize_op = c_vision.RandomResizedCrop((256, 512), (1, 1), (0.5, 0.5))
    data1 = data1.map(input_columns=["image"], operations=decode_op)
    data1 = data1.map(input_columns=["image"], operations=random_crop_and_resize_op)

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(input_columns=["image"], operations=decode_op)
    num_iter = 0
    crop_and_resize_images = []
    original_images = []
    for item1, item2 in zip(data1.create_dict_iterator(), data2.create_dict_iterator()):
        crop_and_resize = item1["image"]
        original = item2["image"]
        # Note: resize the original image with the same size as the one applied RandomResizedCrop()
        original = cv2.resize(original, (512, 256))
        mse = diff_mse(crop_and_resize, original)
        assert mse == 0
        logger.info("random_crop_and_resize_op_{}, mse: {}".format(num_iter + 1, mse))
        num_iter += 1
        crop_and_resize_images.append(crop_and_resize)
        original_images.append(original)
    if plot:
        visualize(original_images, crop_and_resize_images)

def test_random_crop_and_resize_01():
    """
    Test RandomCropAndResize with md5 check, expected to pass
    """
    logger.info("test_random_crop_and_resize_01")
    ds.config.set_seed(0)
    ds.config.set_num_parallel_workers(1)

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = c_vision.Decode()
    random_crop_and_resize_op = c_vision.RandomResizedCrop((256, 512), (0.5, 1), (0.5, 1))
    data1 = data1.map(input_columns=["image"], operations=decode_op)
    data1 = data1.map(input_columns=["image"], operations=random_crop_and_resize_op)

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        py_vision.Decode(),
        py_vision.RandomResizedCrop((256, 512), (0.5, 1), (0.5, 1)),
        py_vision.ToTensor()
    ]
    transform = py_vision.ComposeOp(transforms)
    data2 = data2.map(input_columns=["image"], operations=transform())

    filename1 = "random_crop_and_resize_01_c_result.npz"
    filename2 = "random_crop_and_resize_01_py_result.npz"
    save_and_check_md5(data1, filename1, generate_golden=GENERATE_GOLDEN)
    save_and_check_md5(data2, filename2, generate_golden=GENERATE_GOLDEN)

def test_random_crop_and_resize_02():
    """
    Test RandomCropAndResize with md5 check:Image interpolation mode is Inter.NEAREST,
    expected to pass
    """
    logger.info("test_random_crop_and_resize_02")
    ds.config.set_seed(0)
    ds.config.set_num_parallel_workers(1)

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = c_vision.Decode()
    random_crop_and_resize_op = c_vision.RandomResizedCrop((256, 512), interpolation=mode.Inter.NEAREST)
    data1 = data1.map(input_columns=["image"], operations=decode_op)
    data1 = data1.map(input_columns=["image"], operations=random_crop_and_resize_op)

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        py_vision.Decode(),
        py_vision.RandomResizedCrop((256, 512), interpolation=mode.Inter.NEAREST),
        py_vision.ToTensor()
    ]
    transform = py_vision.ComposeOp(transforms)
    data2 = data2.map(input_columns=["image"], operations=transform())

    filename1 = "random_crop_and_resize_02_c_result.npz"
    filename2 = "random_crop_and_resize_02_py_result.npz"
    save_and_check_md5(data1, filename1, generate_golden=GENERATE_GOLDEN)
    save_and_check_md5(data2, filename2, generate_golden=GENERATE_GOLDEN)

def test_random_crop_and_resize_03():
    """
    Test RandomCropAndResize with md5 check: max_attempts is 1, expected to pass
    """
    logger.info("test_random_crop_and_resize_03")
    ds.config.set_seed(0)
    ds.config.set_num_parallel_workers(1)

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = c_vision.Decode()
    random_crop_and_resize_op = c_vision.RandomResizedCrop((256, 512), max_attempts=1)
    data1 = data1.map(input_columns=["image"], operations=decode_op)
    data1 = data1.map(input_columns=["image"], operations=random_crop_and_resize_op)

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        py_vision.Decode(),
        py_vision.RandomResizedCrop((256, 512), max_attempts=1),
        py_vision.ToTensor()
    ]
    transform = py_vision.ComposeOp(transforms)
    data2 = data2.map(input_columns=["image"], operations=transform())

    filename1 = "random_crop_and_resize_03_c_result.npz"
    filename2 = "random_crop_and_resize_03_py_result.npz"
    save_and_check_md5(data1, filename1, generate_golden=GENERATE_GOLDEN)
    save_and_check_md5(data2, filename2, generate_golden=GENERATE_GOLDEN)

def test_random_crop_and_resize_04_c():
    """
    Test RandomCropAndResize with c_tranforms: invalid range of scale (max<min),
    expected to raise ValueError
    """
    logger.info("test_random_crop_and_resize_04_c")
    ds.config.set_seed(0)
    ds.config.set_num_parallel_workers(1)

    try:
        # Generate dataset
        data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
        decode_op = c_vision.Decode()
        # If input range of scale is not in the order of (min, max), ValueError will be raised.
        random_crop_and_resize_op = c_vision.RandomResizedCrop((256, 512), (1, 0.5), (0.5, 0.5))
        data = data.map(input_columns=["image"], operations=decode_op)
        data = data.map(input_columns=["image"], operations=random_crop_and_resize_op)
        image_list = []
        for item in data.create_dict_iterator():
            image = item["image"]
            image_list.append(image.shape)
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input range is not valid" in str(e)

def test_random_crop_and_resize_04_py():
    """
    Test RandomCropAndResize with py_transforms: invalid range of scale (max<min),
    expected to raise ValueError
    """
    logger.info("test_random_crop_and_resize_04_py")
    ds.config.set_seed(0)
    ds.config.set_num_parallel_workers(1)

    try:
        # Generate dataset
        data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
        transforms = [
            py_vision.Decode(),
            # If input range of scale is not in the order of (min, max), ValueError will be raised.
            py_vision.RandomResizedCrop((256, 512), (1, 0.5), (0.5, 0.5)),
            py_vision.ToTensor()
        ]
        transform = py_vision.ComposeOp(transforms)
        data = data.map(input_columns=["image"], operations=transform())
        image_list = []
        for item in data.create_dict_iterator():
            image = item["image"]
            image_list.append(image.shape)
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input range is not valid" in str(e)

def test_random_crop_and_resize_05_c():
    """
    Test RandomCropAndResize with c_transforms: invalid range of ratio (max<min),
    expected to raise ValueError
    """
    logger.info("test_random_crop_and_resize_05_c")
    ds.config.set_seed(0)
    ds.config.set_num_parallel_workers(1)

    try:
        # Generate dataset
        data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
        decode_op = c_vision.Decode()
        random_crop_and_resize_op = c_vision.RandomResizedCrop((256, 512), (1, 1), (1, 0.5))
        # If input range of ratio is not in the order of (min, max), ValueError will be raised.
        data = data.map(input_columns=["image"], operations=decode_op)
        data = data.map(input_columns=["image"], operations=random_crop_and_resize_op)
        image_list = []
        for item in data.create_dict_iterator():
            image = item["image"]
            image_list.append(image.shape)
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input range is not valid" in str(e)

def test_random_crop_and_resize_05_py():
    """
    Test RandomCropAndResize with py_transforms: invalid range of ratio (max<min),
    expected to raise ValueError
    """
    logger.info("test_random_crop_and_resize_05_py")
    ds.config.set_seed(0)
    ds.config.set_num_parallel_workers(1)

    try:
        # Generate dataset
        data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
        transforms = [
            py_vision.Decode(),
            # If input range of ratio is not in the order of (min, max), ValueError will be raised.
            py_vision.RandomResizedCrop((256, 512), (1, 1), (1, 0.5)),
            py_vision.ToTensor()
        ]
        transform = py_vision.ComposeOp(transforms)
        data = data.map(input_columns=["image"], operations=transform())
        image_list = []
        for item in data.create_dict_iterator():
            image = item["image"]
            image_list.append(image.shape)
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input range is not valid" in str(e)

def test_random_crop_and_resize_comp(plot=False):
    """
    Test RandomCropAndResize and compare between python and c image augmentation
    """
    logger.info("test_random_crop_and_resize_comp")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = c_vision.Decode()
    random_crop_and_resize_op = c_vision.RandomResizedCrop(512, (1, 1), (0.5, 0.5))
    data1 = data1.map(input_columns=["image"], operations=decode_op)
    data1 = data1.map(input_columns=["image"], operations=random_crop_and_resize_op)

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        py_vision.Decode(),
        py_vision.RandomResizedCrop(512, (1, 1), (0.5, 0.5)),
        py_vision.ToTensor()
    ]
    transform = py_vision.ComposeOp(transforms)
    data2 = data2.map(input_columns=["image"], operations=transform())

    image_c_cropped = []
    image_py_cropped = []
    for item1, item2 in zip(data1.create_dict_iterator(), data2.create_dict_iterator()):
        c_image = item1["image"]
        py_image = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_c_cropped.append(c_image)
        image_py_cropped.append(py_image)
    if plot:
        visualize(image_c_cropped, image_py_cropped)

if __name__ == "__main__":
    test_random_crop_and_resize_op(True)
    test_random_crop_and_resize_01()
    test_random_crop_and_resize_02()
    test_random_crop_and_resize_03()
    test_random_crop_and_resize_04_c()
    test_random_crop_and_resize_04_py()
    test_random_crop_and_resize_05_c()
    test_random_crop_and_resize_05_py()
    test_random_crop_and_resize_comp(True)
