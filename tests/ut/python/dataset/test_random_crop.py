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
Testing RandomCrop op in DE
"""
import numpy as np

import mindspore.dataset.transforms.py_transforms
import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.vision.py_transforms as py_vision
import mindspore.dataset.vision.utils as mode
import mindspore.dataset as ds
from mindspore import log as logger
from util import save_and_check_md5, visualize_list, config_get_set_seed, \
    config_get_set_num_parallel_workers


GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_random_crop_op_c(plot=False):
    """
    Test RandomCrop Op in c transforms
    """
    logger.info("test_random_crop_op_c")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    random_crop_op = c_vision.RandomCrop([512, 512], [200, 200, 200, 200])
    decode_op = c_vision.Decode()

    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=random_crop_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=decode_op, input_columns=["image"])

    image_cropped = []
    image = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image1 = item1["image"]
        image2 = item2["image"]
        image_cropped.append(image1)
        image.append(image2)
    if plot:
        visualize_list(image, image_cropped)

def test_random_crop_op_py(plot=False):
    """
    Test RandomCrop op in py transforms
    """
    logger.info("test_random_crop_op_py")
    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms1 = [
        py_vision.Decode(),
        py_vision.RandomCrop([512, 512], [200, 200, 200, 200]),
        py_vision.ToTensor()
    ]
    transform1 = mindspore.dataset.transforms.py_transforms.Compose(transforms1)
    data1 = data1.map(operations=transform1, input_columns=["image"])
    # Second dataset
    # Second dataset for comparison
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms2 = [
        py_vision.Decode(),
        py_vision.ToTensor()
    ]
    transform2 = mindspore.dataset.transforms.py_transforms.Compose(transforms2)
    data2 = data2.map(operations=transform2, input_columns=["image"])

    crop_images = []
    original_images = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        crop = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        original = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        crop_images.append(crop)
        original_images.append(original)
    if plot:
        visualize_list(original_images, crop_images)

def test_random_crop_01_c():
    """
    Test RandomCrop op with c_transforms: size is a single integer, expected to pass
    """
    logger.info("test_random_crop_01_c")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # Note: If size is an int, a square crop of size (size, size) is returned.
    random_crop_op = c_vision.RandomCrop(512)
    decode_op = c_vision.Decode()
    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=random_crop_op, input_columns=["image"])

    filename = "random_crop_01_c_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)

def test_random_crop_01_py():
    """
    Test RandomCrop op with py_transforms: size is a single integer, expected to pass
    """
    logger.info("test_random_crop_01_py")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # Note: If size is an int, a square crop of size (size, size) is returned.
    transforms = [
        py_vision.Decode(),
        py_vision.RandomCrop(512),
        py_vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])

    filename = "random_crop_01_py_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)

def test_random_crop_02_c():
    """
    Test RandomCrop op with c_transforms: size is a list/tuple with length 2, expected to pass
    """
    logger.info("test_random_crop_02_c")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # Note: If size is a sequence of length 2, it should be (height, width).
    random_crop_op = c_vision.RandomCrop([512, 375])
    decode_op = c_vision.Decode()
    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=random_crop_op, input_columns=["image"])

    filename = "random_crop_02_c_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)

def test_random_crop_02_py():
    """
    Test RandomCrop op with py_transforms: size is a list/tuple with length 2, expected to pass
    """
    logger.info("test_random_crop_02_py")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # Note: If size is a sequence of length 2, it should be (height, width).
    transforms = [
        py_vision.Decode(),
        py_vision.RandomCrop([512, 375]),
        py_vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])

    filename = "random_crop_02_py_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)

def test_random_crop_03_c():
    """
    Test RandomCrop op with c_transforms: input image size == crop size, expected to pass
    """
    logger.info("test_random_crop_03_c")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # Note: The size of the image is 4032*2268
    random_crop_op = c_vision.RandomCrop([2268, 4032])
    decode_op = c_vision.Decode()
    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=random_crop_op, input_columns=["image"])

    filename = "random_crop_03_c_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)

def test_random_crop_03_py():
    """
    Test RandomCrop op with py_transforms: input image size == crop size, expected to pass
    """
    logger.info("test_random_crop_03_py")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # Note: The size of the image is 4032*2268
    transforms = [
        py_vision.Decode(),
        py_vision.RandomCrop([2268, 4032]),
        py_vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])

    filename = "random_crop_03_py_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)

def test_random_crop_04_c():
    """
    Test RandomCrop op with c_transforms: input image size < crop size, expected to fail
    """
    logger.info("test_random_crop_04_c")

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # Note: The size of the image is 4032*2268
    random_crop_op = c_vision.RandomCrop([2268, 4033])
    decode_op = c_vision.Decode()
    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=random_crop_op, input_columns=["image"])
    try:
        data.create_dict_iterator(num_epochs=1).__next__()
    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "crop size is bigger than the image dimensions" in str(e)

def test_random_crop_04_py():
    """
    Test RandomCrop op with py_transforms:
    input image size < crop size, expected to fail
    """
    logger.info("test_random_crop_04_py")

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # Note: The size of the image is 4032*2268
    transforms = [
        py_vision.Decode(),
        py_vision.RandomCrop([2268, 4033]),
        py_vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])
    try:
        data.create_dict_iterator(num_epochs=1).__next__()
    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Crop size" in str(e)

def test_random_crop_05_c():
    """
    Test RandomCrop op with c_transforms:
    input image size < crop size but pad_if_needed is enabled,
    expected to pass
    """
    logger.info("test_random_crop_05_c")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # Note: The size of the image is 4032*2268
    random_crop_op = c_vision.RandomCrop([2268, 4033], [200, 200, 200, 200], pad_if_needed=True)
    decode_op = c_vision.Decode()
    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=random_crop_op, input_columns=["image"])

    filename = "random_crop_05_c_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)

def test_random_crop_05_py():
    """
    Test RandomCrop op with py_transforms:
    input image size < crop size but pad_if_needed is enabled,
    expected to pass
    """
    logger.info("test_random_crop_05_py")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # Note: The size of the image is 4032*2268
    transforms = [
        py_vision.Decode(),
        py_vision.RandomCrop([2268, 4033], [200, 200, 200, 200], pad_if_needed=True),
        py_vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])

    filename = "random_crop_05_py_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)

def test_random_crop_06_c():
    """
    Test RandomCrop op with c_transforms:
    invalid size, expected to raise TypeError
    """
    logger.info("test_random_crop_06_c")

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    try:
        # Note: if size is neither an int nor a list of length 2, an exception will raise
        random_crop_op = c_vision.RandomCrop([512, 512, 375])
        decode_op = c_vision.Decode()
        data = data.map(operations=decode_op, input_columns=["image"])
        data = data.map(operations=random_crop_op, input_columns=["image"])
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Size should be a single integer" in str(e)

def test_random_crop_06_py():
    """
    Test RandomCrop op with py_transforms:
    invalid size, expected to raise TypeError
    """
    logger.info("test_random_crop_06_py")

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    try:
        # Note: if size is neither an int nor a list of length 2, an exception will raise
        transforms = [
            py_vision.Decode(),
            py_vision.RandomCrop([512, 512, 375]),
            py_vision.ToTensor()
        ]
        transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
        data = data.map(operations=transform, input_columns=["image"])
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Size should be a single integer" in str(e)

def test_random_crop_07_c():
    """
    Test RandomCrop op with c_transforms:
    padding_mode is Border.CONSTANT and fill_value is 255 (White),
    expected to pass
    """
    logger.info("test_random_crop_07_c")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # Note: The padding_mode is default as Border.CONSTANT and set filling color to be white.
    random_crop_op = c_vision.RandomCrop(512, [200, 200, 200, 200], fill_value=(255, 255, 255))
    decode_op = c_vision.Decode()
    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=random_crop_op, input_columns=["image"])

    filename = "random_crop_07_c_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)

def test_random_crop_07_py():
    """
    Test RandomCrop op with py_transforms:
    padding_mode is Border.CONSTANT and fill_value is 255 (White),
    expected to pass
    """
    logger.info("test_random_crop_07_py")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # Note: The padding_mode is default as Border.CONSTANT and set filling color to be white.
    transforms = [
        py_vision.Decode(),
        py_vision.RandomCrop(512, [200, 200, 200, 200], fill_value=(255, 255, 255)),
        py_vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])

    filename = "random_crop_07_py_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)

def test_random_crop_08_c():
    """
    Test RandomCrop op with c_transforms: padding_mode is Border.EDGE,
    expected to pass
    """
    logger.info("test_random_crop_08_c")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # Note: The padding_mode is Border.EDGE.
    random_crop_op = c_vision.RandomCrop(512, [200, 200, 200, 200], padding_mode=mode.Border.EDGE)
    decode_op = c_vision.Decode()
    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=random_crop_op, input_columns=["image"])

    filename = "random_crop_08_c_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)

def test_random_crop_08_py():
    """
    Test RandomCrop op with py_transforms: padding_mode is Border.EDGE,
    expected to pass
    """
    logger.info("test_random_crop_08_py")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # Note: The padding_mode is Border.EDGE.
    transforms = [
        py_vision.Decode(),
        py_vision.RandomCrop(512, [200, 200, 200, 200], padding_mode=mode.Border.EDGE),
        py_vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])

    filename = "random_crop_08_py_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)

def test_random_crop_09():
    """
    Test RandomCrop op: invalid type of input image (not PIL), expected to raise TypeError
    """
    logger.info("test_random_crop_09")

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        py_vision.Decode(),
        py_vision.ToTensor(),
        # Note: if input is not PIL image, TypeError will raise
        py_vision.RandomCrop(512)
    ]
    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])
    try:
        data.create_dict_iterator(num_epochs=1).__next__()
    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "should be PIL image" in str(e)

def test_random_crop_comp(plot=False):
    """
    Test RandomCrop and compare between python and c image augmentation
    """
    logger.info("Test RandomCrop with c_transform and py_transform comparison")
    cropped_size = 512

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    random_crop_op = c_vision.RandomCrop(cropped_size)
    decode_op = c_vision.Decode()
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=random_crop_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        py_vision.Decode(),
        py_vision.RandomCrop(cropped_size),
        py_vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
    data2 = data2.map(operations=transform, input_columns=["image"])

    image_c_cropped = []
    image_py_cropped = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        c_image = item1["image"]
        py_image = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_c_cropped.append(c_image)
        image_py_cropped.append(py_image)
    if plot:
        visualize_list(image_c_cropped, image_py_cropped, visualize_mode=2)


if __name__ == "__main__":
    test_random_crop_01_c()
    test_random_crop_02_c()
    test_random_crop_03_c()
    test_random_crop_04_c()
    test_random_crop_05_c()
    test_random_crop_06_c()
    test_random_crop_07_c()
    test_random_crop_08_c()
    test_random_crop_01_py()
    test_random_crop_02_py()
    test_random_crop_03_py()
    test_random_crop_04_py()
    test_random_crop_05_py()
    test_random_crop_06_py()
    test_random_crop_07_py()
    test_random_crop_08_py()
    test_random_crop_09()
    test_random_crop_op_c(True)
    test_random_crop_op_py(True)
    test_random_crop_comp(True)
