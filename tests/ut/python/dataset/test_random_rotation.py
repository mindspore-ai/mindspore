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
Testing RandomRotation op in DE
"""
import numpy as np
import cv2

import mindspore.dataset as ds
import mindspore.dataset.transforms.py_transforms
import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.dataset.vision.utils import Inter
from mindspore import log as logger
from util import visualize_image, visualize_list, diff_mse, save_and_check_md5, \
    config_get_set_seed, config_get_set_num_parallel_workers

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

GENERATE_GOLDEN = False


def test_random_rotation_op_c(plot=False):
    """
    Test RandomRotation in c++ transformations op
    """
    logger.info("test_random_rotation_op_c")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    decode_op = c_vision.Decode()
    # use [90, 90] to force rotate 90 degrees, expand is set to be True to match output size
    random_rotation_op = c_vision.RandomRotation((90, 90), expand=True)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=random_rotation_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=decode_op, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        if num_iter > 0:
            break
        rotation_de = item1["image"]
        original = item2["image"]
        logger.info("shape before rotate: {}".format(original.shape))
        rotation_cv = cv2.rotate(original, cv2.ROTATE_90_COUNTERCLOCKWISE)
        mse = diff_mse(rotation_de, rotation_cv)
        logger.info("random_rotation_op_{}, mse: {}".format(num_iter + 1, mse))
        assert mse == 0
        num_iter += 1
        if plot:
            visualize_image(original, rotation_de, mse, rotation_cv)


def test_random_rotation_op_py(plot=False):
    """
    Test RandomRotation in python transformations op
    """
    logger.info("test_random_rotation_op_py")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    # use [90, 90] to force rotate 90 degrees, expand is set to be True to match output size
    transform1 = mindspore.dataset.transforms.py_transforms.Compose([py_vision.Decode(),
                                                                     py_vision.RandomRotation((90, 90), expand=True),
                                                                     py_vision.ToTensor()])
    data1 = data1.map(operations=transform1, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transform2 = mindspore.dataset.transforms.py_transforms.Compose([py_vision.Decode(),
                                                                     py_vision.ToTensor()])
    data2 = data2.map(operations=transform2, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        if num_iter > 0:
            break
        rotation_de = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        original = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        logger.info("shape before rotate: {}".format(original.shape))
        rotation_cv = cv2.rotate(original, cv2.ROTATE_90_COUNTERCLOCKWISE)
        mse = diff_mse(rotation_de, rotation_cv)
        logger.info("random_rotation_op_{}, mse: {}".format(num_iter + 1, mse))
        assert mse == 0
        num_iter += 1
        if plot:
            visualize_image(original, rotation_de, mse, rotation_cv)

def test_random_rotation_op_py_ANTIALIAS():
    """
    Test RandomRotation in python transformations op
    """
    logger.info("test_random_rotation_op_py_ANTIALIAS")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    # use [90, 90] to force rotate 90 degrees, expand is set to be True to match output size
    transform1 = mindspore.dataset.transforms.py_transforms.Compose([py_vision.Decode(),
                                                                     py_vision.RandomRotation((90, 90),
                                                                                              expand=True,
                                                                                              resample=Inter.ANTIALIAS),
                                                                     py_vision.ToTensor()])
    data1 = data1.map(operations=transform1, input_columns=["image"])

    num_iter = 0
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1
    logger.info("use RandomRotation by Inter.ANTIALIAS process {} images.".format(num_iter))

def test_random_rotation_expand():
    """
    Test RandomRotation op
    """
    logger.info("test_random_rotation_op")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = c_vision.Decode()
    # expand is set to be True to match output size
    random_rotation_op = c_vision.RandomRotation((0, 90), expand=True)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=random_rotation_op, input_columns=["image"])

    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1):
        rotation = item["image"]
        logger.info("shape after rotate: {}".format(rotation.shape))
        num_iter += 1


def test_random_rotation_md5():
    """
    Test RandomRotation with md5 check
    """
    logger.info("Test RandomRotation with md5 check")
    original_seed = config_get_set_seed(5)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = c_vision.Decode()
    resize_op = c_vision.RandomRotation((0, 90),
                                        expand=True,
                                        resample=Inter.BILINEAR,
                                        center=(50, 50),
                                        fill_value=150)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=resize_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    transform2 = mindspore.dataset.transforms.py_transforms.Compose([py_vision.Decode(),
                                                                     py_vision.RandomRotation((0, 90),
                                                                                              expand=True,
                                                                                              resample=Inter.BILINEAR,
                                                                                              center=(50, 50),
                                                                                              fill_value=150),
                                                                     py_vision.ToTensor()])
    data2 = data2.map(operations=transform2, input_columns=["image"])

    # Compare with expected md5 from images
    filename1 = "random_rotation_01_c_result.npz"
    save_and_check_md5(data1, filename1, generate_golden=GENERATE_GOLDEN)
    filename2 = "random_rotation_01_py_result.npz"
    save_and_check_md5(data2, filename2, generate_golden=GENERATE_GOLDEN)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_rotation_diff(plot=False):
    """
    Test RandomRotation op
    """
    logger.info("test_random_rotation_op")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = c_vision.Decode()

    rotation_op = c_vision.RandomRotation((45, 45))
    ctrans = [decode_op,
              rotation_op
              ]

    data1 = data1.map(operations=ctrans, input_columns=["image"])

    # Second dataset
    transforms = [
        py_vision.Decode(),
        py_vision.RandomRotation((45, 45)),
        py_vision.ToTensor(),
    ]
    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=transform, input_columns=["image"])

    num_iter = 0
    image_list_c, image_list_py = [], []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        num_iter += 1
        c_image = item1["image"]
        py_image = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_list_c.append(c_image)
        image_list_py.append(py_image)

        logger.info("shape of c_image: {}".format(c_image.shape))
        logger.info("shape of py_image: {}".format(py_image.shape))

        logger.info("dtype of c_image: {}".format(c_image.dtype))
        logger.info("dtype of py_image: {}".format(py_image.dtype))

        mse = diff_mse(c_image, py_image)
        assert mse < 0.001  # Rounding error
    if plot:
        visualize_list(image_list_c, image_list_py, visualize_mode=2)


if __name__ == "__main__":
    test_random_rotation_op_c(plot=True)
    test_random_rotation_op_py(plot=True)
    test_random_rotation_op_py_ANTIALIAS()
    test_random_rotation_expand()
    test_random_rotation_md5()
    test_rotation_diff(plot=True)
