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
Testing the random horizontal flip op in DE
"""
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.transforms.py_transforms
import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore import log as logger
from util import save_and_check_md5, visualize_list, visualize_image, diff_mse, \
    config_get_set_seed, config_get_set_num_parallel_workers

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def h_flip(image):
    """
    Apply the random_horizontal
    """

    # with the seed provided in this test case, it will always flip.
    # that's why we flip here too
    image = image[:, ::-1, :]
    return image


def test_random_horizontal_op(plot=False):
    """
    Test RandomHorizontalFlip op
    """
    logger.info("test_random_horizontal_op")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = c_vision.Decode()
    random_horizontal_op = c_vision.RandomHorizontalFlip(1.0)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=random_horizontal_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=decode_op, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):

        # with the seed value, we can only guarantee the first number generated
        if num_iter > 0:
            break

        image_h_flipped = item1["image"]
        image = item2["image"]
        image_h_flipped_2 = h_flip(image)

        mse = diff_mse(image_h_flipped, image_h_flipped_2)
        assert mse == 0
        logger.info("image_{}, mse: {}".format(num_iter + 1, mse))
        num_iter += 1
        if plot:
            visualize_image(image, image_h_flipped, mse, image_h_flipped_2)


def test_random_horizontal_valid_prob_c():
    """
    Test RandomHorizontalFlip op with c_transforms: valid non-default input, expect to pass
    """
    logger.info("test_random_horizontal_valid_prob_c")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = c_vision.Decode()
    random_horizontal_op = c_vision.RandomHorizontalFlip(0.8)
    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=random_horizontal_op, input_columns=["image"])

    filename = "random_horizontal_01_c_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_horizontal_valid_prob_py():
    """
    Test RandomHorizontalFlip op with py_transforms: valid non-default input, expect to pass
    """
    logger.info("test_random_horizontal_valid_prob_py")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        py_vision.Decode(),
        py_vision.RandomHorizontalFlip(0.8),
        py_vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])

    filename = "random_horizontal_01_py_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_horizontal_invalid_prob_c():
    """
    Test RandomHorizontalFlip op in c_transforms: invalid input, expect to raise error
    """
    logger.info("test_random_horizontal_invalid_prob_c")

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = c_vision.Decode()
    try:
        # Note: Valid range of prob should be [0.0, 1.0]
        random_horizontal_op = c_vision.RandomHorizontalFlip(1.5)
        data = data.map(operations=decode_op, input_columns=["image"])
        data = data.map(operations=random_horizontal_op, input_columns=["image"])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input prob is not within the required interval of [0.0, 1.0]." in str(e)


def test_random_horizontal_invalid_prob_py():
    """
    Test RandomHorizontalFlip op in py_transforms: invalid input, expect to raise error
    """
    logger.info("test_random_horizontal_invalid_prob_py")

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

    try:
        transforms = [
            py_vision.Decode(),
            # Note: Valid range of prob should be [0.0, 1.0]
            py_vision.RandomHorizontalFlip(1.5),
            py_vision.ToTensor()
        ]
        transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
        data = data.map(operations=transform, input_columns=["image"])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input prob is not within the required interval of [0.0, 1.0]." in str(e)


def test_random_horizontal_comp(plot=False):
    """
    Test test_random_horizontal_flip and compare between python and c image augmentation ops
    """
    logger.info("test_random_horizontal_comp")
    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = c_vision.Decode()
    # Note: The image must be flipped if prob is set to be 1
    random_horizontal_op = c_vision.RandomHorizontalFlip(1)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=random_horizontal_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        py_vision.Decode(),
        # Note: The image must be flipped if prob is set to be 1
        py_vision.RandomHorizontalFlip(1),
        py_vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
    data2 = data2.map(operations=transform, input_columns=["image"])

    images_list_c = []
    images_list_py = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_c = item1["image"]
        image_py = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        images_list_c.append(image_c)
        images_list_py.append(image_py)

        # Check if the output images are the same
        mse = diff_mse(image_c, image_py)
        assert mse < 0.001
    if plot:
        visualize_list(images_list_c, images_list_py, visualize_mode=2)


if __name__ == "__main__":
    test_random_horizontal_op(plot=True)
    test_random_horizontal_valid_prob_c()
    test_random_horizontal_valid_prob_py()
    test_random_horizontal_invalid_prob_c()
    test_random_horizontal_invalid_prob_py()
    test_random_horizontal_comp(plot=True)
