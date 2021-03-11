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
"""
Testing RandomPerspective op in DE
"""
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.transforms.py_transforms
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.dataset.vision.utils import Inter
from mindspore import log as logger
from util import visualize_list, save_and_check_md5, \
    config_get_set_seed, config_get_set_num_parallel_workers

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_random_perspective_op(plot=False):
    """
    Test RandomPerspective in python transformations
    """
    logger.info("test_random_perspective_op")
    # define map operations
    transforms1 = [
        py_vision.Decode(),
        py_vision.RandomPerspective(),
        py_vision.ToTensor()
    ]
    transform1 = mindspore.dataset.transforms.py_transforms.Compose(transforms1)

    transforms2 = [
        py_vision.Decode(),
        py_vision.ToTensor()
    ]
    transform2 = mindspore.dataset.transforms.py_transforms.Compose(transforms2)

    #  First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=transform1, input_columns=["image"])
    #  Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=transform2, input_columns=["image"])

    image_perspective = []
    image_original = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image2 = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_perspective.append(image1)
        image_original.append(image2)
    if plot:
        visualize_list(image_original, image_perspective)


def skip_test_random_perspective_md5():
    """
    Test RandomPerspective with md5 comparison
    """
    logger.info("test_random_perspective_md5")
    original_seed = config_get_set_seed(5)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # define map operations
    transforms = [
        py_vision.Decode(),
        py_vision.RandomPerspective(distortion_scale=0.3, prob=0.7,
                                    interpolation=Inter.BILINEAR),
        py_vision.Resize(1450),  # resize to a smaller size to prevent round-off error
        py_vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)

    #  Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data = data.map(operations=transform, input_columns=["image"])

    # check results with md5 comparison
    filename = "random_perspective_01_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers((original_num_parallel_workers))


def test_random_perspective_exception_distortion_scale_range():
    """
    Test RandomPerspective: distortion_scale is not in [0, 1], expected to raise ValueError
    """
    logger.info("test_random_perspective_exception_distortion_scale_range")
    try:
        _ = py_vision.RandomPerspective(distortion_scale=1.5)
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input distortion_scale is not within the required interval of [0.0, 1.0]."


def test_random_perspective_exception_prob_range():
    """
    Test RandomPerspective: prob is not in [0, 1], expected to raise ValueError
    """
    logger.info("test_random_perspective_exception_prob_range")
    try:
        _ = py_vision.RandomPerspective(prob=1.2)
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input prob is not within the required interval of [0.0, 1.0]."


if __name__ == "__main__":
    test_random_perspective_op(plot=True)
    skip_test_random_perspective_md5()
    test_random_perspective_exception_distortion_scale_range()
    test_random_perspective_exception_prob_range()
