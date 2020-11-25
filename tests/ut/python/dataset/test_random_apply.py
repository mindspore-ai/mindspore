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
Testing RandomApply op in DE
"""
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.transforms.py_transforms as py_transforms
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore import log as logger
from util import visualize_list, config_get_set_seed, \
    config_get_set_num_parallel_workers, save_and_check_md5

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_random_apply_op(plot=False):
    """
    Test RandomApply in python transformations
    """
    logger.info("test_random_apply_op")
    # define map operations
    transforms_list = [py_vision.CenterCrop(64), py_vision.RandomRotation(30)]
    transforms1 = [
        py_vision.Decode(),
        py_transforms.RandomApply(transforms_list, prob=0.6),
        py_vision.ToTensor()
    ]
    transform1 = py_transforms.Compose(transforms1)

    transforms2 = [
        py_vision.Decode(),
        py_vision.ToTensor()
    ]
    transform2 = py_transforms.Compose(transforms2)

    #  First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=transform1, input_columns=["image"])
    #  Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=transform2, input_columns=["image"])

    image_apply = []
    image_original = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image2 = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_apply.append(image1)
        image_original.append(image2)
    if plot:
        visualize_list(image_original, image_apply)


def test_random_apply_md5():
    """
    Test RandomApply op with md5 check
    """
    logger.info("test_random_apply_md5")
    original_seed = config_get_set_seed(10)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    # define map operations
    transforms_list = [py_vision.CenterCrop(64), py_vision.RandomRotation(30)]
    transforms = [
        py_vision.Decode(),
        # Note: using default value "prob=0.5"
        py_transforms.RandomApply(transforms_list),
        py_vision.ToTensor()
    ]
    transform = py_transforms.Compose(transforms)

    #  Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data = data.map(operations=transform, input_columns=["image"])

    # check results with md5 comparison
    filename = "random_apply_01_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers((original_num_parallel_workers))


def test_random_apply_exception_random_crop_badinput():
    """
    Test RandomApply: test invalid input for one of the transform functions,
    expected to raise error
    """
    logger.info("test_random_apply_exception_random_crop_badinput")
    original_seed = config_get_set_seed(200)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    # define map operations
    transforms_list = [py_vision.Resize([32, 32]),
                       py_vision.RandomCrop(100),  # crop size > image size
                       py_vision.RandomRotation(30)]
    transforms = [
        py_vision.Decode(),
        py_transforms.RandomApply(transforms_list, prob=0.6),
        py_vision.ToTensor()
    ]
    transform = py_transforms.Compose(transforms)
    #  Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data = data.map(operations=transform, input_columns=["image"])
    try:
        _ = data.create_dict_iterator(num_epochs=1).__next__()
    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Crop size" in str(e)
    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


if __name__ == '__main__':
    test_random_apply_op(plot=True)
    test_random_apply_md5()
    test_random_apply_exception_random_crop_badinput()
