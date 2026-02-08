# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
Testing RandomOrder op in DE
"""
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.transforms as data_trans
from mindspore.dataset import vision
from mindspore import log as logger
from util import visualize_list, config_get_set_seed, \
    config_get_set_num_parallel_workers, save_and_check_md5_pil

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_random_order_op(plot=False):
    """
    Feature: RandomOrder op
    Description: Test RandomOrder in python transformations
    Expectation: Pipelines execute successfully
    """
    logger.info("test_random_order_op")
    # define map operations
    transforms_list = [vision.CenterCrop(64), vision.RandomRotation(30)]
    transforms1 = [
        vision.Decode(True),
        data_trans.RandomOrder(transforms_list),
        vision.ToTensor()
    ]
    transform1 = data_trans.Compose(transforms1)

    transforms2 = [
        vision.Decode(True),
        vision.ToTensor()
    ]
    transform2 = data_trans.Compose(transforms2)

    #  First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=transform1, input_columns=["image"])
    #  Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=transform2, input_columns=["image"])

    image_order = []
    image_original = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image2 = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_order.append(image1)
        image_original.append(image2)
    if plot:
        visualize_list(image_original, image_order)


def test_random_order_md5():
    """
    Feature: RandomOrder op
    Description: Test RandomOrder op with md5 check
    Expectation: Pipeline results match in md5 comparison
    """
    logger.info("test_random_order_md5")
    original_seed = config_get_set_seed(8)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    # define map operations
    transforms_list = [vision.RandomCrop(64), vision.RandomRotation(30)]
    transforms = [
        vision.Decode(True),
        data_trans.RandomOrder(transforms_list),
        vision.ToTensor()
    ]
    transform = data_trans.Compose(transforms)

    #  Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data = data.map(operations=transform, input_columns=["image"])

    # check results with md5 comparison
    filename = "random_order_01_result.npz"
    save_and_check_md5_pil(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers((original_num_parallel_workers))


if __name__ == '__main__':
    test_random_order_op(plot=True)
    test_random_order_md5()
