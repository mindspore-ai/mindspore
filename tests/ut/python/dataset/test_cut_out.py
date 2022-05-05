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
Testing CutOut op in DE
"""
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.transforms.transforms
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger
from util import visualize_image, visualize_list, diff_mse, save_and_check_md5, \
    config_get_set_seed, config_get_set_num_parallel_workers

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

GENERATE_GOLDEN = False


def test_cut_out_op(plot=False):
    """
    Test CutOut
    """
    logger.info("test_cut_out")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

    transforms_1 = [
        vision.Decode(True),
        vision.ToTensor(),
        vision.RandomErasing(value='random')
    ]
    transform_1 = mindspore.dataset.transforms.transforms.Compose(transforms_1)
    data1 = data1.map(operations=transform_1, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    cut_out_op = vision.CutOut(80, is_hwc=True)

    transforms_2 = [
        decode_op,
        cut_out_op
    ]

    data2 = data2.map(operations=transforms_2, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        num_iter += 1
        image_1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        # C image doesn't require transpose
        image_2 = item2["image"]

        logger.info("shape of image_1: {}".format(image_1.shape))
        logger.info("shape of image_2: {}".format(image_2.shape))

        logger.info("dtype of image_1: {}".format(image_1.dtype))
        logger.info("dtype of image_2: {}".format(image_2.dtype))

        mse = diff_mse(image_1, image_2)
        if plot:
            visualize_image(image_1, image_2, mse)


def test_cut_out_op_multicut(plot=False):
    """
    Test CutOut
    """
    logger.info("test_cut_out")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

    transforms_1 = [
        vision.Decode(True),
        vision.ToTensor(),
    ]
    transform_1 = mindspore.dataset.transforms.transforms.Compose(transforms_1)
    data1 = data1.map(operations=transform_1, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    cut_out_op = vision.CutOut(80, num_patches=10, is_hwc=True)

    transforms_2 = [
        decode_op,
        cut_out_op
    ]

    data2 = data2.map(operations=transforms_2, input_columns=["image"])

    num_iter = 0
    image_list_1, image_list_2 = [], []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        num_iter += 1
        image_1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        # C image doesn't require transpose
        image_2 = item2["image"]
        image_list_1.append(image_1)
        image_list_2.append(image_2)

        logger.info("shape of image_1: {}".format(image_1.shape))
        logger.info("shape of image_2: {}".format(image_2.shape))

        logger.info("dtype of image_1: {}".format(image_1.dtype))
        logger.info("dtype of image_2: {}".format(image_2.dtype))
    if plot:
        visualize_list(image_list_1, image_list_2)


def test_cut_out_md5():
    """
    Test CutOut with md5 check
    """
    logger.info("test_cut_out_md5")
    original_seed = config_get_set_seed(2)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    cut_out_op = vision.CutOut(100, is_hwc=True)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=cut_out_op, input_columns=["image"])

    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        vision.Decode(True),
        vision.ToTensor(),
        vision.CutOut(100, is_hwc=False)
    ]
    transform = mindspore.dataset.transforms.transforms.Compose(transforms)
    data2 = data2.map(operations=transform, input_columns=["image"])

    # Compare with expected md5 from images
    filename1 = "cut_out_01_c_result.npz"
    save_and_check_md5(data1, filename1, generate_golden=GENERATE_GOLDEN)
    filename2 = "cut_out_02_c_result.npz"
    save_and_check_md5(data2, filename2, generate_golden=GENERATE_GOLDEN)

    # Restore config
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_cut_out_comp_hwc(plot=False):
    """
    Feature: CutOut op
    Description: Test CutOut with HWC input, Decode(to_pil=True) & ToTensor versus Decode(to_pil=False) comparison
    Expectation: Test succeeds. Manual confirmation of logged info. Manual visualization confirmation.
    """
    logger.info("test_cut_out_comp")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

    transforms_1 = [
        vision.Decode(True),
        vision.ToTensor(),
        vision.CutOut(250, is_hwc=False)
    ]
    transform_1 = mindspore.dataset.transforms.transforms.Compose(transforms_1)
    data1 = data1.map(operations=transform_1, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

    transforms_2 = [
        vision.Decode(),
        vision.CutOut(250, is_hwc=True)
    ]

    data2 = data2.map(operations=transforms_2, input_columns=["image"])

    num_iter = 0
    image_list_1, image_list_2 = [], []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        num_iter += 1
        image_1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        # C image doesn't require transpose
        image_2 = item2["image"]
        image_list_1.append(image_1)
        image_list_2.append(image_2)

        logger.info("shape of image_1: {}".format(image_1.shape))
        logger.info("shape of image_2: {}".format(image_2.shape))

        logger.info("dtype of image_1: {}".format(image_1.dtype))
        logger.info("dtype of image_2: {}".format(image_2.dtype))
    if plot:
        visualize_list(image_list_1, image_list_2, visualize_mode=2)


def skip_test_cut_out_comp_chw():
    """
    Feature: CutOut op
    Description: Test CutOut with CHW input, Decode(to_pil=True) & ToTensor versus Decode(to_pil=False) & HWC2CHW
                 comparison.
    Expectation: Test succeeds.  Manual confirmation of logged info.
    """
    logger.info("test_cut_out_comp_chw")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

    transforms_1 = [
        vision.Decode(),
        vision.HWC2CHW(),
        vision.CutOut(200, is_hwc=False)
    ]
    transform_1 = mindspore.dataset.transforms.transforms.Compose(transforms_1)
    data1 = data1.map(operations=transform_1, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

    transforms_2 = [
        vision.Decode(True),
        vision.ToTensor(),
        vision.CutOut(200, is_hwc=False)
    ]

    data2 = data2.map(operations=transforms_2, input_columns=["image"])

    num_iter = 0
    image_list_1, image_list_2 = [], []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        num_iter += 1
        image_1 = item1["image"]
        image_2 = item2["image"]
        if image_1.shape != image_2.shape:
            raise RuntimeError("image_1.shape != image_2.shape: " + str(image_1.shape) + " " + str(image_2.shape))
        image_list_1.append(image_1)
        image_list_2.append(image_2)

        logger.info("shape of image_1: {}".format(image_1.shape))
        logger.info("shape of image_2: {}".format(image_2.shape))

        logger.info("dtype of image_1: {}".format(image_1.dtype))
        logger.info("dtype of image_2: {}".format(image_2.dtype))


if __name__ == "__main__":
    test_cut_out_op(plot=True)
    test_cut_out_op_multicut(plot=True)
    test_cut_out_md5()
    test_cut_out_comp_hwc(plot=True)
    skip_test_cut_out_comp_chw()
