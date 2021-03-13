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
Testing RandomCropWithBBox op in DE
"""
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.vision.utils as mode

from mindspore import log as logger
from util import visualize_with_bounding_boxes, InvalidBBoxType, check_bad_bbox, \
    config_get_set_seed, config_get_set_num_parallel_workers, save_and_check_md5

GENERATE_GOLDEN = False

# Updated VOC dataset with correct annotations - DATA_DIR
DATA_DIR_VOC = "../data/dataset/testVOC2012_2"
# COCO dataset - DATA_DIR, ANNOTATION_DIR
DATA_DIR_COCO = ["../data/dataset/testCOCO/train/", "../data/dataset/testCOCO/annotations/train.json"]


def test_random_crop_with_bbox_op_c(plot_vis=False):
    """
    Prints images and bboxes side by side with and without RandomCropWithBBox Op applied
    """
    logger.info("test_random_crop_with_bbox_op_c")

    # Load dataset
    dataVoc1 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)
    dataVoc2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)

    # define test OP with values to match existing Op UT
    test_op = c_vision.RandomCropWithBBox([512, 512], [200, 200, 200, 200])

    # map to apply ops
    dataVoc2 = dataVoc2.map(operations=[test_op], input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            column_order=["image", "bbox"])  # Add column for "bbox"

    unaugSamp, augSamp = [], []

    for unAug, Aug in zip(dataVoc1.create_dict_iterator(num_epochs=1, output_numpy=True),
                          dataVoc2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        unaugSamp.append(unAug)
        augSamp.append(Aug)

    if plot_vis:
        visualize_with_bounding_boxes(unaugSamp, augSamp)


def test_random_crop_with_bbox_op_coco_c(plot_vis=False):
    """
    Prints images and bboxes side by side with and without RandomCropWithBBox Op applied,
    Testing with Coco dataset
    """
    logger.info("test_random_crop_with_bbox_op_coco_c")
    # load dataset
    dataCoco1 = ds.CocoDataset(DATA_DIR_COCO[0], annotation_file=DATA_DIR_COCO[1], task="Detection",
                               decode=True, shuffle=False)

    dataCoco2 = ds.CocoDataset(DATA_DIR_COCO[0], annotation_file=DATA_DIR_COCO[1], task="Detection",
                               decode=True, shuffle=False)

    test_op = c_vision.RandomCropWithBBox([512, 512], [200, 200, 200, 200])

    dataCoco2 = dataCoco2.map(operations=[test_op], input_columns=["image", "bbox"],
                              output_columns=["image", "bbox"],
                              column_order=["image", "bbox"])

    unaugSamp, augSamp = [], []

    for unAug, Aug in zip(dataCoco1.create_dict_iterator(num_epochs=1, output_numpy=True),
                          dataCoco2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        unaugSamp.append(unAug)
        augSamp.append(Aug)

    if plot_vis:
        visualize_with_bounding_boxes(unaugSamp, augSamp, "bbox")


def test_random_crop_with_bbox_op2_c(plot_vis=False):
    """
    Prints images and bboxes side by side with and without RandomCropWithBBox Op applied,
    with md5 check, expected to pass
    """
    logger.info("test_random_crop_with_bbox_op2_c")
    original_seed = config_get_set_seed(593447)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Load dataset
    dataVoc1 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)
    dataVoc2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)

    # define test OP with values to match existing Op unit - test
    test_op = c_vision.RandomCropWithBBox(512, [200, 200, 200, 200], fill_value=(255, 255, 255))

    # map to apply ops
    dataVoc2 = dataVoc2.map(operations=[test_op], input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            column_order=["image", "bbox"])

    filename = "random_crop_with_bbox_01_c_result.npz"
    save_and_check_md5(dataVoc2, filename, generate_golden=GENERATE_GOLDEN)

    unaugSamp, augSamp = [], []

    for unAug, Aug in zip(dataVoc1.create_dict_iterator(num_epochs=1, output_numpy=True),
                          dataVoc2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        unaugSamp.append(unAug)
        augSamp.append(Aug)

    if plot_vis:
        visualize_with_bounding_boxes(unaugSamp, augSamp)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_crop_with_bbox_op3_c(plot_vis=False):
    """
    Prints images and bboxes side by side with and without RandomCropWithBBox Op applied,
    with Padding Mode explicitly passed
    """
    logger.info("test_random_crop_with_bbox_op3_c")

    # Load dataset
    dataVoc1 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)
    dataVoc2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)

    # define test OP with values to match existing Op unit - test
    test_op = c_vision.RandomCropWithBBox(512, [200, 200, 200, 200], padding_mode=mode.Border.EDGE)

    # map to apply ops
    dataVoc2 = dataVoc2.map(operations=[test_op], input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            column_order=["image", "bbox"])

    unaugSamp, augSamp = [], []

    for unAug, Aug in zip(dataVoc1.create_dict_iterator(num_epochs=1, output_numpy=True),
                          dataVoc2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        unaugSamp.append(unAug)
        augSamp.append(Aug)

    if plot_vis:
        visualize_with_bounding_boxes(unaugSamp, augSamp)


def test_random_crop_with_bbox_op_edge_c(plot_vis=False):
    """
    Prints images and bboxes side by side with and without RandomCropWithBBox Op applied,
    applied on dynamically generated edge case, expected to pass
    """
    logger.info("test_random_crop_with_bbox_op_edge_c")

    # Load dataset
    dataVoc1 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)
    dataVoc2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)

    # define test OP with values to match existing Op unit - test
    test_op = c_vision.RandomCropWithBBox(512, [200, 200, 200, 200], padding_mode=mode.Border.EDGE)

    # maps to convert data into valid edge case data
    dataVoc1 = dataVoc1.map(
        operations=[lambda img, bboxes: (img, np.array([[0, 0, img.shape[1], img.shape[0]]]).astype(bboxes.dtype))],
        input_columns=["image", "bbox"],
        output_columns=["image", "bbox"],
        column_order=["image", "bbox"])

    # Test Op added to list of Operations here
    dataVoc2 = dataVoc2.map(
        operations=[lambda img, bboxes: (img, np.array([[0, 0, img.shape[1], img.shape[0]]]).astype(bboxes.dtype)),
                    test_op], input_columns=["image", "bbox"],
        output_columns=["image", "bbox"],
        column_order=["image", "bbox"])

    unaugSamp, augSamp = [], []

    for unAug, Aug in zip(dataVoc1.create_dict_iterator(num_epochs=1, output_numpy=True),
                          dataVoc2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        unaugSamp.append(unAug)
        augSamp.append(Aug)

    if plot_vis:
        visualize_with_bounding_boxes(unaugSamp, augSamp)


def test_random_crop_with_bbox_op_invalid_c():
    """
    Test RandomCropWithBBox Op on invalid constructor parameters, expected to raise ValueError
    """
    logger.info("test_random_crop_with_bbox_op_invalid_c")

    # Load dataset
    dataVoc2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)

    try:
        # define test OP with values to match existing Op unit - test
        test_op = c_vision.RandomCropWithBBox([512, 512, 375])

        # map to apply ops
        dataVoc2 = dataVoc2.map(operations=[test_op], input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                column_order=["image", "bbox"])  # Add column for "bbox"

        for _ in dataVoc2.create_dict_iterator(num_epochs=1):
            break
    except TypeError as err:
        logger.info("Got an exception in DE: {}".format(str(err)))
        assert "Size should be a single integer" in str(err)


def test_random_crop_with_bbox_op_bad_c():
    """
    Tests RandomCropWithBBox Op with invalid bounding boxes, expected to catch multiple errors.
    """
    logger.info("test_random_crop_with_bbox_op_bad_c")
    test_op = c_vision.RandomCropWithBBox([512, 512], [200, 200, 200, 200])

    data_voc2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)
    check_bad_bbox(data_voc2, test_op, InvalidBBoxType.WidthOverflow, "bounding boxes is out of bounds of the image")
    data_voc2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)
    check_bad_bbox(data_voc2, test_op, InvalidBBoxType.HeightOverflow, "bounding boxes is out of bounds of the image")
    data_voc2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)
    check_bad_bbox(data_voc2, test_op, InvalidBBoxType.NegativeXY, "negative value")
    data_voc2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)
    check_bad_bbox(data_voc2, test_op, InvalidBBoxType.WrongShape, "4 features")


def test_random_crop_with_bbox_op_bad_padding():
    """
    Test RandomCropWithBBox Op on invalid constructor parameters, expected to raise ValueError
    """
    logger.info("test_random_crop_with_bbox_op_invalid_c")

    dataVoc2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)

    try:
        test_op = c_vision.RandomCropWithBBox([512, 512], padding=-1)

        dataVoc2 = dataVoc2.map(operations=[test_op], input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                column_order=["image", "bbox"])

        for _ in dataVoc2.create_dict_iterator(num_epochs=1):
            break
    except ValueError as err:
        logger.info("Got an exception in DE: {}".format(str(err)))
        assert "Input padding is not within the required interval of [0, 2147483647]." in str(err)

    try:
        test_op = c_vision.RandomCropWithBBox([512, 512], padding=[16777216, 16777216, 16777216, 16777216])

        dataVoc2 = dataVoc2.map(operations=[test_op], input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                column_order=["image", "bbox"])

        for _ in dataVoc2.create_dict_iterator(num_epochs=1):
            break
    except RuntimeError as err:
        logger.info("Got an exception in DE: {}".format(str(err)))
        assert "padding size is three times bigger than the image size" in str(err)


if __name__ == "__main__":
    test_random_crop_with_bbox_op_c(plot_vis=True)
    test_random_crop_with_bbox_op_coco_c(plot_vis=True)
    test_random_crop_with_bbox_op2_c(plot_vis=True)
    test_random_crop_with_bbox_op3_c(plot_vis=True)
    test_random_crop_with_bbox_op_edge_c(plot_vis=True)
    test_random_crop_with_bbox_op_invalid_c()
    test_random_crop_with_bbox_op_bad_c()
    test_random_crop_with_bbox_op_bad_padding()
