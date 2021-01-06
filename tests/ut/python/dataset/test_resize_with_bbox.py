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
Testing the resize with bounding boxes op in DE
"""
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as c_vision
from mindspore import log as logger
from util import visualize_with_bounding_boxes, InvalidBBoxType, check_bad_bbox, \
    save_and_check_md5

GENERATE_GOLDEN = False

DATA_DIR = "../data/dataset/testVOC2012_2"
DATA_DIR_2 = ["../data/dataset/testCOCO/train/",
              "../data/dataset/testCOCO/annotations/train.json"]  # DATA_DIR, ANNOTATION_DIR


def test_resize_with_bbox_op_voc_c(plot_vis=False):
    """
    Prints images and bboxes side by side with and without ResizeWithBBox Op applied
    testing with VOC dataset
    """
    logger.info("test_resize_with_bbox_op_voc_c")

    # Load dataset
    dataVoc1 = ds.VOCDataset(DATA_DIR, task="Detection", usage="train", shuffle=False, decode=True)

    dataVoc2 = ds.VOCDataset(DATA_DIR, task="Detection", usage="train", shuffle=False, decode=True)

    test_op = c_vision.ResizeWithBBox(100)

    # map to apply ops
    dataVoc2 = dataVoc2.map(operations=[test_op], input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            column_order=["image", "bbox"])

    filename = "resize_with_bbox_op_01_c_voc_result.npz"
    save_and_check_md5(dataVoc2, filename, generate_golden=GENERATE_GOLDEN)

    unaugSamp, augSamp = [], []

    for unAug, Aug in zip(dataVoc1.create_dict_iterator(num_epochs=1, output_numpy=True),
                          dataVoc2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        unaugSamp.append(unAug)
        augSamp.append(Aug)

    if plot_vis:
        visualize_with_bounding_boxes(unaugSamp, augSamp)


def test_resize_with_bbox_op_coco_c(plot_vis=False):
    """
    Prints images and bboxes side by side with and without ResizeWithBBox Op applied,
    tests with MD5 check, expected to pass
    Testing with COCO dataset
    """
    logger.info("test_resize_with_bbox_op_coco_c")

    # Load dataset
    dataCOCO1 = ds.CocoDataset(DATA_DIR_2[0], annotation_file=DATA_DIR_2[1], task="Detection",
                               decode=True, shuffle=False)

    dataCOCO2 = ds.CocoDataset(DATA_DIR_2[0], annotation_file=DATA_DIR_2[1], task="Detection",
                               decode=True, shuffle=False)

    test_op = c_vision.ResizeWithBBox(200)

    # map to apply ops

    dataCOCO2 = dataCOCO2.map(operations=[test_op], input_columns=["image", "bbox"],
                              output_columns=["image", "bbox"],
                              column_order=["image", "bbox"])

    filename = "resize_with_bbox_op_01_c_coco_result.npz"
    save_and_check_md5(dataCOCO2, filename, generate_golden=GENERATE_GOLDEN)

    unaugSamp, augSamp = [], []

    for unAug, Aug in zip(dataCOCO1.create_dict_iterator(num_epochs=1, output_numpy=True),
                          dataCOCO2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        unaugSamp.append(unAug)
        augSamp.append(Aug)

    if plot_vis:
        visualize_with_bounding_boxes(unaugSamp, augSamp, annot_name="bbox")


def test_resize_with_bbox_op_edge_c(plot_vis=False):
    """
    Prints images and bboxes side by side with and without ResizeWithBBox Op applied,
    applied on dynamically generated edge case, expected to pass. edge case is when bounding
    box has dimensions as the image itself.
    """
    logger.info("test_resize_with_bbox_op_edge_c")
    dataVoc1 = ds.VOCDataset(DATA_DIR, task="Detection", usage="train", shuffle=False, decode=True)

    dataVoc2 = ds.VOCDataset(DATA_DIR, task="Detection", usage="train", shuffle=False, decode=True)

    test_op = c_vision.ResizeWithBBox(500)

    # maps to convert data into valid edge case data
    dataVoc1 = dataVoc1.map(
        operations=[lambda img, bboxes: (img, np.array([[0, 0, img.shape[1], img.shape[0]]]).astype(bboxes.dtype))],
        input_columns=["image", "bbox"],
        output_columns=["image", "bbox"],
        column_order=["image", "bbox"])

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


def test_resize_with_bbox_op_invalid_c():
    """
    Test ResizeWithBBox Op on invalid constructor parameters, expected to raise ValueError
    """
    logger.info("test_resize_with_bbox_op_invalid_c")

    try:
        # invalid interpolation value
        c_vision.ResizeWithBBox(400, interpolation="invalid")

    except TypeError as err:
        logger.info("Got an exception in DE: {}".format(str(err)))
        assert "interpolation" in str(err)


def test_resize_with_bbox_op_bad_c():
    """
    Tests ResizeWithBBox Op with invalid bounding boxes, expected to catch multiple errors
    """
    logger.info("test_resize_with_bbox_op_bad_c")
    test_op = c_vision.ResizeWithBBox((200, 300))

    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection", usage="train", shuffle=False, decode=True)
    check_bad_bbox(data_voc2, test_op, InvalidBBoxType.WidthOverflow, "bounding boxes is out of bounds of the image")
    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection", usage="train", shuffle=False, decode=True)
    check_bad_bbox(data_voc2, test_op, InvalidBBoxType.HeightOverflow, "bounding boxes is out of bounds of the image")
    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection", usage="train", shuffle=False, decode=True)
    check_bad_bbox(data_voc2, test_op, InvalidBBoxType.NegativeXY, "negative value")
    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection", usage="train", shuffle=False, decode=True)
    check_bad_bbox(data_voc2, test_op, InvalidBBoxType.WrongShape, "4 features")


def test_resize_with_bbox_op_params_outside_of_interpolation_dict():
    """
    Test passing in an invalid key for interpolation
    """
    logger.info("test_resize_with_bbox_op_params_outside_of_interpolation_dict")

    size = (500, 500)
    more_para = None
    with pytest.raises(KeyError, match="None"):
        c_vision.ResizeWithBBox(size, more_para)


if __name__ == "__main__":
    test_resize_with_bbox_op_voc_c(plot_vis=False)
    test_resize_with_bbox_op_coco_c(plot_vis=False)
    test_resize_with_bbox_op_edge_c(plot_vis=False)
    test_resize_with_bbox_op_invalid_c()
    test_resize_with_bbox_op_bad_c()
