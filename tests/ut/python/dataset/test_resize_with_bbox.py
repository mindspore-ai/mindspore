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
import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.c_transforms as c_vision

from mindspore import log as logger
from util import visualize_with_bounding_boxes, InvalidBBoxType, check_bad_bbox, \
    save_and_check_md5

GENERATE_GOLDEN = False

DATA_DIR = "../data/dataset/testVOC2012_2"


def fix_annotate(bboxes):
    """
    Fix annotations to format followed by mindspore.
    :param bboxes: in [label, x_min, y_min, w, h, truncate, difficult] format
    :return: annotation in [x_min, y_min, w, h, label, truncate, difficult] format
    """
    for (i, box) in enumerate(bboxes):
        bboxes[i] = np.roll(box, -1)
    return bboxes


def test_resize_with_bbox_op_c(plot_vis=False):
    """
    Prints images and bboxes side by side with and without ResizeWithBBox Op applied,
    tests with MD5 check, expected to pass
    """
    logger.info("test_resize_with_bbox_op_c")

    # Load dataset
    dataVoc1 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train",
                             decode=True, shuffle=False)

    dataVoc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train",
                             decode=True, shuffle=False)

    test_op = c_vision.ResizeWithBBox(200)

    dataVoc1 = dataVoc1.map(input_columns=["annotation"],
                            output_columns=["annotation"],
                            operations=fix_annotate)
    dataVoc2 = dataVoc2.map(input_columns=["annotation"],
                            output_columns=["annotation"],
                            operations=fix_annotate)
    # map to apply ops
    dataVoc2 = dataVoc2.map(input_columns=["image", "annotation"],
                            output_columns=["image", "annotation"],
                            columns_order=["image", "annotation"],
                            operations=[test_op])

    filename = "resize_with_bbox_op_01_c_result.npz"
    save_and_check_md5(dataVoc2, filename, generate_golden=GENERATE_GOLDEN)

    unaugSamp, augSamp = [], []

    for unAug, Aug in zip(dataVoc1.create_dict_iterator(), dataVoc2.create_dict_iterator()):
        unaugSamp.append(unAug)
        augSamp.append(Aug)

    if plot_vis:
        visualize_with_bounding_boxes(unaugSamp, augSamp)


def test_resize_with_bbox_op_edge_c(plot_vis=False):
    """
    Prints images and bboxes side by side with and without ResizeWithBBox Op applied,
    applied on dynamically generated edge case, expected to pass. edge case is when bounding
    box has dimensions as the image itself.
    """
    logger.info("test_resize_with_bbox_op_edge_c")
    dataVoc1 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train",
                             decode=True, shuffle=False)

    dataVoc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train",
                             decode=True, shuffle=False)

    test_op = c_vision.ResizeWithBBox(500)

    dataVoc1 = dataVoc1.map(input_columns=["annotation"],
                            output_columns=["annotation"],
                            operations=fix_annotate)
    dataVoc2 = dataVoc2.map(input_columns=["annotation"],
                            output_columns=["annotation"],
                            operations=fix_annotate)

    # maps to convert data into valid edge case data
    dataVoc1 = dataVoc1.map(input_columns=["image", "annotation"],
                            output_columns=["image", "annotation"],
                            columns_order=["image", "annotation"],
                            operations=[lambda img, bboxes: (
                                img, np.array([[0, 0, img.shape[1], img.shape[0]]]).astype(bboxes.dtype))])

    # Test Op added to list of Operations here
    dataVoc2 = dataVoc2.map(input_columns=["image", "annotation"],
                            output_columns=["image", "annotation"],
                            columns_order=["image", "annotation"],
                            operations=[lambda img, bboxes: (
                                img, np.array([[0, 0, img.shape[1], img.shape[0]]]).astype(bboxes.dtype)), test_op])

    unaugSamp, augSamp = [], []

    for unAug, Aug in zip(dataVoc1.create_dict_iterator(), dataVoc2.create_dict_iterator()):
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

    except ValueError as err:
        logger.info("Got an exception in DE: {}".format(str(err)))
        assert "interpolation" in str(err)


def test_resize_with_bbox_op_bad_c():
    """
    Tests ResizeWithBBox Op with invalid bounding boxes, expected to catch multiple errors
    """
    logger.info("test_resize_with_bbox_op_bad_c")
    test_op = c_vision.ResizeWithBBox((200, 300))

    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    check_bad_bbox(data_voc2, test_op, InvalidBBoxType.WidthOverflow, "bounding boxes is out of bounds of the image")
    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    check_bad_bbox(data_voc2, test_op, InvalidBBoxType.HeightOverflow, "bounding boxes is out of bounds of the image")
    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    check_bad_bbox(data_voc2, test_op, InvalidBBoxType.NegativeXY, "min_x")
    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    check_bad_bbox(data_voc2, test_op, InvalidBBoxType.WrongShape, "4 features")


if __name__ == "__main__":
    test_resize_with_bbox_op_c(plot_vis=False)
    test_resize_with_bbox_op_edge_c(plot_vis=False)
    test_resize_with_bbox_op_invalid_c()
    test_resize_with_bbox_op_bad_c()
