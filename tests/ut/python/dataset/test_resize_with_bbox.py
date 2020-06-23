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
from enum import Enum
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mindspore.dataset.transforms.vision.c_transforms as c_vision
from mindspore import log as logger
import mindspore.dataset as ds

GENERATE_GOLDEN = False

DATA_DIR = "../data/dataset/testVOC2012"


def fix_annotate(bboxes):
    """
    :param bboxes: in [label, x_min, y_min, w, h, truncate, difficult] format
    :return: annotation in [x_min, y_min, w, h, label, truncate, difficult] format
    """
    for bbox in bboxes:
        tmp = bbox[0]
        bbox[0] = bbox[1]
        bbox[1] = bbox[2]
        bbox[2] = bbox[3]
        bbox[3] = bbox[4]
        bbox[4] = tmp
    return bboxes


class BoxType(Enum):
    """
    Defines box types for test cases
    """
    WidthOverflow = 1
    HeightOverflow = 2
    NegativeXY = 3
    OnEdge = 4
    WrongShape = 5


class AddBadAnnotation:  # pylint: disable=too-few-public-methods
    """
    Used to add erroneous bounding boxes to object detection pipelines.
    Usage:
    >>> # Adds a box that covers the whole image. Good for testing edge cases
    >>> de = de.map(input_columns=["image", "annotation"],
    >>>            output_columns=["image", "annotation"],
    >>>            operations=AddBadAnnotation(BoxType.OnEdge))
    """

    def __init__(self, box_type):
        self.box_type = box_type

    def __call__(self, img, bboxes):
        """
        Used to generate erroneous bounding box examples on given img.
        :param img: image where the bounding boxes are.
        :param bboxes: in [x_min, y_min, w, h, label, truncate, difficult] format
        :return: bboxes with bad examples added
        """
        height = img.shape[0]
        width = img.shape[1]
        if self.box_type == BoxType.WidthOverflow:
            # use box that overflows on width
            return img, np.array([[0, 0, width + 1, height - 1, 0, 0, 0]]).astype(np.uint32)

        if self.box_type == BoxType.HeightOverflow:
            # use box that overflows on height
            return img, np.array([[0, 0, width - 1, height + 1, 0, 0, 0]]).astype(np.uint32)

        if self.box_type == BoxType.NegativeXY:
            # use box with negative xy
            return img, np.array([[-10, -10, width - 1, height - 1, 0, 0, 0]]).astype(np.uint32)

        if self.box_type == BoxType.OnEdge:
            # use box that covers the whole image
            return img, np.array([[0, 0, width - 1, height - 1, 0, 0, 0]]).astype(np.uint32)

        if self.box_type == BoxType.WrongShape:
            # use box that covers the whole image
            return img, np.array([[0, 0, width - 1]]).astype(np.uint32)
        return img, bboxes


def check_bad_box(data, box_type, expected_error):
    try:
        test_op = c_vision.ResizeWithBBox(100)
        data = data.map(input_columns=["annotation"],
                        output_columns=["annotation"],
                        operations=fix_annotate)
        # map to use width overflow
        data = data.map(input_columns=["image", "annotation"],
                        output_columns=["image", "annotation"],
                        columns_order=["image", "annotation"],
                        operations=AddBadAnnotation(box_type))  # Add column for "annotation"
        # map to apply ops
        data = data.map(input_columns=["image", "annotation"],
                        output_columns=["image", "annotation"],
                        columns_order=["image", "annotation"],
                        operations=[test_op])  # Add column for "annotation"
        for _, _ in enumerate(data.create_dict_iterator()):
            break
    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert expected_error in str(e)


def add_bounding_boxes(axis, bboxes):
    """
    :param axis: axis to modify
    :param bboxes: bounding boxes to draw on the axis
    :return: None
    """
    for bbox in bboxes:
        rect = patches.Rectangle((bbox[0], bbox[1]),
                                 bbox[2], bbox[3],
                                 linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        axis.add_patch(rect)


def visualize(unaugmented_data, augment_data):
    for idx, (un_aug_item, aug_item) in enumerate(
            zip(unaugmented_data.create_dict_iterator(), augment_data.create_dict_iterator())):
        axis = plt.subplot(141)
        plt.imshow(un_aug_item["image"])
        add_bounding_boxes(axis, un_aug_item["annotation"])  # add Orig BBoxes
        plt.title("Original" + str(idx + 1))
        logger.info("Original ", str(idx + 1), " :", un_aug_item["annotation"])

        axis = plt.subplot(142)
        plt.imshow(aug_item["image"])
        add_bounding_boxes(axis, aug_item["annotation"])  # add AugBBoxes
        plt.title("Augmented" + str(idx + 1))
        logger.info("Augmented ", str(idx + 1), " ", aug_item["annotation"], "\n")
        plt.show()


def test_resize_with_bbox_op(plot=False):
    """
    Test resize_with_bbox_op
    """
    logger.info("Test resize with bbox")

    # original images
    data_original = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)

    # augmented images
    data_augmented = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)

    data_original = data_original.map(input_columns=["annotation"],
                                      output_columns=["annotation"],
                                      operations=fix_annotate)

    data_augmented = data_augmented.map(input_columns=["annotation"],
                                        output_columns=["annotation"],
                                        operations=fix_annotate)

    # define map operations
    test_op = c_vision.ResizeWithBBox(100)  # input value being the target size of resizeOp

    data_augmented = data_augmented.map(input_columns=["image", "annotation"],
                                        output_columns=["image", "annotation"],
                                        columns_order=["image", "annotation"], operations=[test_op])
    if plot:
        visualize(data_original, data_augmented)


def test_resize_with_bbox_invalid_bounds():
    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    check_bad_box(data_voc2, BoxType.WidthOverflow, "bounding boxes is out of bounds of the image")
    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    check_bad_box(data_voc2, BoxType.HeightOverflow, "bounding boxes is out of bounds of the image")
    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    check_bad_box(data_voc2, BoxType.NegativeXY, "min_x")
    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    check_bad_box(data_voc2, BoxType.WrongShape, "4 features")


def test_resize_with_bbox_invalid_size():
    """
        Test resize_with_bbox_op
        """
    logger.info("Test resize with bbox with invalid target size")

    # original images
    data = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)

    data = data.map(input_columns=["annotation"],
                    output_columns=["annotation"],
                    operations=fix_annotate)

    # negative target size as input
    try:
        test_op = c_vision.ResizeWithBBox(-10)

        # map to apply ops
        data = data.map(input_columns=["image", "annotation"],
                        output_columns=["image", "annotation"],
                        columns_order=["image", "annotation"],
                        operations=[test_op])  # Add column for "annotation"

        for _, _ in enumerate(data.create_dict_iterator()):
            break

    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input is not" in str(e)

    # zero target size as input
    try:
        test_op = c_vision.ResizeWithBBox(0)

        # map to apply ops
        data = data.map(input_columns=["image", "annotation"],
                        output_columns=["image", "annotation"],
                        columns_order=["image", "annotation"],
                        operations=[test_op])  # Add column for "annotation"

        for _, _ in enumerate(data.create_dict_iterator()):
            break

    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input is not" in str(e)

    # invalid input shape
    try:
        test_op = c_vision.ResizeWithBBox((10, 10, 10))

        # map to apply ops
        data = data.map(input_columns=["image", "annotation"],
                        output_columns=["image", "annotation"],
                        columns_order=["image", "annotation"],
                        operations=[test_op])  # Add column for "annotation"

        for _, _ in enumerate(data.create_dict_iterator()):
            break

    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Size should be" in str(e)


def test_resize_with_bbox_invalid_interpolation():
    """
       Test resize_with_bbox_op
       """
    logger.info("Test resize with bbox with invalid interpolation size")

    # original images
    data = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)

    data = data.map(input_columns=["annotation"],
                    output_columns=["annotation"],
                    operations=fix_annotate)

    # invalid interpolation
    try:
        test_op = c_vision.ResizeWithBBox(100, interpolation="invalid")

        # map to apply ops
        data = data.map(input_columns=["image", "annotation"],
                        output_columns=["image", "annotation"],
                        columns_order=["image", "annotation"],
                        operations=[test_op])  # Add column for "annotation"

        for _, _ in enumerate(data.create_dict_iterator()):
            break

    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "interpolation" in str(e)

if __name__ == "__main__":
    test_resize_with_bbox_op(plot=False)
    test_resize_with_bbox_invalid_bounds()
    test_resize_with_bbox_invalid_size()
    test_resize_with_bbox_invalid_interpolation()
