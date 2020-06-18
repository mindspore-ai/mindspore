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
Testing the random horizontal flip with bounding boxes op in DE
"""
from enum import Enum
from mindspore import log as logger
import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.c_transforms as c_vision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

GENERATE_GOLDEN = False

DATA_DIR = "../data/dataset/testVOC2012_2"


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
            return img, np.array([[0, 0, width + 1, height, 0, 0, 0]]).astype(np.uint32)

        if self.box_type == BoxType.HeightOverflow:
            # use box that overflows on height
            return img, np.array([[0, 0, width, height + 1, 0, 0, 0]]).astype(np.uint32)

        if self.box_type == BoxType.NegativeXY:
            # use box with negative xy
            return img, np.array([[-10, -10, width, height, 0, 0, 0]]).astype(np.uint32)

        if self.box_type == BoxType.OnEdge:
            # use box that covers the whole image
            return img, np.array([[0, 0, width, height, 0, 0, 0]]).astype(np.uint32)

        if self.box_type == BoxType.WrongShape:
            # use box that covers the whole image
            return img, np.array([[0, 0, width - 1]]).astype(np.uint32)
        return img, bboxes


def h_flip(image):
    """
    Apply the random_horizontal
    """

    # with the seed provided in this test case, it will always flip.
    # that's why we flip here too
    image = image[:, ::-1, :]
    return image


def check_bad_box(data, box_type, expected_error):
    """
    :param data: de object detection pipeline
    :param box_type: type of bad box
    :param expected_error: error expected to get due to bad box
    :return: None
    """
    # DEFINE TEST OP HERE -- (PROB 1 IN CASE OF RANDOM)
    try:
        test_op = c_vision.RandomHorizontalFlipWithBBox(1)
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
    except RuntimeError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert expected_error in str(error)


def fix_annotate(bboxes):
    """
    Fix annotations to format followed by mindspore.
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
    """
    :param unaugmented_data: original data
    :param augment_data: data after augmentations
    :return: None
    """
    for idx, (un_aug_item, aug_item) in \
            enumerate(zip(unaugmented_data.create_dict_iterator(),
                          augment_data.create_dict_iterator())):
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


def test_random_horizontal_bbox_op(plot=False):
    """
    Test RandomHorizontalFlipWithBBox op
    Prints images side by side with and without Aug applied + bboxes to compare and test
    """
    logger.info("test_random_horizontal_bbox_c")

    data_voc1 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)

    # DEFINE TEST OP HERE -- (PROB 1 IN CASE OF RANDOM)
    test_op = c_vision.RandomHorizontalFlipWithBBox(1)

    # maps to fix annotations to minddata standard
    data_voc1 = data_voc1.map(input_columns=["annotation"],
                              output_columns=["annotation"],
                              operations=fix_annotate)
    data_voc2 = data_voc2.map(input_columns=["annotation"],
                              output_columns=["annotation"],
                              operations=fix_annotate)
    # map to apply ops
    data_voc2 = data_voc2.map(input_columns=["image", "annotation"],
                              output_columns=["image", "annotation"],
                              columns_order=["image", "annotation"],
                              operations=[test_op])  # Add column for "annotation"
    if plot:
        visualize(data_voc1, data_voc2)


def test_random_horizontal_bbox_valid_prob_c(plot=False):
    """
    Test RandomHorizontalFlipWithBBox op
    Prints images side by side with and without Aug applied + bboxes to compare and test
    """
    logger.info("test_random_horizontal_bbox_valid_prob_c")

    data_voc1 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    # DEFINE TEST OP HERE -- (PROB 1 IN CASE OF RANDOM)
    test_op = c_vision.RandomHorizontalFlipWithBBox(0.3)

    # maps to fix annotations to minddata standard
    data_voc1 = data_voc1.map(input_columns=["annotation"],
                              output_columns=["annotation"],
                              operations=fix_annotate)
    data_voc2 = data_voc2.map(input_columns=["annotation"],
                              output_columns=["annotation"],
                              operations=fix_annotate)
    # map to apply ops
    data_voc2 = data_voc2.map(input_columns=["image", "annotation"],
                              output_columns=["image", "annotation"],
                              columns_order=["image", "annotation"],
                              operations=[test_op])  # Add column for "annotation"
    if plot:
        visualize(data_voc1, data_voc2)


def test_random_horizontal_bbox_invalid_prob_c():
    """
    Test RandomHorizontalFlipWithBBox op with invalid input probability
    """
    logger.info("test_random_horizontal_bbox_invalid_prob_c")

    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)

    try:
        # Note: Valid range of prob should be [0.0, 1.0]
        test_op = c_vision.RandomHorizontalFlipWithBBox(1.5)
        data_voc2 = data_voc2.map(input_columns=["annotation"],
                                  output_columns=["annotation"],
                                  operations=fix_annotate)
        # map to apply ops
        data_voc2 = data_voc2.map(input_columns=["image", "annotation"],
                                  output_columns=["image", "annotation"],
                                  columns_order=["image", "annotation"],
                                  operations=[test_op])  # Add column for "annotation"
    except ValueError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "Input is not" in str(error)


def test_random_horizontal_bbox_invalid_bounds_c():
    """
    Test RandomHorizontalFlipWithBBox op with invalid bounding boxes
    """
    logger.info("test_random_horizontal_bbox_invalid_bounds_c")

    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    check_bad_box(data_voc2, BoxType.WidthOverflow, "bounding boxes is out of bounds of the image")
    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    check_bad_box(data_voc2, BoxType.HeightOverflow, "bounding boxes is out of bounds of the image")
    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    check_bad_box(data_voc2, BoxType.NegativeXY, "min_x")
    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    check_bad_box(data_voc2, BoxType.WrongShape, "4 features")

if __name__ == "__main__":
    # set to false to not show plots
    test_random_horizontal_bbox_op(False)
    test_random_horizontal_bbox_valid_prob_c(False)
    test_random_horizontal_bbox_invalid_prob_c()
    test_random_horizontal_bbox_invalid_bounds_c()
