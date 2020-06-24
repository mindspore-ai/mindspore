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
Testing RandomCropWithBBox op
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.c_transforms as c_vision
import mindspore.dataset.transforms.vision.utils as mode

from mindspore import log as logger

# updated VOC dataset with correct annotations
DATA_DIR = "../data/dataset/testVOC2012_2"


def fix_annotate(bboxes):
    """
    Update Current VOC dataset format to Proposed HQ BBox format

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


def add_bounding_boxes(ax, bboxes):
    for bbox in bboxes:
        rect = patches.Rectangle((bbox[0], bbox[1]),
                                 bbox[2], bbox[3],
                                 linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)


def vis_check(orig, aug):
    if not isinstance(orig, list) or not isinstance(aug, list):
        return False
    if len(orig) != len(aug):
        return False
    return True


def visualize(orig, aug):

    if not vis_check(orig, aug):
        return

    plotrows = 3
    compset = int(len(orig)/plotrows)

    orig, aug = np.array(orig), np.array(aug)

    orig = np.split(orig[:compset*plotrows], compset) + [orig[compset*plotrows:]]
    aug = np.split(aug[:compset*plotrows], compset) + [aug[compset*plotrows:]]

    for ix, allData in enumerate(zip(orig, aug)):
        base_ix = ix * plotrows  # will signal what base level we're on
        fig, axs = plt.subplots(len(allData[0]), 2)
        fig.tight_layout(pad=1.5)

        for x, (dataA, dataB) in enumerate(zip(allData[0], allData[1])):
            cur_ix = base_ix + x

            axs[x, 0].imshow(dataA["image"])
            add_bounding_boxes(axs[x, 0], dataA["annotation"])
            axs[x, 0].title.set_text("Original" + str(cur_ix+1))
            print("Original **\n ", str(cur_ix+1), " :", dataA["annotation"])

            axs[x, 1].imshow(dataB["image"])
            add_bounding_boxes(axs[x, 1], dataB["annotation"])
            axs[x, 1].title.set_text("Augmented" + str(cur_ix+1))
            print("Augmented **\n", str(cur_ix+1), " ", dataB["annotation"], "\n")

        plt.show()

# Functions to pass to Gen for creating invalid bounding boxes


def gen_bad_bbox_neg_xy(im, bbox):
    im_h, im_w = im.shape[0], im.shape[1]
    bbox[0][:4] = [-50, -50, im_w - 10, im_h - 10]
    return im, bbox


def gen_bad_bbox_overflow_width(im, bbox):
    im_h, im_w = im.shape[0], im.shape[1]
    bbox[0][:4] = [0, 0, im_w + 10, im_h - 10]
    return im, bbox


def gen_bad_bbox_overflow_height(im, bbox):
    im_h, im_w = im.shape[0], im.shape[1]
    bbox[0][:4] = [0, 0, im_w - 10, im_h + 10]
    return im, bbox


def gen_bad_bbox_wrong_shape(im, bbox):
    bbox = np.array([[0, 0, 0]]).astype(bbox.dtype)
    return im, bbox


badGenFuncs = [gen_bad_bbox_neg_xy,
               gen_bad_bbox_overflow_width,
               gen_bad_bbox_overflow_height,
               gen_bad_bbox_wrong_shape]


assertVal = ["min_x",
             "is out of bounds of the image",
             "is out of bounds of the image",
             "4 features"]


# Gen Edge case BBox
def gen_bbox_edge(im, bbox):
    im_h, im_w = im.shape[0], im.shape[1]
    bbox[0][:4] = [0, 0, im_w, im_h]
    return im, bbox


def test_random_crop_with_bbox_op_c(plot_vis=False):
    """
     Prints images side by side with and without Aug applied + bboxes
    """
    # Load dataset
    dataVoc1 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    dataVoc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)

    # define test OP with values to match existing Op unit - test
    test_op = c_vision.RandomCropWithBBox([512, 512], [200, 200, 200, 200])

    # maps to fix annotations to HQ standard
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
                            operations=[test_op])  # Add column for "annotation"

    unaugSamp, augSamp = [], []

    for unAug, Aug in zip(dataVoc1.create_dict_iterator(), dataVoc2.create_dict_iterator()):
        unaugSamp.append(unAug)
        augSamp.append(Aug)

    if plot_vis:
        visualize(unaugSamp, augSamp)


def test_random_crop_with_bbox_op2_c(plot_vis=False):
    """
     Prints images side by side with and without Aug applied + bboxes
     With Fill Value
    """

    # Load dataset
    dataVoc1 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    dataVoc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)

    # define test OP with values to match existing Op unit - test
    test_op = c_vision.RandomCropWithBBox(512, [200, 200, 200, 200], fill_value=(255, 255, 255))

    # maps to fix annotations to HQ standard
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
                            operations=[test_op])  # Add column for "annotation"

    unaugSamp, augSamp = [], []

    for unAug, Aug in zip(dataVoc1.create_dict_iterator(), dataVoc2.create_dict_iterator()):
        unaugSamp.append(unAug)
        augSamp.append(Aug)

    if plot_vis:
        visualize(unaugSamp, augSamp)


def test_random_crop_with_bbox_op3_c(plot_vis=False):
    """
     Prints images side by side with and without Aug applied + bboxes
     With Padding Mode passed
    """
    # Load dataset
    dataVoc1 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    dataVoc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)

    # define test OP with values to match existing Op unit - test
    test_op = c_vision.RandomCropWithBBox(512, [200, 200, 200, 200], padding_mode=mode.Border.EDGE)

    # maps to fix annotations to HQ standard
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
                            operations=[test_op])  # Add column for "annotation"

    unaugSamp, augSamp = [], []

    for unAug, Aug in zip(dataVoc1.create_dict_iterator(), dataVoc2.create_dict_iterator()):
        unaugSamp.append(unAug)
        augSamp.append(Aug)

    if plot_vis:
        visualize(unaugSamp, augSamp)


def test_random_crop_with_bbox_op_edge_c(plot_vis=False):
    """
     Prints images side by side with and without Aug applied + bboxes
     Testing for an Edge case
    """
    # Load dataset
    dataVoc1 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    dataVoc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)

    # define test OP with values to match existing Op unit - test
    test_op = c_vision.RandomCropWithBBox(512, [200, 200, 200, 200], padding_mode=mode.Border.EDGE)

    # maps to fix annotations to HQ standard
    dataVoc1 = dataVoc1.map(input_columns=["annotation"],
                            output_columns=["annotation"],
                            operations=fix_annotate)
    dataVoc2 = dataVoc2.map(input_columns=["annotation"],
                            output_columns=["annotation"],
                            operations=fix_annotate)

    # Modify BBoxes to serve as valid edge cases
    dataVoc2 = dataVoc2.map(input_columns=["image", "annotation"],
                            output_columns=["image", "annotation"],
                            columns_order=["image", "annotation"],
                            operations=[gen_bbox_edge])

    # map to apply ops
    dataVoc2 = dataVoc2.map(input_columns=["image", "annotation"],
                            output_columns=["image", "annotation"],
                            columns_order=["image", "annotation"],
                            operations=[test_op])  # Add column for "annotation"

    unaugSamp, augSamp = [], []

    for unAug, Aug in zip(dataVoc1.create_dict_iterator(), dataVoc2.create_dict_iterator()):
        unaugSamp.append(unAug)
        augSamp.append(Aug)

    if plot_vis:
        visualize(unaugSamp, augSamp)


def test_random_crop_with_bbox_op_invalid_c():
    """
    Checking for invalid params passed to Aug Constructor
    """
    # Load dataset
    dataVoc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)

    try:
        # define test OP with values to match existing Op unit - test
        test_op = c_vision.RandomCropWithBBox([512, 512, 375])

        # maps to fix annotations to HQ standard

        dataVoc2 = dataVoc2.map(input_columns=["annotation"],
                                output_columns=["annotation"],
                                operations=fix_annotate)

        # map to apply ops
        dataVoc2 = dataVoc2.map(input_columns=["image", "annotation"],
                                output_columns=["image", "annotation"],
                                columns_order=["image", "annotation"],
                                operations=[test_op])  # Add column for "annotation"

        for _ in dataVoc2.create_dict_iterator():
            break
    except TypeError as err:
        logger.info("Got an exception in DE: {}".format(str(err)))
        assert "Size should be a single integer" in str(err)


def test_random_crop_with_bbox_op_bad_c():
    # Should Fail - Errors logged to logger
    for ix, badFunc in enumerate(badGenFuncs):
        try:
            dataVoc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train",
                                     decode=True, shuffle=False)

            test_op = c_vision.RandomCropWithBBox([512, 512], [200, 200, 200, 200])

            dataVoc2 = dataVoc2.map(input_columns=["annotation"],
                                    output_columns=["annotation"],
                                    operations=fix_annotate)

            dataVoc2 = dataVoc2.map(input_columns=["image", "annotation"],
                                    output_columns=["image", "annotation"],
                                    columns_order=["image", "annotation"],
                                    operations=[badFunc])

            # map to apply ops
            dataVoc2 = dataVoc2.map(input_columns=["image", "annotation"],
                                    output_columns=["image", "annotation"],
                                    columns_order=["image", "annotation"],
                                    operations=[test_op])

            for _ in dataVoc2.create_dict_iterator():
                break  # first sample will cause exception

        except RuntimeError as err:
            logger.info("Got an exception in DE: {}".format(str(err)))
            assert assertVal[ix] in str(err)


if __name__ == "__main__":
    test_random_crop_with_bbox_op_c(plot_vis=True)
    test_random_crop_with_bbox_op2_c(plot_vis=True)
    test_random_crop_with_bbox_op3_c(plot_vis=True)
    test_random_crop_with_bbox_op_edge_c(plot_vis=True)
    test_random_crop_with_bbox_op_invalid_c()
    test_random_crop_with_bbox_op_bad_c()
