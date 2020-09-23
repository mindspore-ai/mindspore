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

import hashlib
import json
import os
import itertools
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PIL
# import jsbeautifier
import mindspore.dataset as ds
from mindspore import log as logger

# These are list of plot title in different visualize modes
PLOT_TITLE_DICT = {
    1: ["Original image", "Transformed image"],
    2: ["c_transform image", "py_transform image"]
}
SAVE_JSON = False


def _save_golden(cur_dir, golden_ref_dir, result_dict):
    """
    Save the dictionary values as the golden result in .npz file
    """
    logger.info("cur_dir is {}".format(cur_dir))
    logger.info("golden_ref_dir is {}".format(golden_ref_dir))
    np.savez(golden_ref_dir, np.array(list(result_dict.values())))


def _save_golden_dict(cur_dir, golden_ref_dir, result_dict):
    """
    Save the dictionary (both keys and values) as the golden result in .npz file
    """
    logger.info("cur_dir is {}".format(cur_dir))
    logger.info("golden_ref_dir is {}".format(golden_ref_dir))
    np.savez(golden_ref_dir, np.array(list(result_dict.items())))


def _compare_to_golden(golden_ref_dir, result_dict):
    """
    Compare as numpy arrays the test result to the golden result
    """
    test_array = np.array(list(result_dict.values()))
    golden_array = np.load(golden_ref_dir, allow_pickle=True)['arr_0']
    np.testing.assert_array_equal(test_array, golden_array)


def _compare_to_golden_dict(golden_ref_dir, result_dict, check_pillow_version=False):
    """
    Compare as dictionaries the test result to the golden result
    """
    golden_array = np.load(golden_ref_dir, allow_pickle=True)['arr_0']
    # Note: The version of PILLOW that is used in Jenkins CI is compared with below
    if (not check_pillow_version or PIL.__version__ == '7.1.2'):
        np.testing.assert_equal(result_dict, dict(golden_array))
    else:
        # Beware: If error, PILLOW version of golden results may be incompatible with current PILLOW version
        np.testing.assert_equal(result_dict, dict(golden_array),
                                'Items are not equal and problem may be due to PILLOW version incompatibility')

def _save_json(filename, parameters, result_dict):
    """
    Save the result dictionary in json file
    """
    fout = open(filename[:-3] + "json", "w")
    options = jsbeautifier.default_options()
    options.indent_size = 2

    out_dict = {**parameters, **{"columns": result_dict}}
    fout.write(jsbeautifier.beautify(json.dumps(out_dict), options))


def save_and_check_dict(data, filename, generate_golden=False):
    """
    Save the dataset dictionary and compare (as dictionary) with golden file.
    Use create_dict_iterator to access the dataset.
    """
    num_iter = 0
    result_dict = {}

    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        for data_key in list(item.keys()):
            if data_key not in result_dict:
                result_dict[data_key] = []
            result_dict[data_key].append(item[data_key].tolist())
        num_iter += 1

    logger.info("Number of data in ds1: {}".format(num_iter))

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    golden_ref_dir = os.path.join(cur_dir, "../../data/dataset", 'golden', filename)
    if generate_golden:
        # Save as the golden result
        _save_golden_dict(cur_dir, golden_ref_dir, result_dict)

    _compare_to_golden_dict(golden_ref_dir, result_dict, False)

    if SAVE_JSON:
        # Save result to a json file for inspection
        parameters = {"params": {}}
        _save_json(filename, parameters, result_dict)


def save_and_check_md5(data, filename, generate_golden=False):
    """
    Save the dataset dictionary and compare (as dictionary) with golden file (md5).
    Use create_dict_iterator to access the dataset.
    """
    num_iter = 0
    result_dict = {}

    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        for data_key in list(item.keys()):
            if data_key not in result_dict:
                result_dict[data_key] = []
            # save the md5 as numpy array
            result_dict[data_key].append(np.frombuffer(hashlib.md5(item[data_key]).digest(), dtype='<f4'))
        num_iter += 1

    logger.info("Number of data in ds1: {}".format(num_iter))

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    golden_ref_dir = os.path.join(cur_dir, "../../data/dataset", 'golden', filename)
    if generate_golden:
        # Save as the golden result
        _save_golden_dict(cur_dir, golden_ref_dir, result_dict)

    _compare_to_golden_dict(golden_ref_dir, result_dict, True)


def save_and_check_tuple(data, parameters, filename, generate_golden=False):
    """
    Save the dataset dictionary and compare (as numpy array) with golden file.
    Use create_tuple_iterator to access the dataset.
    """
    num_iter = 0
    result_dict = {}

    for item in data.create_tuple_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        for data_key, _ in enumerate(item):
            if data_key not in result_dict:
                result_dict[data_key] = []
            result_dict[data_key].append(item[data_key].tolist())
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    golden_ref_dir = os.path.join(cur_dir, "../../data/dataset", 'golden', filename)
    if generate_golden:
        # Save as the golden result
        _save_golden(cur_dir, golden_ref_dir, result_dict)

    _compare_to_golden(golden_ref_dir, result_dict)

    if SAVE_JSON:
        # Save result to a json file for inspection
        _save_json(filename, parameters, result_dict)


def config_get_set_seed(seed_new):
    """
    Get and return the original configuration seed value.
    Set the new configuration seed value.
    """
    seed_original = ds.config.get_seed()
    ds.config.set_seed(seed_new)
    logger.info("seed: original = {}  new = {} ".format(seed_original, seed_new))
    return seed_original


def config_get_set_num_parallel_workers(num_parallel_workers_new):
    """
    Get and return the original configuration num_parallel_workers value.
    Set the new configuration num_parallel_workers value.
    """
    num_parallel_workers_original = ds.config.get_num_parallel_workers()
    ds.config.set_num_parallel_workers(num_parallel_workers_new)
    logger.info("num_parallel_workers: original = {}  new = {} ".format(num_parallel_workers_original,
                                                                        num_parallel_workers_new))
    return num_parallel_workers_original


def diff_mse(in1, in2):
    mse = (np.square(in1.astype(float) / 255 - in2.astype(float) / 255)).mean()
    return mse * 100


def diff_me(in1, in2):
    mse = (np.abs(in1.astype(float) - in2.astype(float))).mean()
    return mse / 255 * 100


def visualize_one_channel_dataset(images_original, images_transformed, labels):
    """
    Helper function to visualize one channel grayscale images
    """
    num_samples = len(images_original)
    for i in range(num_samples):
        plt.subplot(2, num_samples, i + 1)
        # Note: Use squeeze() to convert (H, W, 1) images to (H, W)
        plt.imshow(images_original[i].squeeze(), cmap=plt.cm.gray)
        plt.title(PLOT_TITLE_DICT[1][0] + ":" + str(labels[i]))

        plt.subplot(2, num_samples, i + num_samples + 1)
        plt.imshow(images_transformed[i].squeeze(), cmap=plt.cm.gray)
        plt.title(PLOT_TITLE_DICT[1][1] + ":" + str(labels[i]))
    plt.show()


def visualize_list(image_list_1, image_list_2, visualize_mode=1):
    """
    visualizes a list of images using DE op
    """
    plot_title = PLOT_TITLE_DICT[visualize_mode]
    num = len(image_list_1)
    for i in range(num):
        plt.subplot(2, num, i + 1)
        plt.imshow(image_list_1[i])
        plt.title(plot_title[0])

        plt.subplot(2, num, i + num + 1)
        plt.imshow(image_list_2[i])
        plt.title(plot_title[1])

    plt.show()


def visualize_image(image_original, image_de, mse=None, image_lib=None):
    """
    visualizes one example image with optional input: mse, image using 3rd party op.
    If three images are passing in, different image is calculated by 2nd and 3rd images.
    """
    num = 2
    if image_lib is not None:
        num += 1
    if mse is not None:
        num += 1
    plt.subplot(1, num, 1)
    plt.imshow(image_original)
    plt.title("Original image")

    plt.subplot(1, num, 2)
    plt.imshow(image_de)
    plt.title("DE Op image")

    if image_lib is not None:
        plt.subplot(1, num, 3)
        plt.imshow(image_lib)
        plt.title("Lib Op image")
        if mse is not None:
            plt.subplot(1, num, 4)
            plt.imshow(image_de - image_lib)
            plt.title("Diff image,\n mse : {}".format(mse))
    elif mse is not None:
        plt.subplot(1, num, 3)
        plt.imshow(image_original - image_de)
        plt.title("Diff image,\n mse : {}".format(mse))

    plt.show()


def visualize_with_bounding_boxes(orig, aug, annot_name="bbox", plot_rows=3):
    """
    Take a list of un-augmented and augmented images with "bbox" bounding boxes
    Plot images to compare test correct BBox augment functionality
    :param orig: list of original images and bboxes (without aug)
    :param aug: list of augmented images and bboxes
    :param annot_name: the dict key for bboxes in data, e.g "bbox" (COCO) / "bbox" (VOC)
    :param plot_rows: number of rows on plot (rows = samples on one plot)
    :return: None
    """

    def add_bounding_boxes(ax, bboxes):
        for bbox in bboxes:
            rect = patches.Rectangle((bbox[0], bbox[1]),
                                     bbox[2] * 0.997, bbox[3] * 0.997,
                                     linewidth=1.80, edgecolor='r', facecolor='none')
            # Add the patch to the Axes
            # Params to Rectangle slightly modified to prevent drawing overflow
            ax.add_patch(rect)

    # Quick check to confirm correct input parameters
    if not isinstance(orig, list) or not isinstance(aug, list):
        return
    if len(orig) != len(aug) or not orig:
        return

    batch_size = int(len(orig) / plot_rows)  # creates batches of images to plot together
    split_point = batch_size * plot_rows

    orig, aug = np.array(orig), np.array(aug)

    if len(orig) > plot_rows:
        # Create batches of required size and add remainder to last batch
        orig = np.split(orig[:split_point], batch_size) + (
            [orig[split_point:]] if (split_point < orig.shape[0]) else [])  # check to avoid empty arrays being added
        aug = np.split(aug[:split_point], batch_size) + ([aug[split_point:]] if (split_point < aug.shape[0]) else [])
    else:
        orig = [orig]
        aug = [aug]

    for ix, allData in enumerate(zip(orig, aug)):
        base_ix = ix * plot_rows  # current batch starting index
        curPlot = len(allData[0])

        fig, axs = plt.subplots(curPlot, 2)
        fig.tight_layout(pad=1.5)

        for x, (dataA, dataB) in enumerate(zip(allData[0], allData[1])):
            cur_ix = base_ix + x
            # select plotting axes based on number of image rows on plot - else case when 1 row
            (axA, axB) = (axs[x, 0], axs[x, 1]) if (curPlot > 1) else (axs[0], axs[1])

            axA.imshow(dataA["image"])
            add_bounding_boxes(axA, dataA[annot_name])
            axA.title.set_text("Original" + str(cur_ix + 1))

            axB.imshow(dataB["image"])
            add_bounding_boxes(axB, dataB[annot_name])
            axB.title.set_text("Augmented" + str(cur_ix + 1))

            logger.info("Original **\n{} : {}".format(str(cur_ix + 1), dataA[annot_name]))
            logger.info("Augmented **\n{} : {}\n".format(str(cur_ix + 1), dataB[annot_name]))

        plt.show()


class InvalidBBoxType(Enum):
    """
    Defines Invalid Bounding Bbox types for test cases
    """
    WidthOverflow = 1
    HeightOverflow = 2
    NegativeXY = 3
    WrongShape = 4


def check_bad_bbox(data, test_op, invalid_bbox_type, expected_error):
    """
    :param data: de object detection pipeline
    :param test_op: Augmentation Op to test on image
    :param invalid_bbox_type: type of bad box
    :param expected_error: error expected to get due to bad box
    :return: None
    """

    def add_bad_bbox(img, bboxes, invalid_bbox_type_):
        """
        Used to generate erroneous bounding box examples on given img.
        :param img: image where the bounding boxes are.
        :param bboxes: in [x_min, y_min, w, h, label, truncate, difficult] format
        :param box_type_: type of bad box
        :return: bboxes with bad examples added
        """
        height = img.shape[0]
        width = img.shape[1]
        if invalid_bbox_type_ == InvalidBBoxType.WidthOverflow:
            # use box that overflows on width
            return img, np.array([[0, 0, width + 1, height, 0, 0, 0]]).astype(np.float32)

        if invalid_bbox_type_ == InvalidBBoxType.HeightOverflow:
            # use box that overflows on height
            return img, np.array([[0, 0, width, height + 1, 0, 0, 0]]).astype(np.float32)

        if invalid_bbox_type_ == InvalidBBoxType.NegativeXY:
            # use box with negative xy
            return img, np.array([[-10, -10, width, height, 0, 0, 0]]).astype(np.float32)

        if invalid_bbox_type_ == InvalidBBoxType.WrongShape:
            # use box that has incorrect shape
            return img, np.array([[0, 0, width - 1]]).astype(np.float32)
        return img, bboxes

    try:
        # map to use selected invalid bounding box type
        data = data.map(operations=lambda img, bboxes: add_bad_bbox(img, bboxes, invalid_bbox_type),
                        input_columns=["image", "bbox"],
                        output_columns=["image", "bbox"],
                        column_order=["image", "bbox"])
        # map to apply ops
        data = data.map(operations=[test_op], input_columns=["image", "bbox"],
                        output_columns=["image", "bbox"],
                        column_order=["image", "bbox"])  # Add column for "bbox"
        for _, _ in enumerate(data.create_dict_iterator(num_epochs=1, output_numpy=True)):
            break
    except RuntimeError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert expected_error in str(error)


# return true if datasets are equal
def dataset_equal(data1, data2, mse_threshold):
    if data1.get_dataset_size() != data2.get_dataset_size():
        return False
    equal = True
    for item1, item2 in itertools.zip_longest(data1, data2):
        for column1, column2 in itertools.zip_longest(item1, item2):
            mse = diff_mse(column1.asnumpy(), column2.asnumpy())
            if mse > mse_threshold:
                equal = False
                break
        if not equal:
            break
    return equal


# return true if datasets are equal after modification to target
# params: data_unchanged - dataset kept unchanged
#         data_target    - dataset to be modified by foo
#         mse_threshold  - maximum allowable value of mse
#         foo            - function applied to data_target columns BEFORE compare
#         foo_args       - arguments passed into foo
def dataset_equal_with_function(data_unchanged, data_target, mse_threshold, foo, *foo_args):
    if data_unchanged.get_dataset_size() != data_target.get_dataset_size():
        return False
    equal = True
    for item1, item2 in itertools.zip_longest(data_unchanged, data_target):
        for column1, column2 in itertools.zip_longest(item1, item2):
            # note the function is to be applied to the second dataset
            column2 = foo(column2.asnumpy(), *foo_args)
            mse = diff_mse(column1.asnumpy(), column2)
            if mse > mse_threshold:
                equal = False
                break
        if not equal:
            break
    return equal
