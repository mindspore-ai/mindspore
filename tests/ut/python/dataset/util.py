# Copyright 2019-2022 Huawei Technologies Co., Ltd
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
import mindspore.dataset.vision.transforms as vision

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
    if check_pillow_version and PIL.__version__ >= '9.0.0':
        try:
            np.testing.assert_equal(result_dict, dict(golden_array))
        except AssertionError:
            logger.warning(
                "Results from Pillow >= 9.0.0 is incompatibale with Pillow < 9.0.0, need more validation.")
    elif check_pillow_version:
        # Note: The version of PILLOW that is used in Jenkins CI is >= 9.0.0 and
        #       some of the md5 results files that are generated with PILLOW 7.2.0
        #       are not compatible with PILLOW 9.0.0.
        np.testing.assert_equal(result_dict, dict(golden_array),
                                'Items are not equal and problem may be due to PILLOW version incompatibility')
    else:
        np.testing.assert_equal(result_dict, dict(golden_array))


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

    # each data is a dictionary
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        for data_key in list(item.keys()):
            if data_key not in result_dict:
                result_dict[data_key] = []
            result_dict[data_key].append(item[data_key].tolist())
        num_iter += 1

    logger.info("Number of data in ds1: {}".format(num_iter))

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    golden_ref_dir = os.path.join(
        cur_dir, "../../data/dataset", 'golden', filename)
    if generate_golden:
        # Save as the golden result
        _save_golden_dict(cur_dir, golden_ref_dir, result_dict)

    _compare_to_golden_dict(golden_ref_dir, result_dict, False)

    if SAVE_JSON:
        # Save result to a json file for inspection
        parameters = {"params": {}}
        _save_json(filename, parameters, result_dict)


def _helper_save_and_check_md5(data, filename, generate_golden=False):
    """
    Helper for save_and_check_md5 for both PIL and non-PIL
    """
    num_iter = 0
    result_dict = {}

    # each data is a dictionary
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        for data_key in list(item.keys()):
            if data_key not in result_dict:
                result_dict[data_key] = []
            # save the md5 as numpy array
            result_dict.get(data_key).append(np.frombuffer(
                hashlib.md5(item[data_key]).digest(), dtype='<f4'))
        num_iter += 1

    logger.info("Number of data in ds1: {}".format(num_iter))

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    golden_ref_dir = os.path.join(
        cur_dir, "../../data/dataset", 'golden', filename)
    if generate_golden:
        # Save as the golden result
        _save_golden_dict(cur_dir, golden_ref_dir, result_dict)

    return golden_ref_dir, result_dict


def save_and_check_md5(data, filename, generate_golden=False):
    """
    Save the dataset dictionary and compare (as dictionary) with golden file (md5) for non-PIL only.
    Use create_dict_iterator to access the dataset.
    """
    golden_ref_dir, result_dict = _helper_save_and_check_md5(data, filename, generate_golden)
    _compare_to_golden_dict(golden_ref_dir, result_dict, False)


def save_and_check_md5_pil(data, filename, generate_golden=False):
    """
    Save the dataset dictionary and compare (as dictionary) with golden file (md5) for PIL only.
    If PIL version >= 9.0.0, only log warning when assertion fails and allow the test to succeed.
    Use create_dict_iterator to access the dataset.
    """
    golden_ref_dir, result_dict = _helper_save_and_check_md5(data, filename, generate_golden)
    _compare_to_golden_dict(golden_ref_dir, result_dict, True)


def save_and_check_tuple(data, parameters, filename, generate_golden=False):
    """
    Save the dataset dictionary and compare (as numpy array) with golden file.
    Use create_tuple_iterator to access the dataset.
    """
    num_iter = 0
    result_dict = {}

    # each data is a dictionary
    for item in data.create_tuple_iterator(num_epochs=1, output_numpy=True):
        for data_key, _ in enumerate(item):
            if data_key not in result_dict:
                result_dict[data_key] = []
            result_dict[data_key].append(item[data_key].tolist())
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    golden_ref_dir = os.path.join(
        cur_dir, "../../data/dataset", 'golden', filename)
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
    logger.info("seed: original = {}  new = {} ".format(
        seed_original, seed_new))
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


def config_get_set_enable_shared_mem(enable_shared_mem_new):
    """
    Get and return the original configuration enable_shared_mem value.
    Set the new configuration enable_shared_mem value.
    """
    enable_shared_mem_original = ds.config.get_enable_shared_mem()
    ds.config.set_enable_shared_mem(enable_shared_mem_new)
    logger.info("enable_shared_mem: original = {}  new = {} ".format(enable_shared_mem_original,
                                                                     enable_shared_mem_new))
    return enable_shared_mem_original


def diff_mse(in1, in2):
    mse = (np.square(in1.astype(float) / 255 - in2.astype(float) / 255)).mean()
    return mse * 100


def diff_me(in1, in2):
    mse = (np.abs(in1.astype(float) - in2.astype(float))).mean()
    return mse / 255 * 100


def visualize_audio(waveform, expect_waveform):
    """
    Visualizes audio waveform.
    """
    plt.figure(1)
    plt.subplot(1, 3, 1)
    plt.imshow(waveform)
    plt.title("waveform")

    plt.subplot(1, 3, 2)
    plt.imshow(expect_waveform)
    plt.title("expect waveform")

    plt.subplot(1, 3, 3)
    plt.imshow(waveform - expect_waveform)
    plt.title("difference")

    plt.show()


def visualize_one_channel_dataset(images_original, images_transformed, labels):
    """
    Helper function to visualize one channel grayscale images
    """
    num_samples = len(images_original)
    for i in range(num_samples):
        plt.subplot(2, num_samples, i + 1)
        # Note: Use squeeze() to convert (H, W, 1) images to (H, W)
        plt.imshow(images_original[i].squeeze(), cmap=plt.cm.gray)
        plt.title(PLOT_TITLE_DICT.get(1)[0] + ":" + str(labels[i]))

        plt.subplot(2, num_samples, i + num_samples + 1)
        plt.imshow(images_transformed[i].squeeze(), cmap=plt.cm.gray)
        plt.title(PLOT_TITLE_DICT.get(1)[1] + ":" + str(labels[i]))
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

    # creates batches of images to plot together
    batch_size = int(len(orig) / plot_rows)
    split_point = batch_size * plot_rows

    orig, aug = np.array(orig), np.array(aug)

    if len(orig) > plot_rows:
        # Create batches of required size and add remainder to last batch
        orig = np.split(orig[:split_point], batch_size) + (
            [orig[split_point:]] if (split_point < orig.shape[0]) else [])  # check to avoid empty arrays being added
        aug = np.split(aug[:split_point], batch_size) + \
            ([aug[split_point:]] if (split_point < aug.shape[0]) else [])
    else:
        orig = [orig]
        aug = [aug]

    for ix, all_data in enumerate(zip(orig, aug)):
        base_ix = ix * plot_rows  # current batch starting index
        cur_plot = len(all_data[0])

        fig, axs = plt.subplots(cur_plot, 2)
        fig.tight_layout(pad=1.5)

        for x, (data_a, data_b) in enumerate(zip(all_data[0], all_data[1])):
            cur_ix = base_ix + x
            # select plotting axes based on number of image rows on plot - else case when 1 row
            (ax_a, ax_b) = (axs[x, 0], axs[x, 1]) if (
                cur_plot > 1) else (axs[0], axs[1])

            ax_a.imshow(data_a["image"])
            add_bounding_boxes(ax_a, data_a[annot_name])
            ax_a.title.set_text("Original" + str(cur_ix + 1))

            ax_b.imshow(data_b["image"])
            add_bounding_boxes(ax_b, data_b[annot_name])
            ax_b.title.set_text("Augmented" + str(cur_ix + 1))

            logger.info(
                "Original **\n{} : {}".format(str(cur_ix + 1), data_a[annot_name]))
            logger.info(
                "Augmented **\n{} : {}\n".format(str(cur_ix + 1), data_b[annot_name]))

        plt.show()


def helper_perform_ops_bbox(data, test_op=None, edge_case=False):
    """
    Transform data based on test_op and whether it is an edge_case
    :param data: original images
    :param test_op: random operation being tested
    :param edge_case: boolean whether edge_case is being augmented (note only for bbox data type edge case)
    :return: transformed data
    """
    if edge_case:
        if test_op:
            return data.map(
                operations=[lambda img, bboxes: (
                    img, np.array([[0, 0, img.shape[1], img.shape[0]]]).astype(bboxes.dtype)), test_op],
                input_columns=["image", "bbox"],
                output_columns=["image", "bbox"])
        return data.map(
            operations=[lambda img, bboxes: (
                img, np.array([[0, 0, img.shape[1], img.shape[0]]]).astype(bboxes.dtype))],
            input_columns=["image", "bbox"],
            output_columns=["image", "bbox"])

    if test_op:
        return data.map(operations=[test_op], input_columns=["image", "bbox"],
                        output_columns=["image", "bbox"])

    return data


def helper_perform_ops_bbox_edgecase_float(data):
    """
    Transform data based an edge_case covering full image with float32
    :param data: original images
    :return: transformed data
    """
    return data.map(operations=lambda img, bbox: (img, np.array(
        [[0, 0, img.shape[1], img.shape[0], 0, 0, 0]]).astype(np.float32)),
                    input_columns=["image", "bbox"],
                    output_columns=["image", "bbox"])


def helper_test_visual_bbox(plot_vis, data1, data2):
    """
    Create list based of original images and bboxes with and without aug
    :param plot_vis: boolean based on the test argument
    :param data1: data without test_op
    :param data2: data with test_op
    :return: None
    """
    unaug_samp, aug_samp = [], []

    for un_aug, aug in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                           data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        unaug_samp.append(un_aug)
        aug_samp.append(aug)

    if plot_vis:
        visualize_with_bounding_boxes(unaug_samp, aug_samp)


def helper_random_op_pipeline(data_dir, additional_op=None):
    """
    Create an original/transformed images at data_dir based on additional_op
    :param data_dir: directory of the data
    :param additional_op: additional operation to be pipelined, if None, then gives original images
    :return: transformed image
    """
    data_set = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    transforms = [vision.Decode(), vision.Resize(size=[224, 224])]
    if additional_op:
        transforms.append(additional_op)
    ds_transformed = data_set.map(operations=transforms, input_columns="image")
    ds_transformed = ds_transformed.batch(512)

    for idx, (image, _) in enumerate(ds_transformed):
        if idx == 0:
            images_transformed = image.asnumpy()
        else:
            images_transformed = np.append(images_transformed,
                                           image.asnumpy(),
                                           axis=0)

    return images_transformed


def helper_invalid_bounding_box_test(data_dir, test_op):
    """
    Helper function for invalid bounding box test by calling check_bad_bbox
    :param data_dir: directory of the data
    :param test_op: operation that is being tested
    :return: None
    """
    data = ds.VOCDataset(
        data_dir, task="Detection", usage="train", shuffle=False, decode=True)
    check_bad_bbox(data, test_op, InvalidBBoxType.WidthOverflow,
                   "bounding boxes is out of bounds of the image")
    data = ds.VOCDataset(
        data_dir, task="Detection", usage="train", shuffle=False, decode=True)
    check_bad_bbox(data, test_op, InvalidBBoxType.HeightOverflow,
                   "bounding boxes is out of bounds of the image")
    data = ds.VOCDataset(
        data_dir, task="Detection", usage="train", shuffle=False, decode=True)
    check_bad_bbox(data, test_op,
                   InvalidBBoxType.NegativeXY, "negative value")
    data = ds.VOCDataset(
        data_dir, task="Detection", usage="train", shuffle=False, decode=True)
    check_bad_bbox(data, test_op,
                   InvalidBBoxType.WrongShape, "4 features")


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
                        output_columns=["image", "bbox"])
        # map to apply ops
        data = data.map(operations=[test_op], input_columns=["image", "bbox"],
                        output_columns=["image", "bbox"])
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
