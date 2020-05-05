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

import json
import os
import hashlib
import numpy as np
import matplotlib.pyplot as plt
#import jsbeautifier
from mindspore import log as logger

COLUMNS = ["col_1d", "col_2d", "col_3d", "col_binary", "col_float",
           "col_sint16", "col_sint32", "col_sint64"]
SAVE_JSON = False


def save_golden(cur_dir, golden_ref_dir, result_dict):
    """
    Save the dictionary values as the golden result in .npz file
    """
    logger.info("cur_dir is {}".format(cur_dir))
    logger.info("golden_ref_dir is {}".format(golden_ref_dir))
    np.savez(golden_ref_dir, np.array(list(result_dict.values())))


def save_golden_dict(cur_dir, golden_ref_dir, result_dict):
    """
    Save the dictionary (both keys and values) as the golden result in .npz file
    """
    logger.info("cur_dir is {}".format(cur_dir))
    logger.info("golden_ref_dir is {}".format(golden_ref_dir))
    np.savez(golden_ref_dir, np.array(list(result_dict.items())))


def compare_to_golden(golden_ref_dir, result_dict):
    """
    Compare as numpy arrays the test result to the golden result
    """
    test_array = np.array(list(result_dict.values()))
    golden_array = np.load(golden_ref_dir, allow_pickle=True)['arr_0']
    assert np.array_equal(test_array, golden_array)


def compare_to_golden_dict(golden_ref_dir, result_dict):
    """
    Compare as dictionaries the test result to the golden result
    """
    golden_array = np.load(golden_ref_dir, allow_pickle=True)['arr_0']
    np.testing.assert_equal(result_dict, dict(golden_array))
    # assert result_dict == dict(golden_array)


def save_json(filename, parameters, result_dict):
    """
    Save the result dictionary in json file
    """
    fout = open(filename[:-3] + "json", "w")
    options = jsbeautifier.default_options()
    options.indent_size = 2

    out_dict = {**parameters, **{"columns": result_dict}}
    fout.write(jsbeautifier.beautify(json.dumps(out_dict), options))


def save_and_check(data, parameters, filename, generate_golden=False):
    """
    Save the dataset dictionary and compare (as numpy array) with golden file.
    Use create_dict_iterator to access the dataset.
    """
    num_iter = 0
    result_dict = {}
    for column_name in COLUMNS:
        result_dict[column_name] = []

    for item in data.create_dict_iterator():  # each data is a dictionary
        for data_key in list(item.keys()):
            if data_key not in result_dict:
                result_dict[data_key] = []
            result_dict[data_key].append(item[data_key].tolist())
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    golden_ref_dir = os.path.join(cur_dir, "../../data/dataset", 'golden', filename)
    if generate_golden:
        # Save as the golden result
        save_golden(cur_dir, golden_ref_dir, result_dict)

    compare_to_golden(golden_ref_dir, result_dict)

    if SAVE_JSON:
        # Save result to a json file for inspection
        save_json(filename, parameters, result_dict)


def save_and_check_dict(data, filename, generate_golden=False):
    """
    Save the dataset dictionary and compare (as dictionary) with golden file.
    Use create_dict_iterator to access the dataset.
    """
    num_iter = 0
    result_dict = {}

    for item in data.create_dict_iterator():  # each data is a dictionary
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
        save_golden_dict(cur_dir, golden_ref_dir, result_dict)

    compare_to_golden_dict(golden_ref_dir, result_dict)

    if SAVE_JSON:
        # Save result to a json file for inspection
        parameters = {"params": {}}
        save_json(filename, parameters, result_dict)


def save_and_check_md5(data, filename, generate_golden=False):
    """
    Save the dataset dictionary and compare (as dictionary) with golden file (md5).
    Use create_dict_iterator to access the dataset.
    """
    num_iter = 0
    result_dict = {}

    for item in data.create_dict_iterator():  # each data is a dictionary
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
        save_golden_dict(cur_dir, golden_ref_dir, result_dict)

    compare_to_golden_dict(golden_ref_dir, result_dict)


def ordered_save_and_check(data, parameters, filename, generate_golden=False):
    """
    Save the dataset dictionary and compare (as numpy array) with golden file.
    Use create_tuple_iterator to access the dataset.
    """
    num_iter = 0

    result_dict = {}

    for item in data.create_tuple_iterator():  # each data is a dictionary
        for data_key in range(0, len(item)):
            if data_key not in result_dict:
                result_dict[data_key] = []
            result_dict[data_key].append(item[data_key].tolist())
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    golden_ref_dir = os.path.join(cur_dir, "../../data/dataset", 'golden', filename)
    if generate_golden:
        # Save as the golden result
        save_golden(cur_dir, golden_ref_dir, result_dict)

    compare_to_golden(golden_ref_dir, result_dict)

    if SAVE_JSON:
        # Save result to a json file for inspection
        save_json(filename, parameters, result_dict)


def diff_mse(in1, in2):
    mse = (np.square(in1.astype(float) / 255 - in2.astype(float) / 255)).mean()
    return mse * 100


def diff_me(in1, in2):
    mse = (np.abs(in1.astype(float) - in2.astype(float))).mean()
    return mse / 255 * 100


def visualize(image_original, image_transformed):
    """
    visualizes the image using DE op and Numpy op
    """
    num = len(image_transformed)
    for i in range(num):
        plt.subplot(2, num, i + 1)
        plt.imshow(image_original[i])
        plt.title("Original image")

        plt.subplot(2, num, i + num + 1)
        plt.imshow(image_transformed[i])
        plt.title("Transformed image")

    plt.show()
