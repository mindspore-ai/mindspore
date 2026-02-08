# Copyright 2025 Huawei Technologies Co., Ltd
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
"""text transform - vectors"""

import os
import pytest
import numpy as np
import mindspore.dataset as ds
from mindspore.dataset import text
import mindspore.dataset.text.transforms as T
from mindspore import log


TEST_DATA_DATASET_FUNC ="../data/dataset/"

DATASET_ROOT_PATH = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "vectors_data")
DATASET_ROOT_PATH5 = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "vectors_data", "vectors_dim_different.txt")
DATASET_ROOT_PATH6 = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "vectors_data", "vectors_empty.txt")
DATASET_ROOT_PATH7 = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "vectors_data", "vectors.txt")
DATASET_ROOT_PATH8 = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "vectors_data", "not_exist.txt")
DATASET_ROOT_PATH9 = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "vectors_data", "vectors_with_wrong_info.txt")
glove_file = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "vectors_data", "glove.6B.50d.fragment.txt")
glove_file2 = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "vectors_data", "glove.6B.100d.txt")
word_file = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "vectors_data", "words1.txt")
DATASET_ROOT_PATH2 = "../data/dataset/testVectors/"


def test_vectors_operation_01():
    """
    Feature: Vectors op
    Description: Test Vectors op with different parameters including path and max_vectors
    Expectation: Successfully load and convert words to vectors
    """
    # Description: test with all parameters which include `path` and `max_vector` in function BuildFromFile
    vectors = text.Vectors.from_file(glove_file, max_vectors=100)
    to_vectors = text.ToVectors(vectors)
    data = ds.TextFileDataset(word_file, shuffle=False)
    data = data.map(operations=to_vectors, input_columns=["text"])
    ind = 0
    res = [[0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.044457, -0.49688, -0.17862, -0.00066023, -0.6566,
            0.27843, -0.14767, -0.55677, 0.14658, -0.0095095, 0.011658, 0.10204, -0.12792, -0.8443, -0.12181,
            -0.016801, -0.33279, -0.1552, -0.23131, -0.19181, -1.8823, -0.76746, 0.099051, -0.42125, -0.19526,
            4.0071, -0.18594, -0.52287, -0.31681, 0.00059213, 0.0074449, 0.17778, -0.15897, 0.012041, -0.054223,
            -0.29871, -0.15749, -0.34758, -0.045637, -0.44251, 0.18785, 0.0027849, -0.18411, -0.11514, -0.78581],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973, -0.43478, -0.31086, -0.44999, -0.29486,
            0.16608, 0.11963, -0.41328, -0.42353, 0.59868, 0.28825, -0.11547, -0.041848, -0.67989, -0.25063,
            0.18472, 0.086876, 0.46582, 0.015035, 0.043474, -1.4671, -0.30384, -0.023441, 0.30589, -0.21785,
            3.746, 0.0042284, -0.18436, -0.46209, 0.098329, -0.11907, 0.23919, 0.1161, 0.41705, 0.056763,
            -6.3681e-05, 0.068987, 0.087939, -0.10285, -0.13931, 0.22314, -0.080803, -0.35652, 0.016413, 0.10216],
           [0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603, 0.18157, -0.52393, 0.10381, -0.17566,
            0.078852, -0.36216, -0.11829, -0.83336, 0.11917, -0.16605, 0.061555, -0.012719, -0.56623, 0.013616,
            0.22851, -0.14396, -0.067549, -0.38157, -0.23698, -1.7037, -0.86692, -0.26704, -0.2589, 0.1767,
            3.8676, -0.1613, -0.13273, -0.68881, 0.18444, 0.0052464, -0.33874, -0.078956, 0.24185, 0.36576,
            -0.34727, 0.28483, 0.075693, -0.062178, -0.38988, 0.22902, -0.21617, -0.22562, -0.093918, -0.80375],
           [0.68047, -0.039263, 0.30186, -0.17792, 0.42962, 0.032246, -0.41376, 0.13228, -0.29847, -0.085253,
            0.17118, 0.22419, -0.10046, -0.43653, 0.33418, 0.67846, 0.057204, -0.34448, -0.42785, -0.43275,
            0.55963, 0.10032, 0.18677, -0.26854, 0.037334, -2.0932, 0.22171, -0.39868, 0.20912, -0.55725,
            3.8826, 0.47466, -0.95658, -0.37788, 0.20869, -0.32752, 0.12751, 0.088359, 0.16351, -0.21634,
            -0.094375, 0.018324, 0.21048, -0.03088, -0.19722, 0.082279, -0.09434, -0.073297, -0.064699, -0.26044],
           [0.26818, 0.14346, -0.27877, 0.016257, 0.11384, 0.69923, -0.51332, -0.47368, -0.33075, -0.13834,
            0.2702, 0.30938, -0.45012, -0.4127, -0.09932, 0.038085, 0.029749, 0.10076, -0.25058, -0.51818,
            0.34558, 0.44922, 0.48791, -0.080866, -0.10121, -1.3777, -0.10866, -0.23201, 0.012839, -0.46508,
            3.8463, 0.31362, 0.13643, -0.52244, 0.3302, 0.33707, -0.35601, 0.32431, 0.12041, 0.3512, -0.069043,
            0.36885, 0.25168, -0.24517, 0.25381, 0.1367, -0.31178, -0.6321, -0.25028, -0.38097],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        res_array = np.array(res[ind], dtype=np.float32)
        assert np.array_equal(res_array, d["text"]), ind
        ind += 1

    # Description: test with only default parameter in eager mode
    vectors = text.Vectors.from_file(glove_file)
    to_vectors = T.ToVectors(vectors)
    result1 = to_vectors("the")
    result2 = to_vectors(",")
    result3 = to_vectors(".")
    result4 = to_vectors("of")
    result5 = to_vectors("to")
    result6 = to_vectors("and")
    result7 = to_vectors("nonedfec")
    res = [[0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.044457, -0.49688, -0.17862, -0.00066023, -0.6566,
            0.27843, -0.14767, -0.55677, 0.14658, -0.0095095, 0.011658, 0.10204, -0.12792, -0.8443, -0.12181,
            -0.016801, -0.33279, -0.1552, -0.23131, -0.19181, -1.8823, -0.76746, 0.099051, -0.42125, -0.19526,
            4.0071, -0.18594, -0.52287, -0.31681, 0.00059213, 0.0074449, 0.17778, -0.15897, 0.012041, -0.054223,
            -0.29871, -0.15749, -0.34758, -0.045637, -0.44251, 0.18785, 0.0027849, -0.18411, -0.11514, -0.78581],
           [0.013441, 0.23682, -0.16899, 0.40951, 0.63812, 0.47709, -0.42852, -0.55641, -0.364, -0.23938,
            0.13001, -0.063734, -0.39575, -0.48162, 0.23291, 0.090201, -0.13324, 0.078639, -0.41634, -0.15428,
            0.10068, 0.48891, 0.31226, -0.1252, -0.037512, -1.5179, 0.12612, -0.02442, -0.042961, -0.28351,
            3.5416, -0.11956, -0.014533, -0.1499, 0.21864, -0.33412, -0.13872, 0.31806, 0.70358, 0.44858,
            -0.080262, 0.63003, 0.32111, -0.46765, 0.22786, 0.36034, -0.37818, -0.56657, 0.044691, 0.30392],
           [0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973, -0.43478, -0.31086, -0.44999, -0.29486,
            0.16608, 0.11963, -0.41328, -0.42353, 0.59868, 0.28825, -0.11547, -0.041848, -0.67989, -0.25063,
            0.18472, 0.086876, 0.46582, 0.015035, 0.043474, -1.4671, -0.30384, -0.023441, 0.30589, -0.21785,
            3.746, 0.0042284, -0.18436, -0.46209, 0.098329, -0.11907, 0.23919, 0.1161, 0.41705, 0.056763,
            -6.3681e-05, 0.068987, 0.087939, -0.10285, -0.13931, 0.22314, -0.080803, -0.35652, 0.016413, 0.10216],
           [0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603, 0.18157, -0.52393, 0.10381, -0.17566,
            0.078852, -0.36216, -0.11829, -0.83336, 0.11917, -0.16605, 0.061555, -0.012719, -0.56623, 0.013616,
            0.22851, -0.14396, -0.067549, -0.38157, -0.23698, -1.7037, -0.86692, -0.26704, -0.2589, 0.1767,
            3.8676, -0.1613, -0.13273, -0.68881, 0.18444, 0.0052464, -0.33874, -0.078956, 0.24185, 0.36576,
            -0.34727, 0.28483, 0.075693, -0.062178, -0.38988, 0.22902, -0.21617, -0.22562, -0.093918, -0.80375],
           [0.68047, -0.039263, 0.30186, -0.17792, 0.42962, 0.032246, -0.41376, 0.13228, -0.29847, -0.085253,
            0.17118, 0.22419, -0.10046, -0.43653, 0.33418, 0.67846, 0.057204, -0.34448, -0.42785, -0.43275,
            0.55963, 0.10032, 0.18677, -0.26854, 0.037334, -2.0932, 0.22171, -0.39868, 0.20912, -0.55725,
            3.8826, 0.47466, -0.95658, -0.37788, 0.20869, -0.32752, 0.12751, 0.088359, 0.16351, -0.21634,
            -0.094375, 0.018324, 0.21048, -0.03088, -0.19722, 0.082279, -0.09434, -0.073297, -0.064699, -0.26044],
           [0.26818, 0.14346, -0.27877, 0.016257, 0.11384, 0.69923, -0.51332, -0.47368, -0.33075, -0.13834,
            0.2702, 0.30938, -0.45012, -0.4127, -0.09932, 0.038085, 0.029749, 0.10076, -0.25058, -0.51818,
            0.34558, 0.44922, 0.48791, -0.080866, -0.10121, -1.3777, -0.10866, -0.23201, 0.012839, -0.46508,
            3.8463, 0.31362, 0.13643, -0.52244, 0.3302, 0.33707, -0.35601, 0.32431, 0.12041, 0.3512, -0.069043,
            0.36885, 0.25168, -0.24517, 0.25381, 0.1367, -0.31178, -0.6321, -0.25028, -0.38097],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    res_array = np.array(res, dtype=np.float32)
    assert np.array_equal(result1, res_array[0])
    assert np.array_equal(result2, res_array[1])
    assert np.array_equal(result3, res_array[2])
    assert np.array_equal(result4, res_array[3])
    assert np.array_equal(result5, res_array[4])
    assert np.array_equal(result6, res_array[5])
    assert np.array_equal(result7, res_array[6])


def test_vectors_operation_02():
    """
    Feature: Vectors op
    Description: Test Vectors op with unk_init and lower_case_backup parameters
    Expectation: Successfully handle unknown tokens with custom initialization
    """
    # Description: test with all parameters which include `unk_init`
    vectors = text.Vectors.from_file(glove_file, max_vectors=4)
    myunk = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    to_vectors = T.ToVectors(vectors, unk_init=myunk, lower_case_backup=True)
    result1 = to_vectors("The")
    result2 = to_vectors(",")
    result3 = to_vectors(".")
    result4 = to_vectors("of")
    result5 = to_vectors("to")
    result6 = to_vectors("in")
    result7 = to_vectors("a")
    res = [[0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.044457, -0.49688, -0.17862, -0.00066023, -0.6566,
            0.27843, -0.14767, -0.55677, 0.14658, -0.0095095, 0.011658, 0.10204, -0.12792, -0.8443, -0.12181,
            -0.016801, -0.33279, -0.1552, -0.23131, -0.19181, -1.8823, -0.76746, 0.099051, -0.42125, -0.19526,
            4.0071, -0.18594, -0.52287, -0.31681, 0.00059213, 0.0074449, 0.17778, -0.15897, 0.012041, -0.054223,
            -0.29871, -0.15749, -0.34758, -0.045637, -0.44251, 0.18785, 0.0027849, -0.18411, -0.11514, -0.78581],
           [0.013441, 0.23682, -0.16899, 0.40951, 0.63812, 0.47709, -0.42852, -0.55641, -0.364, -0.23938,
            0.13001, -0.063734, -0.39575, -0.48162, 0.23291, 0.090201, -0.13324, 0.078639, -0.41634, -0.15428,
            0.10068, 0.48891, 0.31226, -0.1252, -0.037512, -1.5179, 0.12612, -0.02442, -0.042961, -0.28351,
            3.5416, -0.11956, -0.014533, -0.1499, 0.21864, -0.33412, -0.13872, 0.31806, 0.70358, 0.44858,
            -0.080262, 0.63003, 0.32111, -0.46765, 0.22786, 0.36034, -0.37818, -0.56657, 0.044691, 0.30392],
           [0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973, -0.43478, -0.31086, -0.44999, -0.29486,
            0.16608, 0.11963, -0.41328, -0.42353, 0.59868, 0.28825, -0.11547, -0.041848, -0.67989, -0.25063,
            0.18472, 0.086876, 0.46582, 0.015035, 0.043474, -1.4671, -0.30384, -0.023441, 0.30589, -0.21785,
            3.746, 0.0042284, -0.18436, -0.46209, 0.098329, -0.11907, 0.23919, 0.1161, 0.41705, 0.056763,
            -6.3681e-05, 0.068987, 0.087939, -0.10285, -0.13931, 0.22314, -0.080803, -0.35652, 0.016413, 0.10216],
           [0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603, 0.18157, -0.52393, 0.10381, -0.17566,
            0.078852, -0.36216, -0.11829, -0.83336, 0.11917, -0.16605, 0.061555, -0.012719, -0.56623, 0.013616,
            0.22851, -0.14396, -0.067549, -0.38157, -0.23698, -1.7037, -0.86692, -0.26704, -0.2589, 0.1767,
            3.8676, -0.1613, -0.13273, -0.68881, 0.18444, 0.0052464, -0.33874, -0.078956, 0.24185, 0.36576,
            -0.34727, 0.28483, 0.075693, -0.062178, -0.38988, 0.22902, -0.21617, -0.22562, -0.093918, -0.80375],
           [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
           [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
           [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
    res_array = np.array(res, dtype=np.float32)
    assert np.array_equal(result1, res_array[0])
    assert np.array_equal(result2, res_array[1])
    assert np.array_equal(result3, res_array[2])
    assert np.array_equal(result4, res_array[3])
    assert np.array_equal(result5, res_array[4])
    assert np.array_equal(result6, res_array[5])
    assert np.array_equal(result7, res_array[6])


def test_vectors_operation_03():
    """
    Feature: Vectors op
    Description: Test Vectors op in eager mode with different vector files and dimensions
    Expectation: Successfully load vectors from multiple files with different dimensions
    """
    # Description: test with all parameters which include `path` and `max_vector` in function BuildFromFile in eager mode
    vectors1 = text.Vectors.from_file(glove_file, max_vectors=4)
    to_vectors1 = T.ToVectors(vectors1)

    vectors2 = text.Vectors.from_file(glove_file2, max_vectors=2)
    to_vectors2 = T.ToVectors(vectors2)
    result = [
        to_vectors1("the"), to_vectors1(","), to_vectors1("."), to_vectors1("of"), to_vectors1("to"),
        to_vectors1("and"), to_vectors1("in"), to_vectors2("the"), to_vectors2(","), to_vectors2("."),
        to_vectors2("of"), to_vectors2("to"), to_vectors2("and"), to_vectors2("in")
    ]

    res1 = [[0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.044457, -0.49688, -0.17862, -0.00066023, -0.6566,
             0.27843, -0.14767, -0.55677, 0.14658, -0.0095095, 0.011658, 0.10204, -0.12792, -0.8443, -0.12181,
             -0.016801, -0.33279, -0.1552, -0.23131, -0.19181, -1.8823, -0.76746, 0.099051, -0.42125, -0.19526,
             4.0071, -0.18594, -0.52287, -0.31681, 0.00059213, 0.0074449, 0.17778, -0.15897, 0.012041, -0.054223,
             -0.29871, -0.15749, -0.34758, -0.045637, -0.44251, 0.18785, 0.0027849, -0.18411, -0.11514, -0.78581],
            [0.013441, 0.23682, -0.16899, 0.40951, 0.63812, 0.47709, -0.42852, -0.55641, -0.364, -0.23938,
             0.13001, -0.063734, -0.39575, -0.48162, 0.23291, 0.090201, -0.13324, 0.078639, -0.41634, -0.15428,
             0.10068, 0.48891, 0.31226, -0.1252, -0.037512, -1.5179, 0.12612, -0.02442, -0.042961, -0.28351,
             3.5416, -0.11956, -0.014533, -0.1499, 0.21864, -0.33412, -0.13872, 0.31806, 0.70358, 0.44858,
             -0.080262, 0.63003, 0.32111, -0.46765, 0.22786, 0.36034, -0.37818, -0.56657, 0.044691, 0.30392],
            [0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973, -0.43478, -0.31086, -0.44999, -0.29486,
             0.16608, 0.11963, -0.41328, -0.42353, 0.59868, 0.28825, -0.11547, -0.041848, -0.67989, -0.25063,
             0.18472, 0.086876, 0.46582, 0.015035, 0.043474, -1.4671, -0.30384, -0.023441, 0.30589, -0.21785,
             3.746, 0.0042284, -0.18436, -0.46209, 0.098329, -0.11907, 0.23919, 0.1161, 0.41705, 0.056763,
             -6.3681e-05, 0.068987, 0.087939, -0.10285, -0.13931, 0.22314, -0.080803, -0.35652, 0.016413, 0.10216],
            [0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603, 0.18157, -0.52393, 0.10381, -0.17566,
             0.078852, -0.36216, -0.11829, -0.83336, 0.11917, -0.16605, 0.061555, -0.012719, -0.56623, 0.013616,
             0.22851, -0.14396, -0.067549, -0.38157, -0.23698, -1.7037, -0.86692, -0.26704, -0.2589, 0.1767,
             3.8676, -0.1613, -0.13273, -0.68881, 0.18444, 0.0052464, -0.33874, -0.078956, 0.24185, 0.36576,
             -0.34727, 0.28483, 0.075693, -0.062178, -0.38988, 0.22902, -0.21617, -0.22562, -0.093918, -0.80375],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    res2 = [[-0.038194, -0.24487, 0.72812, -0.39961, 0.083172, 0.043953, -0.39141, 0.3344, -0.57545, 0.087459,
             0.28787, -0.06731, 0.30906, -0.26384, -0.13231, -0.20757, 0.33395, -0.33848, -0.31743, -0.48336,
             0.1464, -0.37304, 0.34577, 0.052041, 0.44946, -0.46971, 0.02628, -0.54155, -0.15518, -0.14107,
             -0.039722, 0.28277, 0.14393, 0.23464, -0.31021, 0.086173, 0.20397, 0.52624, 0.17164, -0.082378,
             -0.71787, -0.41531, 0.20335, -0.12763, 0.41367, 0.55187, 0.57908, -0.33477, -0.36559, -0.54857,
             -0.062892, 0.26584, 0.30205, 0.99775, -0.80481, -3.0243, 0.01254, -0.36942, 2.2167, 0.72201,
             -0.24978, 0.92136, 0.034514, 0.46745, 1.1079, -0.19358, -0.074575, 0.23353, -0.052062, -0.22044,
             0.057162, -0.15806, -0.30798, -0.41625, 0.37972, 0.15006, -0.53212, -0.2055, -1.2526, 0.071624,
             0.70565, 0.49744, -0.42063, 0.26148, -1.538, -0.30223, -0.073438, -0.28312, 0.37104, -0.25217,
             0.016215, -0.017099, -0.38984, 0.87424, -0.72569, -0.51058, -0.52028, -0.1459, 0.8278, 0.27062],
            [-0.10767, 0.11053, 0.59812, -0.54361, 0.67396, 0.10663, 0.038867, 0.35481, 0.06351, -0.094189,
             0.15786, -0.81665, 0.14172, 0.21939, 0.58505, -0.52158, 0.22783, -0.16642, -0.68228, 0.3587,
             0.42568, 0.19021, 0.91963, 0.57555, 0.46185, 0.42363, -0.095399, -0.42749, -0.16567, -0.056842,
             -0.29595, 0.26037, -0.26606, -0.070404, -0.27662, 0.15821, 0.69825, 0.43081, 0.27952, -0.45437,
             -0.33801, -0.58184, 0.22364, -0.5778, -0.26862, -0.20425, 0.56394, -0.58524, -0.14365, -0.64218,
             0.0054697, -0.35248, 0.16162, 1.1796, -0.47674, -2.7553, -0.1321, -0.047729, 1.0655, 1.1034,
             -0.2208, 0.18669, 0.13177, 0.15117, 0.7131, -0.35215, 0.91348, 0.61783, 0.70992, 0.23955,
             -0.14571, -0.37859, -0.045959, -0.47368, 0.2385, 0.20536, -0.18996, 0.32507, -1.1112, -0.36341,
             0.98679, -0.084776, -0.54008, 0.11726, -1.0194, -0.24424, 0.12771, 0.013884, 0.080374, -0.35414,
             0.34951, -0.7226, 0.37549, 0.4441, -0.99059, 0.61214, -0.35111, -0.83155, 0.45293, 0.082577],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    assert np.array_equal(np.array(result[0], dtype=np.float32), np.array(res1[0], dtype=np.float32))
    assert np.array_equal(np.array(result[1], dtype=np.float32), np.array(res1[1], dtype=np.float32))
    assert np.array_equal(np.array(result[2], dtype=np.float32), np.array(res1[2], dtype=np.float32))
    assert np.array_equal(np.array(result[3], dtype=np.float32), np.array(res1[3], dtype=np.float32))
    assert np.array_equal(np.array(result[4], dtype=np.float32), np.array(res1[4], dtype=np.float32))
    assert np.array_equal(np.array(result[5], dtype=np.float32), np.array(res1[5], dtype=np.float32))
    assert np.array_equal(np.array(result[6], dtype=np.float32), np.array(res1[6], dtype=np.float32))
    assert np.array_equal(np.array(result[7], dtype=np.float32), np.array(res2[0], dtype=np.float32))
    assert np.array_equal(np.array(result[8], dtype=np.float32), np.array(res2[1], dtype=np.float32))
    assert np.array_equal(np.array(result[9], dtype=np.float32), np.array(res2[2], dtype=np.float32))
    assert np.array_equal(np.array(result[10], dtype=np.float32), np.array(res2[3], dtype=np.float32))
    assert np.array_equal(np.array(result[11], dtype=np.float32), np.array(res2[4], dtype=np.float32))
    assert np.array_equal(np.array(result[12], dtype=np.float32), np.array(res2[5], dtype=np.float32))
    assert np.array_equal(np.array(result[13], dtype=np.float32), np.array(res2[6], dtype=np.float32))


def test_vectors_all_tovectors_params_eager():
    """
    Feature: Vectors
    Description: Test with all parameters which include `unk_init`
        and `lower_case_backup` in function ToVectors in eager mode
    Expectation: Output is equal to the expected value
    """
    vectors = text.Vectors.from_file(DATASET_ROOT_PATH2 + "vectors.txt", max_vectors=4)
    myUnk = [-1, -1, -1, -1, -1, -1]
    to_vectors = T.ToVectors(vectors, unk_init=myUnk, lower_case_backup=True)
    result1 = to_vectors("Ok")
    result2 = to_vectors("!")
    result3 = to_vectors("This")
    result4 = to_vectors("is")
    result5 = to_vectors("my")
    result6 = to_vectors("home")
    result7 = to_vectors("none")
    res = [[0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411],
           [0.013441, 0.23682, -0.16899, 0.40951, 0.63812, 0.47709],
           [0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973],
           [0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603],
           [-1, -1, -1, -1, -1, -1],
           [-1, -1, -1, -1, -1, -1],
           [-1, -1, -1, -1, -1, -1]]
    res_array = np.array(res, dtype=np.float32)

    assert np.array_equal(result1, res_array[0])
    assert np.array_equal(result2, res_array[1])
    assert np.array_equal(result3, res_array[2])
    assert np.array_equal(result4, res_array[3])
    assert np.array_equal(result5, res_array[4])
    assert np.array_equal(result6, res_array[5])
    assert np.array_equal(result7, res_array[6])


def test_vectors_from_file():
    """
    Feature: Vectors
    Description: Test with only default parameter
    Expectation: Output is equal to the expected value
    """
    vectors = text.Vectors.from_file(DATASET_ROOT_PATH2 + "vectors.txt")
    to_vectors = text.ToVectors(vectors)
    data = ds.TextFileDataset(DATASET_ROOT_PATH2 + "words.txt", shuffle=False)
    data = data.map(operations=to_vectors, input_columns=["text"])
    ind = 0
    res = [[0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411],
           [0, 0, 0, 0, 0, 0],
           [0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973],
           [0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603],
           [0.68047, -0.039263, 0.30186, -0.17792, 0.42962, 0.032246],
           [0.26818, 0.14346, -0.27877, 0.016257, 0.11384, 0.69923],
           [0, 0, 0, 0, 0, 0]]
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        res_array = np.array(res[ind], dtype=np.float32)
        assert np.array_equal(res_array, d["text"]), ind
        ind += 1


def test_vectors_from_file_all_buildfromfile_params():
    """
    Feature: Vectors
    Description: Test with all parameters which include `path` and `max_vector` in function BuildFromFile
    Expectation: Output is equal to the expected value
    """
    vectors = text.Vectors.from_file(DATASET_ROOT_PATH2 + "vectors.txt", max_vectors=100)
    to_vectors = text.ToVectors(vectors)
    data = ds.TextFileDataset(DATASET_ROOT_PATH2 + "words.txt", shuffle=False)
    data = data.map(operations=to_vectors, input_columns=["text"])
    ind = 0
    res = [[0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411],
           [0, 0, 0, 0, 0, 0],
           [0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973],
           [0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603],
           [0.68047, -0.039263, 0.30186, -0.17792, 0.42962, 0.032246],
           [0.26818, 0.14346, -0.27877, 0.016257, 0.11384, 0.69923],
           [0, 0, 0, 0, 0, 0]]
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        res_array = np.array(res[ind], dtype=np.float32)
        assert np.array_equal(res_array, d["text"]), ind
        ind += 1


def test_vectors_from_file_all_buildfromfile_params_eager():
    """
    Feature: Vectors
    Description: Test with all parameters which include `path` and `max_vector` in function BuildFromFile in eager mode
    Expectation: Output is equal to the expected value
    """
    vectors = text.Vectors.from_file(DATASET_ROOT_PATH2 + "vectors.txt", max_vectors=4)
    to_vectors = T.ToVectors(vectors)
    result1 = to_vectors("ok")
    result2 = to_vectors("!")
    result3 = to_vectors("this")
    result4 = to_vectors("is")
    result5 = to_vectors("my")
    result6 = to_vectors("home")
    result7 = to_vectors("none")
    res = [[0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411],
           [0.013441, 0.23682, -0.16899, 0.40951, 0.63812, 0.47709],
           [0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973],
           [0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0]]
    res_array = np.array(res, dtype=np.float32)

    assert np.array_equal(result1, res_array[0])
    assert np.array_equal(result2, res_array[1])
    assert np.array_equal(result3, res_array[2])
    assert np.array_equal(result4, res_array[3])
    assert np.array_equal(result5, res_array[4])
    assert np.array_equal(result6, res_array[5])
    assert np.array_equal(result7, res_array[6])


def test_vectors_from_file_eager():
    """
    Feature: Vectors
    Description: Test with only default parameter in eager mode
    Expectation: Output is equal to the expected value
    """
    vectors = text.Vectors.from_file(DATASET_ROOT_PATH2 + "vectors.txt")
    to_vectors = T.ToVectors(vectors)
    result1 = to_vectors("ok")
    result2 = to_vectors("!")
    result3 = to_vectors("this")
    result4 = to_vectors("is")
    result5 = to_vectors("my")
    result6 = to_vectors("home")
    result7 = to_vectors("none")
    res = [[0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411],
           [0.013441, 0.23682, -0.16899, 0.40951, 0.63812, 0.47709],
           [0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973],
           [0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603],
           [0.68047, -0.039263, 0.30186, -0.17792, 0.42962, 0.032246],
           [0.26818, 0.14346, -0.27877, 0.016257, 0.11384, 0.69923],
           [0, 0, 0, 0, 0, 0]]
    res_array = np.array(res, dtype=np.float32)

    assert np.array_equal(result1, res_array[0])
    assert np.array_equal(result2, res_array[1])
    assert np.array_equal(result3, res_array[2])
    assert np.array_equal(result4, res_array[3])
    assert np.array_equal(result5, res_array[4])
    assert np.array_equal(result6, res_array[5])
    assert np.array_equal(result7, res_array[6])


def test_vectors_invalid_input():
    """
    Feature: Vectors
    Description: Test the validate function with invalid parameters
    Expectation: Correct error is raised as expected
    """
    def test_invalid_input(test_name, file_path, error, error_msg, max_vectors=None,
                           unk_init=None, lower_case_backup=False, token="ok"):
        log.info("Test Vectors with wrong input: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            vectors = text.Vectors.from_file(file_path, max_vectors=max_vectors)
            to_vectors = T.ToVectors(vectors, unk_init=unk_init, lower_case_backup=lower_case_backup)
            to_vectors(token)
        assert error_msg in str(error_info.value)

    test_invalid_input("Not all vectors have the same number of dimensions",
                       DATASET_ROOT_PATH2 + "vectors_dim_different.txt", error=RuntimeError,
                       error_msg="all vectors must have the same number of dimensions, but got dim 5 while expecting 6")
    test_invalid_input("the file is empty.", DATASET_ROOT_PATH2 + "vectors_empty.txt",
                       error=RuntimeError, error_msg="invalid file, file is empty.")
    test_invalid_input("the count of `unknown_init`'s element is different with word vector.",
                       DATASET_ROOT_PATH2 + "vectors.txt",
                       error=RuntimeError, error_msg="ToVectors: " +
                       "unk_init must be the same length as vectors, but got unk_init: 2 and vectors: 6",
                       unk_init=[-1, -1])
    test_invalid_input("The file not exist", DATASET_ROOT_PATH2 + "not_exist.txt", error=RuntimeError,
                       error_msg="get real path failed")
    test_invalid_input("The token is 1-dimensional",
                       DATASET_ROOT_PATH2 + "vectors_with_wrong_info.txt", error=RuntimeError,
                       error_msg="token with 1-dimensional vector.")
    test_invalid_input("max_vectors parameter must be greater than 0",
                       DATASET_ROOT_PATH2 + "vectors.txt", error=ValueError,
                       error_msg="Input max_vectors is not within the required interval", max_vectors=-1)
    test_invalid_input("invalid max_vectors parameter type as a float",
                       DATASET_ROOT_PATH2 + "vectors.txt", error=TypeError,
                       error_msg="Argument max_vectors with value 1.0 is not of type [<class 'int'>],"
                       " but got <class 'float'>.", max_vectors=1.0)
    test_invalid_input("invalid max_vectors parameter type as a string",
                       DATASET_ROOT_PATH2 + "vectors.txt", error=TypeError,
                       error_msg="Argument max_vectors with value 1 is not of type [<class 'int'>],"
                       " but got <class 'str'>.", max_vectors="1")
    test_invalid_input("invalid token parameter type as a float", DATASET_ROOT_PATH2 + "vectors.txt",
                       error=RuntimeError, error_msg="input tensor type should be string.", token=1.0)
    test_invalid_input("invalid lower_case_backup parameter type as a string", DATASET_ROOT_PATH2 + "vectors.txt",
                       error=TypeError, error_msg="Argument lower_case_backup with " +
                       "value True is not of type [<class 'bool'>],"
                       " but got <class 'str'>.", lower_case_backup="True")
    test_invalid_input("invalid lower_case_backup parameter type as a string", DATASET_ROOT_PATH2 + "vectors.txt",
                       error=TypeError, error_msg="Argument lower_case_backup with " +
                       "value True is not of type [<class 'bool'>],"
                       " but got <class 'str'>.", lower_case_backup="True")


def test_vectors_exception_01():
    """
    Feature: Vectors op
    Description: Test Vectors op with invalid parameters and file formats
    Expectation: Raise expected exceptions for invalid inputs
    """
    # Description: test the validate function with invalid parameters.
    token = "ok"
    with pytest.raises(RuntimeError,
                       match="all vectors must have the same number of dimensions, but got dim 5 while expecting 6"):
        vectors = text.Vectors.from_file(DATASET_ROOT_PATH5)
        to_vectors = T.ToVectors(vectors)
        to_vectors(token)

    # the file is empty.
    token = "ok"
    with pytest.raises(RuntimeError, match="invalid file, file is empty."):
        vectors = text.Vectors.from_file(DATASET_ROOT_PATH6)
        to_vectors = T.ToVectors(vectors)
        to_vectors(token)

    # the count of `unknown_init`'s element is different with word vector.
    token = "ok"
    with pytest.raises(RuntimeError,
                       match="unk_init must be the same length as vectors, but got unk_init: 2 and vectors: 6"):
        vectors = text.Vectors.from_file(DATASET_ROOT_PATH7)
        to_vectors = T.ToVectors(vectors, unk_init=[-1, -1])
        to_vectors(token)

    # The file not exist, get real path failed
    token = "ok"
    with pytest.raises(RuntimeError, match="Vectors: "):
        vectors = text.Vectors.from_file(DATASET_ROOT_PATH8)
        to_vectors = T.ToVectors(vectors)
        to_vectors(token)

    # The token is 1-dimensional
    token = "ok"
    with pytest.raises(RuntimeError, match="token with 1-dimensional vector."):
        vectors = text.Vectors.from_file(DATASET_ROOT_PATH9)
        to_vectors = T.ToVectors(vectors)
        to_vectors(token)

    # max_vectors parameter must be greater than 0
    token = "ok"
    with pytest.raises(ValueError,
                       match="Input max_vectors is not within the required interval of \\[0, 2147483647\\]."):
        vectors = text.Vectors.from_file(DATASET_ROOT_PATH7, max_vectors=-1)
        to_vectors = T.ToVectors(vectors)
        to_vectors(token)

    # invalid max_vectors parameter type as a float
    token = "ok"
    with pytest.raises(TypeError, match="Argument max_vectors with value 1.0 is not of type \\["
                                        "<class 'int'>\\], but got <class 'float'>."):
        vectors = text.Vectors.from_file(DATASET_ROOT_PATH7, max_vectors=1.0)
        to_vectors = T.ToVectors(vectors)
        to_vectors(token)

    # invalid max_vectors parameter type as a string
    token = "ok"
    with pytest.raises(TypeError, match="Argument max_vectors with value 1 is not of type \\["
                                        "<class 'int'>\\], but got <class 'str'>."):
        vectors = text.Vectors.from_file(DATASET_ROOT_PATH7, max_vectors="1")
        to_vectors = T.ToVectors(vectors)
        to_vectors(token)

    # invalid token parameter type as a float
    with pytest.raises(RuntimeError, match="input tensor type should be string."):
        vectors = text.Vectors.from_file(DATASET_ROOT_PATH7)
        to_vectors = T.ToVectors(vectors)
        to_vectors(1.0)

    # invalid token parameter type as a float
    with pytest.raises(TypeError, match="Argument lower_case_backup with value True is not of type"
                                        " \\[<class 'bool'>\\], but got <class 'str'>."):
        vectors = text.Vectors.from_file(DATASET_ROOT_PATH7)
        to_vectors = T.ToVectors(vectors, lower_case_backup="True")
        to_vectors(1.0)
