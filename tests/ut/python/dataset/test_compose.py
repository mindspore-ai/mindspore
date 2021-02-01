# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

import numpy as np
import pytest
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as c_transforms
import mindspore.dataset.transforms.py_transforms as py_transforms

import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.vision.py_transforms as py_vision

from util import visualize_list, save_and_check_md5, config_get_set_seed, config_get_set_num_parallel_workers

GENERATE_GOLDEN = False


def test_compose():
    """
    Test C++ and Python Compose Op
    """
    ds.config.set_seed(0)

    def test_config(arr, op_list):
        try:
            data = ds.NumpySlicesDataset(arr, column_names="col", shuffle=False)
            data = data.map(input_columns=["col"], operations=op_list)
            res = []
            for i in data.create_dict_iterator(output_numpy=True):
                res.append(i["col"].tolist())
            return res
        except (TypeError, ValueError) as e:
            return str(e)

    # Test simple compose with only 1 op, this would generate a warning
    assert test_config([[1, 0], [3, 4]], c_transforms.Compose([c_transforms.Fill(2)])) == [[2, 2], [2, 2]]

    # Test 1 column -> 2 columns -> 1 -> 2 -> 1
    assert test_config([[1, 0]],
                       c_transforms.Compose(
                           [c_transforms.Duplicate(), c_transforms.Concatenate(), c_transforms.Duplicate(),
                            c_transforms.Concatenate()])) \
           == [[1, 0] * 4]

    # Test one Python transform followed by a C++ transform. Type after OneHot is a float (mixed use-case)
    assert test_config([1, 0],
                       c_transforms.Compose([py_transforms.OneHotOp(2), c_transforms.TypeCast(mstype.int32)])) \
           == [[[0, 1]], [[1, 0]]]

    # Test exceptions.
    with pytest.raises(TypeError) as error_info:
        c_transforms.Compose([1, c_transforms.TypeCast(mstype.int32)])
    assert "op_list[0] is neither a c_transform op (TensorOperation) nor a callable pyfunc." in str(error_info.value)

    # Test empty op list
    with pytest.raises(ValueError) as error_info:
        test_config([1, 0], c_transforms.Compose([]))
    assert "op_list can not be empty." in str(error_info.value)

    # Test Python compose op
    assert test_config([1, 0], py_transforms.Compose([py_transforms.OneHotOp(2)])) == [[[0, 1]], [[1, 0]]]
    assert test_config([1, 0], py_transforms.Compose([py_transforms.OneHotOp(2), (lambda x: x + x)])) == [[[0, 2]],
                                                                                                          [[2, 0]]]

    # Test nested Python compose op
    assert test_config([1, 0],
                       py_transforms.Compose([py_transforms.Compose([py_transforms.OneHotOp(2)]), (lambda x: x + x)])) \
           == [[[0, 2]], [[2, 0]]]

    # Test passing a list of Python ops without Compose wrapper
    assert test_config([1, 0],
                       [py_transforms.Compose([py_transforms.OneHotOp(2)]), (lambda x: x + x)]) \
           == [[[0, 2]], [[2, 0]]]
    assert test_config([1, 0], [py_transforms.OneHotOp(2), (lambda x: x + x)]) == [[[0, 2]], [[2, 0]]]

    # Test a non callable function
    with pytest.raises(ValueError) as error_info:
        py_transforms.Compose([1])
    assert "transforms[0] is not callable." in str(error_info.value)

    # Test empty Python op list
    with pytest.raises(ValueError) as error_info:
        test_config([1, 0], py_transforms.Compose([]))
    assert "transforms list is empty." in str(error_info.value)

    # Pass in extra brackets
    with pytest.raises(TypeError) as error_info:
        py_transforms.Compose([(lambda x: x + x)])()
    assert "Compose was called without an image. Fix invocation (avoid it being invoked as Compose([...])())." in str(
        error_info.value)


def test_lambdas():
    """
    Test Multi Column Python Compose Op
    """
    ds.config.set_seed(0)

    def test_config(arr, input_columns, output_cols, op_list):
        data = ds.NumpySlicesDataset(arr, column_names=input_columns, shuffle=False)
        data = data.map(operations=op_list, input_columns=input_columns, output_columns=output_cols,
                        column_order=output_cols)
        res = []
        for i in data.create_dict_iterator(output_numpy=True):
            for col_name in output_cols:
                res.append(i[col_name].tolist())
        return res

    arr = ([[1]], [[3]])

    assert test_config(arr, ["col0", "col1"], ["a"], py_transforms.Compose([(lambda x, y: x)])) == [[1]]
    assert test_config(arr, ["col0", "col1"], ["a"], py_transforms.Compose([lambda x, y: x, lambda x: x])) == [[1]]
    assert test_config(arr, ["col0", "col1"], ["a", "b"],
                       py_transforms.Compose([lambda x, y: x, lambda x: (x, x * 2)])) == \
           [[1], [2]]
    assert test_config(arr, ["col0", "col1"], ["a", "b"],
                       [lambda x, y: (x, x + y), lambda x, y: (x, y * 2)]) == [[1], [8]]


def test_c_py_compose_transforms_module():
    """
    Test combining Python and C++ transforms
    """
    ds.config.set_seed(0)

    def test_config(arr, input_columns, output_cols, op_list):
        data = ds.NumpySlicesDataset(arr, column_names=input_columns, shuffle=False)
        data = data.map(operations=op_list, input_columns=input_columns, output_columns=output_cols,
                        column_order=output_cols)
        res = []
        for i in data.create_dict_iterator(output_numpy=True):
            for col_name in output_cols:
                res.append(i[col_name].tolist())
        return res

    arr = [1, 0]
    assert test_config(arr, ["cols"], ["cols"],
                       [py_transforms.OneHotOp(2), c_transforms.Mask(c_transforms.Relational.EQ, 1)]) == \
           [[[False, True]],
            [[True, False]]]
    assert test_config(arr, ["cols"], ["cols"],
                       [py_transforms.OneHotOp(2), (lambda x: x + x), c_transforms.Fill(1)]) \
           == [[[1, 1]], [[1, 1]]]
    assert test_config(arr, ["cols"], ["cols"],
                       [py_transforms.OneHotOp(2), (lambda x: x + x), c_transforms.Fill(1), (lambda x: x + x)]) \
           == [[[2, 2]], [[2, 2]]]
    assert test_config([[1, 3]], ["cols"], ["cols"],
                       [c_transforms.PadEnd([3], -1), (lambda x: x + x)]) \
           == [[2, 6, -2]]

    arr = ([[1]], [[3]])
    assert test_config(arr, ["col0", "col1"], ["a"], [(lambda x, y: x + y), c_transforms.PadEnd([2], -1)]) == [[4, -1]]


def test_c_py_compose_vision_module(plot=False, run_golden=True):
    """
    Test combining Python and C++ vision transforms
    """
    original_seed = config_get_set_seed(10)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    def test_config(plot, file_name, op_list):
        data_dir = "../data/dataset/testImageNetData/train/"
        data1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data1 = data1.map(operations=op_list, input_columns=["image"])
        data2 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data2 = data2.map(operations=c_vision.Decode(), input_columns=["image"])
        original_images = []
        transformed_images = []

        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            transformed_images.append(item["image"])
        for item in data2.create_dict_iterator(num_epochs=1, output_numpy=True):
            original_images.append(item["image"])

        if run_golden:
            # Compare with expected md5 from images
            save_and_check_md5(data1, file_name, generate_golden=GENERATE_GOLDEN)

        if plot:
            visualize_list(original_images, transformed_images)

    test_config(op_list=[c_vision.Decode(),
                         py_vision.ToPIL(),
                         py_vision.Resize((224, 224)),
                         np.array],
                plot=plot, file_name="compose_c_py_1.npz")

    test_config(op_list=[c_vision.Decode(),
                         c_vision.Resize((224, 244)),
                         py_vision.ToPIL(),
                         np.array,
                         c_vision.Resize((24, 24))],
                plot=plot, file_name="compose_c_py_2.npz")

    test_config(op_list=[py_vision.Decode(),
                         py_vision.Resize((224, 224)),
                         np.array,
                         c_vision.RandomColor()],
                plot=plot, file_name="compose_c_py_3.npz")

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers((original_num_parallel_workers))


def test_py_transforms_with_c_vision():
    """
    These examples will fail, as c_transform should not be used in py_transforms.Random(Apply/Choice/Order)
    """

    ds.config.set_seed(0)

    def test_config(op_list):
        data_dir = "../data/dataset/testImageNetData/train/"
        data = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data = data.map(operations=op_list)
        res = []
        for i in data.create_dict_iterator(output_numpy=True):
            for col_name in output_cols:
                res.append(i[col_name].tolist())
        return res

    with pytest.raises(ValueError) as error_info:
        test_config(py_transforms.RandomApply([c_vision.RandomResizedCrop(200)]))
    assert "transforms[0] is not a py transforms." in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        test_config(py_transforms.RandomChoice([c_vision.RandomResizedCrop(200)]))
    assert "transforms[0] is not a py transforms." in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        test_config(py_transforms.RandomOrder([np.array, c_vision.RandomResizedCrop(200)]))
    assert "transforms[1] is not a py transforms." in str(error_info.value)

    with pytest.raises(RuntimeError) as error_info:
        test_config([py_transforms.OneHotOp(20, 0.1)])
    assert "The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()" in str(
        error_info.value)


def test_py_vision_with_c_transforms():
    """
    Test combining Python vision operations with C++ transforms operations
    """

    ds.config.set_seed(0)

    def test_config(op_list):
        data_dir = "../data/dataset/testImageNetData/train/"
        data1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data1 = data1.map(operations=op_list, input_columns=["image"])
        transformed_images = []

        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            transformed_images.append(item["image"])
        return transformed_images

    # Test with Mask Op
    output_arr = test_config([py_vision.Decode(),
                              py_vision.CenterCrop((2)), np.array,
                              c_transforms.Mask(c_transforms.Relational.GE, 100)])

    exp_arr = [np.array([[[True, False, False],
                          [True, False, False]],
                         [[True, False, False],
                          [True, False, False]]]),
               np.array([[[True, False, False],
                          [True, False, False]],
                         [[True, False, False],
                          [True, False, False]]])]

    for exp_a, output in zip(exp_arr, output_arr):
        np.testing.assert_array_equal(exp_a, output)

    # Test with Fill Op
    output_arr = test_config([py_vision.Decode(),
                              py_vision.CenterCrop((4)), np.array,
                              c_transforms.Fill(10)])

    exp_arr = [np.ones((4, 4, 3)) * 10] * 2
    for exp_a, output in zip(exp_arr, output_arr):
        np.testing.assert_array_equal(exp_a, output)

    # Test with Concatenate Op, which will raise an error since ConcatenateOp only supports rank 1 tensors.
    with pytest.raises(RuntimeError) as error_info:
        test_config([py_vision.Decode(),
                     py_vision.CenterCrop((2)), np.array,
                     c_transforms.Concatenate(0)])
    assert "only 1D input supported" in str(error_info.value)


def test_compose_with_custom_function():
    """
    Test Python Compose with custom function
    """

    def custom_function(x):
        return (x, x * x)

    # First dataset
    op_list = [
        lambda x: x * 3,
        custom_function,
        # convert two column output to one
        lambda *images: np.stack(images)
    ]

    data = ds.NumpySlicesDataset([[1, 2]], column_names=["col0"], shuffle=False)
    data = data.map(input_columns=["col0"], operations=op_list)
    #

    res = []
    for i in data.create_dict_iterator(output_numpy=True):
        res.append(i["col0"].tolist())
    assert res == [[[3, 6], [9, 36]]]


if __name__ == "__main__":
    test_compose()
    test_lambdas()
    test_c_py_compose_transforms_module()
    test_c_py_compose_vision_module(plot=True)
    test_py_transforms_with_c_vision()
    test_py_vision_with_c_transforms()
    test_compose_with_custom_function()
