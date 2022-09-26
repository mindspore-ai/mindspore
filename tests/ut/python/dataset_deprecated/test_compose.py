# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
Test Compose op in Dataset
"""
import numpy as np
import pytest
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.text as text
import mindspore.dataset.transforms.c_transforms as c_transforms
import mindspore.dataset.transforms.py_transforms as py_transforms
import mindspore.dataset.transforms.transforms as data_trans
import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.vision.py_transforms as py_vision
import mindspore.dataset.vision.transforms as vision

from ..dataset.util import visualize_list, save_and_check_md5_pil, \
    config_get_set_seed, config_get_set_num_parallel_workers

GENERATE_GOLDEN = False

DATA_DIR_PK = "../data/dataset/testPK/data"
DATA_DIR_VOCAB = "../data/dataset/testVocab/words.txt"


def test_compose():
    """
    Feature: Compose Op
    Description: Test C++ and Python Compose Op
    Expectation: Invalid input is detected
    """
    original_seed = config_get_set_seed(0)

    def test_config(arr, op_list):
        try:
            data = ds.NumpySlicesDataset(arr, column_names="col", shuffle=False)
            data = data.map(input_columns=["col"], operations=op_list)
            res = []
            for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
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
        == [[0, 1], [1, 0]]

    # Test exceptions.
    with pytest.raises(TypeError) as error_info:
        c_transforms.Compose([1, c_transforms.TypeCast(mstype.int32)])
    assert "transforms[0] is neither a transforms op (TensorOperation) nor a callable pyfunc." in str(error_info.value)

    # Test empty op list
    with pytest.raises(ValueError) as error_info:
        test_config([1, 0], c_transforms.Compose([]))
    assert "transforms list can not be empty." in str(error_info.value)

    # Test Python compose op
    assert test_config([1, 0], py_transforms.Compose([py_transforms.OneHotOp(2)])) == [[0, 1], [1, 0]]
    assert test_config([1, 0], py_transforms.Compose([py_transforms.OneHotOp(2), (lambda x: x + x)])) == [[0, 2],
                                                                                                          [2, 0]]

    # Test nested Python compose op
    assert test_config([1, 0],
                       py_transforms.Compose([py_transforms.Compose([py_transforms.OneHotOp(2)]), (lambda x: x + x)])) \
        == [[0, 2], [2, 0]]

    # Test passing a list of Python ops without Compose wrapper
    assert test_config([1, 0],
                       [py_transforms.Compose([py_transforms.OneHotOp(2)]), (lambda x: x + x)]) \
        == [[0, 2], [2, 0]]
    assert test_config([1, 0], [py_transforms.OneHotOp(2), (lambda x: x + x)]) == [[0, 2], [2, 0]]

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

    # Restore configuration
    ds.config.set_seed(original_seed)


def test_lambdas():
    """
    Feature: Compose Op
    Description: Test Multi Column Python Compose Op
    Expectation: Dataset pipeline runs successfully and results are verified
    """
    original_seed = config_get_set_seed(0)

    def test_config(arr, input_columns, output_columns, op_list):
        data = ds.NumpySlicesDataset(arr, column_names=input_columns, shuffle=False)
        data = data.map(operations=op_list, input_columns=input_columns, output_columns=output_columns)
        res = []
        for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
            for col_name in output_columns:
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

    # Restore configuration
    ds.config.set_seed(original_seed)


def test_c_py_compose_transforms_module():
    """
    Feature: Compose Op
    Description: Test combining Python and C++ transforms
    Expectation: Dataset pipeline runs successfully and results are verified
    """
    original_seed = config_get_set_seed(0)

    def test_config(arr, input_columns, output_cols, op_list):
        data = ds.NumpySlicesDataset(arr, column_names=input_columns, shuffle=False)
        data = data.map(operations=op_list, input_columns=input_columns, output_columns=output_cols)
        res = []
        for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
            for col_name in output_cols:
                res.append(i[col_name].tolist())
        return res

    arr = [1, 0]
    assert test_config(arr, ["cols"], ["cols"],
                       [py_transforms.OneHotOp(2), c_transforms.Mask(c_transforms.Relational.EQ, 1)]) == \
        [[False, True],
         [True, False]]
    assert test_config(arr, ["cols"], ["cols"],
                       [py_transforms.OneHotOp(2), (lambda x: x + x), c_transforms.Fill(1)]) \
        == [[1, 1], [1, 1]]
    assert test_config(arr, ["cols"], ["cols"],
                       [py_transforms.OneHotOp(2), (lambda x: x + x), c_transforms.Fill(1), (lambda x: x + x)]) \
        == [[2, 2], [2, 2]]
    assert test_config([[1, 3]], ["cols"], ["cols"],
                       [c_transforms.PadEnd([3], -1), (lambda x: x + x)]) \
        == [[2, 6, -2]]

    arr = ([[1]], [[3]])
    assert test_config(arr, ["col0", "col1"], ["a"], [(lambda x, y: x + y), c_transforms.PadEnd([2], -1)]) == [[4, -1]]

    # Restore configuration
    ds.config.set_seed(original_seed)


def test_c_py_compose_vision_module(plot=False, run_golden=True):
    """
    Feature: Compose Op
    Description: Test combining Python and C++ vision transforms
    Expectation: Dataset pipeline runs successfully and md5 results are verified
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
            save_and_check_md5_pil(data1, file_name, generate_golden=GENERATE_GOLDEN)

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
    Feature: Compose Op
    Description: Test invalid scenarios since as c_transform should not be used in
        py_transforms.Random(Apply/Choice/Order)
    Expectation: Invalid input is detected
    """
    original_seed = config_get_set_seed(0)

    def test_config(op_list):
        data_dir = "../data/dataset/testImageNetData/train/"
        data = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data = data.map(operations=op_list)
        res = []
        for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
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
    assert "is smaller than the category number" in str(error_info.value)

    # Restore configuration
    ds.config.set_seed(original_seed)


def test_py_vision_with_c_transforms():
    """
    Feature: Compose Op
    Description: Test combining Python vision operations with C++ transforms operations
    Expectation: Dataset pipeline runs successfully for valid input and results verified. Invalid input is detected.
    """
    original_seed = config_get_set_seed(0)

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

    # Restore configuration
    ds.config.set_seed(original_seed)


def test_compose_with_custom_function():
    """
    Feature: Compose Op
    Description: Test Python Compose with custom function
    Expectation: Dataset pipeline runs successfully and results are verified
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
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        res.append(i["col0"].tolist())
    assert res == [[[3, 6], [9, 36]]]


def test_compose_unified_mix_tranforms():
    """
    Feature: Compose op
    Description: Test Unified Compose op containing mixing of old legacy c/py_transforms and new unified transforms
    Expectation: RuntimeError is detected
    """

    def test_config(transforms_list):
        # Not valid to mix legacy c/py_transforms and new unified transforms with unified Compose op
        my_samples = 5
        data_set = ds.ImageFolderDataset(DATA_DIR_PK, num_samples=my_samples, num_parallel_workers=1)

        with pytest.raises(RuntimeError) as error_info:
            compose_op = data_trans.Compose(transforms=transforms_list)
            data_set = data_set.map(operations=compose_op, input_columns="image")
            for _ in enumerate(data_set):
                pass
        assert "Mixing old legacy c/py_transforms and new unified transforms is not allowed" in str(error_info.value)

    test_config([py_vision.Decode(), lambda x: x, vision.RandomVerticalFlip()])
    test_config([vision.Decode(), py_vision.RandomVerticalFlip()])

    test_config([vision.Decode(), c_vision.RandomVerticalFlip()])
    test_config([lambda x: x, vision.Decode(), c_vision.RandomVerticalFlip(), c_vision.HorizontalFlip()])


def test_compose_c_legacy_mix_tranforms():
    """
    Feature: Compose op
    Description: Test Legacy C++ Compose op containing mixing of old legacy c_transforms and new unified transforms
    Expectation: RuntimeError is detected
    """

    def test_config(transforms_list):
        # Not valid to mix legacy c_transforms and new unified transforms with C++ Compose op
        my_samples = 5
        data_set = ds.ImageFolderDataset(DATA_DIR_PK, num_samples=my_samples, num_parallel_workers=1)

        with pytest.raises(RuntimeError) as error_info:
            compose_op = data_trans.Compose(transforms=transforms_list)
            data_set = data_set.map(operations=compose_op, input_columns="image")
            for _ in enumerate(data_set):
                pass
        assert "Mixing old legacy c/py_transforms and new unified transforms is not allowed" in str(error_info.value)

    # Test legacy Compose C++ op with old legacy transform and new unified transform
    test_config([c_vision.Decode(), vision.RandomVerticalFlip()])
    test_config([vision.Decode(), c_vision.RandomVerticalFlip(), c_vision.HorizontalFlip()])


def test_compose_py_legacy_mix_tranforms():
    """
    Feature: Compose op
    Description: Test Legacy Py Compose op containing mixing of old legacy py_transforms and new unified transforms
    Expectation: RuntimeError is detected
    """

    def test_config(transforms_list):
        # Not valid to mix legacy py_transforms and new unified transforms with C++ Compose op
        my_samples = 5
        data_set = ds.ImageFolderDataset(DATA_DIR_PK, num_samples=my_samples, num_parallel_workers=1)

        with pytest.raises(RuntimeError) as error_info:
            compose_op = data_trans.Compose(transforms=transforms_list)
            data_set = data_set.map(operations=compose_op, input_columns="image")
            for _ in enumerate(data_set):
                pass
        assert "Mixing old legacy c/py_transforms and new unified transforms is not allowed" in str(error_info.value)

    # Test legacy Compose Python op with old legacy transform and new unified transform
    test_config([py_vision.Decode(), vision.RandomVerticalFlip()])
    test_config([vision.Decode(True), py_vision.RandomVerticalFlip(), lambda x: x])


def test_compose_text_and_data_transforms():
    """
    Feature: Compose op
    Description: Test Compose op with both Text Transforms and Data Transforms
    Expectation: Dataset pipeline runs successfully and results are verified
    """
    data = ds.TextFileDataset(DATA_DIR_VOCAB, shuffle=False)

    vocab = text.Vocab.from_dataset(data, "text", freq_range=None, top_k=None,
                                    special_tokens=["<pad>", "<unk>"],
                                    special_first=True)

    padend_op = c_transforms.PadEnd([100], pad_value=vocab.tokens_to_ids('<pad>'))
    lookup_op = text.Lookup(vocab, "<unk>")

    # Use both Text Lookup op and Data Transforms PadEnd op in transforms list for Compose
    compose_op = c_transforms.Compose(transforms=[lookup_op, padend_op])
    data = data.map(operations=compose_op, input_columns=["text"])
    res = []
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        res.append(d["text"].item())
    assert res == [4, 5, 3, 6, 7, 2]


def test_compose_unified_text_mix_transforms():
    """
    Feature: Compose op
    Description: Test Unified Compose op containing text transforms plus mixing of old legacy c/py_transforms
    Expectation: RuntimeError is detected
    """

    my_data = ds.TextFileDataset(DATA_DIR_VOCAB, shuffle=False)

    # Use both Text Lookup op and Data Transforms PadEnd op in transforms list for Compose
    vocab = text.Vocab.from_dataset(my_data, "text", freq_range=None, top_k=None,
                                    special_tokens=["<pad>", "<unk>"],
                                    special_first=True)
    lookup_op = text.Lookup(vocab, "<unk>")

    # Test - unified Compose C++ op with legacy data transform
    padend_legacy_op = c_transforms.PadEnd([100], pad_value=vocab.tokens_to_ids('<pad>'))
    with pytest.raises(RuntimeError) as error_info:
        compose_op = data_trans.Compose(transforms=[lookup_op, padend_legacy_op])
        my_data = my_data.map(operations=compose_op, input_columns=["text"])
        for _ in enumerate(my_data):
            pass
    assert "Mixing old legacy c/py_transforms and new unified transforms is not allowed" in str(error_info.value)


def test_compose_legacy_text_mix_transforms():
    """
    Feature: Compose op
    Description: Test legacy Compose op containing text transforms plus mixing of new unified transforms
    Expectation: Dataset pipeline runs successfully for valid input and results verified.
    """

    my_data = ds.TextFileDataset(DATA_DIR_VOCAB, shuffle=False)

    # Use both Text Lookup op and Data Transforms PadEnd op in transforms list for Compose
    vocab = text.Vocab.from_dataset(my_data, "text", freq_range=None, top_k=None,
                                    special_tokens=["<pad>", "<unk>"],
                                    special_first=True)
    lookup_op = text.Lookup(vocab, "<unk>")

    # Test#1 - legacy Compose C++ op with new unified data transform
    padend_unified_op = data_trans.PadEnd([100], pad_value=vocab.tokens_to_ids('<pad>'))
    compose_op = c_transforms.Compose(transforms=[lookup_op, padend_unified_op])
    my_data = my_data.map(operations=compose_op, input_columns=["text"])
    res = []
    for d in my_data.create_dict_iterator(num_epochs=1, output_numpy=True):
        res.append(d["text"].item())
    assert res == [4, 5, 3, 6, 7, 2]


if __name__ == "__main__":
    test_compose()
    test_lambdas()
    test_c_py_compose_transforms_module()
    test_c_py_compose_vision_module(plot=True)
    test_py_transforms_with_c_vision()
    test_py_vision_with_c_transforms()
    test_compose_with_custom_function()
    test_compose_unified_mix_tranforms()
    test_compose_c_legacy_mix_tranforms()
    test_compose_py_legacy_mix_tranforms()
    test_compose_text_and_data_transforms()
    test_compose_unified_text_mix_transforms()
    test_compose_legacy_text_mix_transforms()
