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
from PIL import Image
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
from mindspore.dataset import transforms
from mindspore.dataset import vision

from util import visualize_list, save_and_check_md5_pil, config_get_set_seed, config_get_set_num_parallel_workers

GENERATE_GOLDEN = False


def test_compose():
    """
    Feature: Compose Op
    Description: Test Compose op, C++ implementation and Python implementation, valid and invalid input
    Expectation: For valid input, dataset pipeline runs successfully, and results are verified.
                 For invalid input, error message is verified.
    """
    original_seed = config_get_set_seed(0)

    def test_config(arr, op_list):
        try:
            data = ds.NumpySlicesDataset(
                arr, column_names="col", shuffle=False)
            data = data.map(input_columns=["col"], operations=op_list)
            res = []
            for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                res.append(i["col"].tolist())
            return res
        except (TypeError, ValueError) as e:
            return str(e)

    # Test simple compose with only 1 op, this would generate a warning
    assert test_config([[1, 0], [3, 4]], transforms.Compose(
        [transforms.Fill(2)])) == [[2, 2], [2, 2]]

    # Test 1 column -> 2 columns -> 1 -> 2 -> 1
    assert test_config([[1, 0]],
                       transforms.Compose(
                           [transforms.Duplicate(), transforms.Concatenate(), transforms.Duplicate(),
                            transforms.Concatenate()])) \
        == [[1, 0] * 4]

    # Test one Python transform followed by a C++ transform. Type after OneHot is a float (mixed use-case)
    assert test_config([1, 0],
                       transforms.Compose([transforms.OneHot(2), transforms.TypeCast(mstype.int32)])) \
        == [[0, 1], [1, 0]]

    # Test exceptions.
    with pytest.raises(TypeError) as error_info:
        transforms.Compose([1, transforms.TypeCast(mstype.int32)])
    assert "transforms[0] is neither a transforms op (TensorOperation) nor a callable pyfunc." in str(
        error_info.value)

    # Test empty op list
    with pytest.raises(ValueError) as error_info:
        test_config([1, 0], transforms.Compose([]))
    assert "transforms list can not be empty." in str(error_info.value)

    # Test Python compose op
    assert test_config([1, 0], transforms.Compose(
        [transforms.OneHot(2)])) == [[0, 1], [1, 0]]
    assert test_config([1, 0], transforms.Compose([transforms.OneHot(2), (lambda x: x + x)])) == [[0, 2],
                                                                                                  [2, 0]]

    # Test nested Python compose op
    assert test_config([1, 0],
                       transforms.Compose([transforms.Compose([transforms.OneHot(2)]), (lambda x: x + x)])) \
        == [[0, 2], [2, 0]]

    # Test passing a list of Python implementations without Compose wrapper
    assert test_config([1, 0],
                       [transforms.Compose([transforms.OneHot(2)]), (lambda x: x + x)]) \
        == [[0, 2], [2, 0]]
    assert test_config([1, 0], [transforms.OneHot(
        2), (lambda x: x + x)]) == [[0, 2], [2, 0]]

    # Test a non callable function
    with pytest.raises(TypeError) as error_info:
        transforms.Compose([1])
    assert "transforms[0] is neither a transforms op (TensorOperation) nor a callable pyfunc." in str(
        error_info.value)

    # Test empty Python implementation list
    with pytest.raises(ValueError) as error_info:
        test_config([1, 0], transforms.Compose([]))
    assert "transforms list can not be empty." in str(error_info.value)

    # Pass in extra brackets
    with pytest.raises(RuntimeError) as error_info:
        transforms.Compose([(lambda x: x + x)])()
    assert "Input Tensor is not valid." in str(error_info.value)

    # Restore configuration
    ds.config.set_seed(original_seed)


def test_lambdas():
    """
    Feature: Compose op
    Description: Test multi column Python Compose op
    Expectation: Output is equal to the expected value
    """
    original_seed = config_get_set_seed(0)

    def test_config(arr, input_columns, output_cols, op_list):
        data = ds.NumpySlicesDataset(
            arr, column_names=input_columns, shuffle=False)
        data = data.map(operations=op_list, input_columns=input_columns, output_columns=output_cols)
        res = []
        for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
            for col_name in output_cols:
                res.append(i[col_name].tolist())
        return res

    arr = ([[1]], [[3]])

    assert test_config(arr, ["col0", "col1"], ["a"],
                       transforms.Compose([(lambda x, y: x)])) == [[1]]
    assert test_config(arr, ["col0", "col1"], ["a"], transforms.Compose(
        [lambda x, y: x, lambda x: x])) == [[1]]
    assert test_config(arr, ["col0", "col1"], ["a", "b"],
                       transforms.Compose([lambda x, y: x, lambda x: (x, x * 2)])) == \
        [[1], [2]]
    assert test_config(arr, ["col0", "col1"], ["a", "b"],
                       [lambda x, y: (x, x + y), lambda x, y: (x, y * 2)]) == [[1], [8]]

    # Restore configuration
    ds.config.set_seed(original_seed)


def test_c_py_compose_transforms_module():
    """
    Feature: Compose op
    Description: Test combining Cpp and Python transformations
    Expectation: Output is equal to the expected value
    """
    original_seed = config_get_set_seed(0)

    def test_config(arr, input_columns, output_cols, op_list):
        data = ds.NumpySlicesDataset(
            arr, column_names=input_columns, shuffle=False)
        data = data.map(operations=op_list, input_columns=input_columns, output_columns=output_cols)
        res = []
        for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
            for col_name in output_cols:
                res.append(i[col_name].tolist())
        return res

    arr = [1, 0]
    assert test_config(arr, ["cols"], ["cols"],
                       [transforms.OneHot(2), transforms.Mask(transforms.Relational.EQ, 1)]) == \
        [[False, True],
         [True, False]]
    assert test_config(arr, ["cols"], ["cols"],
                       [transforms.OneHot(2), (lambda x: x + x), transforms.Fill(1)]) \
        == [[1, 1], [1, 1]]
    assert test_config(arr, ["cols"], ["cols"],
                       [transforms.OneHot(2), (lambda x: x + x), transforms.Fill(1), (lambda x: x + x)]) \
        == [[2, 2], [2, 2]]
    assert test_config([[1, 3]], ["cols"], ["cols"],
                       [transforms.PadEnd([3], -1), (lambda x: x + x)]) \
        == [[2, 6, -2]]

    arr = ([[1]], [[3]])
    assert test_config(arr, ["col0", "col1"], ["a"], [
        (lambda x, y: x + y), transforms.PadEnd([2], -1)]) == [[4, -1]]

    # Restore configuration
    ds.config.set_seed(original_seed)


def test_c_py_compose_vision_module(plot=False, run_golden=True):
    """
    Feature: Compose Op
    Description: Test Compose op combining Python and C++ vision transforms
    Expectation: Dataset pipeline runs successfully, results are visually verified and md5 results are verified
    """
    original_seed = config_get_set_seed(10)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    def test_config(plot, file_name, op_list):
        data_dir = "../data/dataset/testImageNetData/train/"
        data1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data1 = data1.map(operations=op_list, input_columns=["image"])
        data2 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data2 = data2.map(operations=vision.Decode(), input_columns=["image"])
        original_images = []
        transformed_images = []

        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            transformed_images.append(item["image"])
        for item in data2.create_dict_iterator(num_epochs=1, output_numpy=True):
            original_images.append(item["image"])

        if run_golden:
            # Compare with expected md5 from images
            save_and_check_md5_pil(
                data1, file_name, generate_golden=GENERATE_GOLDEN)

        if plot:
            visualize_list(original_images, transformed_images)

    test_config(op_list=[vision.Decode(),
                         vision.ToPIL(),
                         vision.Resize((224, 224)),
                         vision.ToNumpy()],
                plot=plot, file_name="compose_c_py_1.npz")

    test_config(op_list=[vision.Decode(),
                         vision.Resize((224, 244)),
                         vision.ToPIL(),
                         vision.ToNumpy(),
                         vision.Resize((24, 24))],
                plot=plot, file_name="compose_c_py_2.npz")

    test_config(op_list=[vision.Decode(True),
                         vision.Resize((224, 224)),
                         np.array,
                         vision.RandomColor()],
                plot=plot, file_name="compose_c_py_3.npz")

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers((original_num_parallel_workers))


def test_vision_with_transforms():
    """
    Feature: Data transforms and vision ops
    Description: Test (Python implementation) vision operations with C++ implementation transforms operations
    Expectation: Valid input succeeds. Invalid input fails.
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
    output_arr = test_config([vision.Decode(True),
                              vision.CenterCrop((2)), vision.ToNumpy(),
                              transforms.Mask(transforms.Relational.GE, 100)])

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
    output_arr = test_config([vision.Decode(True),
                              vision.CenterCrop((4)), vision.ToNumpy(),
                              transforms.Fill(10)])

    exp_arr = [np.ones((4, 4, 3)) * 10] * 2
    for exp_a, output in zip(exp_arr, output_arr):
        np.testing.assert_array_equal(exp_a, output)

    # Test with Concatenate Op, which will raise an error since ConcatenateOp only supports rank 1 tensors.
    with pytest.raises(RuntimeError) as error_info:
        test_config([vision.Decode(True),
                     vision.CenterCrop((2)), vision.ToNumpy(),
                     transforms.Concatenate(0)])
    assert "only 1D input supported" in str(error_info.value)

    # Restore configuration
    ds.config.set_seed(original_seed)


def test_compose_with_custom_function():
    """
    Feature: Compose op
    Description: Test Python Compose op with custom function
    Expectation: Output is equal to the expected value
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

    data = ds.NumpySlicesDataset([[1, 2]], column_names=[
        "col0"], shuffle=False)
    data = data.map(input_columns=["col0"], operations=op_list)
    #

    res = []
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        res.append(i["col0"].tolist())
    assert res == [[[3, 6], [9, 36]]]


def test_compose_mix_ctrans_pytrans():
    """
    Feature: Compose op
    Description: Test Python Compose op mix with c transforms and py transforms
    Expectation: No error occurred
    """
    transform_img = transforms.Compose([
        vision.RandomCropDecodeResize(size=(255, 255), scale=(0.2, 1.0)),
        transforms.RandomApply([vision.RandomColorAdjust(0.4, 0.4, 0.4, 0.1)], prob=0.8),
        vision.ToPIL(),
        vision.RandomGrayscale(prob=0.2),
    ])

    dataset = ds.ImageFolderDataset("../data/dataset/testPK/data/", shuffle=False, num_samples=2)
    dataset = dataset.map(transform_img)

    # pipeline mode
    for _ in dataset:
        pass

    # eager mode
    img = np.fromfile("../data/dataset/testPK/data/class1/0.jpg", np.uint8)
    img = transform_img(img)[0]
    assert isinstance(img, Image.Image)


def test_compose_embed_compose():
    """
    Feature: Compose op
    Description: Test Python Compose op embed compose when reuse the object in EXECUTORS_LIST
    Expectation: No error occurred
    """
    class ComposeB():
        def __init__(self):
            self.transforms = ds.transforms.Compose([vision.HorizontalFlip()])

        def __call__(self, img):
            return Image.fromarray(self.transforms(img))

    class ComposeC():
        def __init__(self):
            self.transforms = ds.transforms.Compose([vision.Resize((30, 30)), vision.CenterCrop((20, 20))])

        def __call__(self, img):
            return Image.fromarray(self.transforms(img))

    class ComposeA():
        def __init__(self):
            self.transforms = ds.transforms.Compose([
                ComposeB(),
                ComposeC(),
            ])

        def __call__(self, data):
            img = data["image"]
            img = self.transforms(img)
            data["image"] = img
            return data

    img = {"image": np.ones((31, 102, 3), dtype=np.uint8)}
    img = ComposeA()(img)


if __name__ == "__main__":
    test_compose()
    test_lambdas()
    test_c_py_compose_transforms_module()
    test_c_py_compose_vision_module(plot=True)
    test_vision_with_transforms()
    test_compose_with_custom_function()
    test_compose_embed_compose()
