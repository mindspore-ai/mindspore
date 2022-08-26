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
"""
Testing Normalize op in DE
"""
import numpy as np
from PIL import Image
import mindspore.dataset as ds
import mindspore.dataset.transforms
import mindspore.dataset.vision as vision
from mindspore import log as logger
from util import diff_mse, save_and_check_md5, save_and_check_md5_pil, visualize_image

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

GENERATE_GOLDEN = False


def normalize_np(image, mean, std):
    """
    Apply the Normalization
    """
    #  DE decodes the image in RGB by default, hence
    #  the values here are in RGB
    image = np.array(image, np.float32)
    image = image - np.array(mean)
    image = image * (1.0 / np.array(std))
    return image


def util_test_normalize(mean, std, add_to_pil):
    """
    Utility function for testing Normalize. Input arguments are given by other tests
    """
    if not add_to_pil:
        # define map operations
        decode_op = vision.Decode()
        normalize_op = vision.Normalize(mean, std, True)
        # Generate dataset
        data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
        data = data.map(operations=decode_op, input_columns=["image"])
        data = data.map(operations=normalize_op, input_columns=["image"])
    else:
        # define map operations
        transforms = [
            vision.Decode(True),
            vision.ToTensor(),
            vision.Normalize(mean, std, False)
        ]
        transform = mindspore.dataset.transforms.Compose(transforms)
        # Generate dataset
        data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
        data = data.map(operations=transform, input_columns=["image"])
    return data


def util_test_normalize_grayscale(num_output_channels, mean, std):
    """
    Utility function for testing Normalize. Input arguments are given by other tests
    """
    transforms = [
        vision.Decode(True),
        vision.Grayscale(num_output_channels),
        vision.ToTensor(),
        vision.Normalize(mean, std, False)
    ]
    transform = mindspore.dataset.transforms.Compose(transforms)
    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data = data.map(operations=transform, input_columns=["image"])
    return data


def test_normalize_op_hwc(plot=False):
    """
    Feature: Normalize op
    Description: Test Normalize with Decode versus NumPy comparison
    Expectation: Test succeeds. MSE difference is negligible.
    """
    logger.info("Test Normalize in with hwc")
    mean = [121.0, 115.0, 100.0]
    std = [70.0, 68.0, 71.0]
    # define map operations
    decode_op = vision.Decode()
    normalize_op = vision.Normalize(mean, std, True)

    #  First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=normalize_op, input_columns=["image"])

    #  Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=decode_op, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_de_normalized = item1["image"]
        image_original = item2["image"]
        image_np_normalized = normalize_np(image_original, mean, std)
        np.testing.assert_almost_equal(image_de_normalized, image_np_normalized, 2)
        mse = diff_mse(image_de_normalized, image_np_normalized)
        logger.info("image_{}, mse: {}".format(num_iter + 1, mse))
        if plot:
            visualize_image(image_original, image_de_normalized, mse, image_np_normalized)
        num_iter += 1


def test_normalize_op_chw(plot=False):
    """
    Feature: Normalize op
    Description: Test Normalize with CHW input, Decode(to_pil=True) & ToTensor versus NumPy comparison
    Expectation: Test succeeds. MSE difference is negligible.
    """
    logger.info("Test Normalize with chw")
    mean = [0.475, 0.45, 0.392]
    std = [0.275, 0.267, 0.278]
    # define map operations
    transforms = [
        vision.Decode(True),
        vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.Compose(transforms)
    normalize_op = vision.Normalize(mean, std, False)

    #  First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=transform, input_columns=["image"])
    data1 = data1.map(operations=normalize_op, input_columns=["image"])

    #  Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=transform, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_de_normalized = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_np_normalized = (normalize_np(item2["image"].transpose(1, 2, 0), mean, std) * 255).astype(np.uint8)
        mse = diff_mse(image_de_normalized, image_np_normalized)
        logger.info("image_{}, mse: {}".format(num_iter + 1, mse))
        assert mse < 0.01
        if plot:
            image_original = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
            visualize_image(image_original, image_de_normalized, mse, image_np_normalized)
        num_iter += 1


def test_normalize_op_video():
    """
    Feature: Normalize op
    Description: Test NormalizeOp in Cpp transformation with 4 dimension input,
                 where the input tensor is (..., T, H, W, C)
    Expectation: The dataset is processed successfully
    """
    logger.info("Test NormalizeOp in cpp transformations with 4 dimension input")
    mean = [121.0, 115.0, 100.0]
    std = [70.0, 68.0, 71.0]
    input_np_original = np.array([[87, 88, 232, 239],
                                  [11, 229, 22, 79],
                                  [250, 20, 173, 213]], dtype=np.float32)
    expect_output = np.array([[-1.0714285, -1.3088236, 0.11267605, 1.7142857],
                              [-0.35211268, -0.55714285, 1.0735294, 0.52112675],
                              [-0.27941176, 0.43661973, 1.3428571, -0.9411765]], dtype=np.float32)
    shape = (2, 2, 1, 3)
    input_np_original = input_np_original.reshape(shape)
    expect_output = expect_output.reshape(shape)

    # define operations
    normalize_op = vision.Normalize(mean, std, True)

    # doing the Normalization
    vidio_de_normalized = normalize_op(input_np_original)

    mse = diff_mse(vidio_de_normalized, expect_output)
    assert mse < 0.01


def test_normalize_op_5d():
    """
    Feature: Normalize op
    Description: Test NormalizeOp in Cpp transformation with 5 dim input, where the input tensor is (..., T, H, W, C)
    Expectation: The dataset is processed successfully
    """
    logger.info("Test NormalizeOp in cpp transformations with 5 dimension input")
    mean = [121.0, 115.0, 100.0]
    std = [70.0, 68.0, 71.0]
    input_np_original = np.array([[87, 88, 232, 239],
                                  [11, 229, 22, 79],
                                  [250, 20, 173, 213]], dtype=np.float32)
    expect_output = np.array([[-1.0714285, -1.3088236, 0.11267605, 1.7142857],
                              [-0.35211268, -0.55714285, 1.0735294, 0.52112675],
                              [-0.27941176, 0.43661973, 1.3428571, -0.9411765]], dtype=np.float32)
    shape = (2, 1, 2, 1, 3)
    input_np_original = input_np_original.reshape(shape)
    expect_output = expect_output.reshape(shape)

    # define operations
    normalize_op = vision.Normalize(mean, std, True)

    # doing the Normalization
    vidio_de_normalized = normalize_op(input_np_original)

    mse = diff_mse(vidio_de_normalized, expect_output)
    assert mse < 0.01


def test_decode_op():
    """
    Feature: Decode op
    Description: Test Decode op basic usage
    Expectation: Dataset pipeline runs successfully and results are verified
    """
    logger.info("Test Decode")

    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image", "label"], num_parallel_workers=1,
                               shuffle=False)

    # define map operations
    decode_op = vision.Decode()

    # apply map operations on images
    data1 = data1.map(operations=decode_op, input_columns=["image"])

    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1):
        logger.info("Looping inside iterator {}".format(num_iter))
        _ = item["image"]
        num_iter += 1


def test_decode_normalize_op():
    """
    Feature: Decode op and Normalize op
    Description: Test Decode op followed by Normalize op in one map
    Expectation: Dataset pipeline runs successfully and results are verified
    """
    logger.info("Test [Decode, Normalize] in one Map")

    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image", "label"], num_parallel_workers=1,
                               shuffle=False)

    # define map operations
    decode_op = vision.Decode()
    normalize_op = vision.Normalize([121.0, 115.0, 100.0], [70.0, 68.0, 71.0], True)

    # apply map operations on images
    data1 = data1.map(operations=[decode_op, normalize_op], input_columns=["image"])

    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1):
        logger.info("Looping inside iterator {}".format(num_iter))
        _ = item["image"]
        num_iter += 1


def test_normalize_md5_01():
    """
    Feature: Normalize Op
    Description: Test Normalize op with md5 check with valid mean and std
    Expectation: Passes the md5 check test
    """
    logger.info("test_normalize_md5_01")
    data_c = util_test_normalize([121.0, 115.0, 100.0], [70.0, 68.0, 71.0], False)
    data_py = util_test_normalize([0.475, 0.45, 0.392], [0.275, 0.267, 0.278], True)

    # check results with md5 comparison
    filename1 = "normalize_01_c_result.npz"
    filename2 = "normalize_01_to_pil_result.npz"
    save_and_check_md5(data_c, filename1, generate_golden=GENERATE_GOLDEN)
    save_and_check_md5_pil(data_py, filename2, generate_golden=GENERATE_GOLDEN)


def test_normalize_md5_02():
    """
    Feature: Normalize Op
    Description: Test Normalize op with md5 check with len(mean)=len(std)=1 with RBG images
    Expectation: Passes the md5 check test
    """
    logger.info("test_normalize_md5_02")
    data_py = util_test_normalize([0.475], [0.275], True)

    # check results with md5 comparison
    filename2 = "normalize_02_to_pil_result.npz"
    save_and_check_md5_pil(data_py, filename2, generate_golden=GENERATE_GOLDEN)


def test_normalize_exception_unequal_size_1():
    """
    Feature: Normalize op
    Description: Test Normalize with error input: len(mean) != len(std)
    Expectation: ValueError raised
    """
    logger.info("test_normalize_exception_unequal_size_1")
    try:
        _ = vision.Normalize([100, 250, 125], [50, 50, 75, 75])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Length of mean and std must be equal."


def test_normalize_exception_out_of_range():
    """
    Feature: Normalize op
    Description: Test Normalize with error input: mean, std out of range
    Expectation: ValueError raised
    """
    logger.info("test_normalize_exception_out_of_range")
    try:
        _ = vision.Normalize([256, 250, 125], [50, 75, 75])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "not within the required interval" in str(e)
    try:
        _ = vision.Normalize([255, 250, 125], [0, 75, 75])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "not within the required interval" in str(e)


def test_normalize_exception_unequal_size_2():
    """
    Feature: Normalize op
    Description: Test Normalize with error input: len(mean) != len(std)
    Expectation: ValueError raised
    """
    logger.info("test_normalize_exception_unequal_size_2")
    try:
        _ = vision.Normalize([0.50, 0.30, 0.75], [0.18, 0.32, 0.71, 0.72], False)
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Length of mean and std must be equal."


def test_normalize_exception_invalid_size():
    """
    Feature: Normalize op
    Description: Test Normalize with error input: len(mean)=len(std)=2
    Expectation: RuntimeError raised
    """
    logger.info("test_normalize_exception_invalid_size")
    data = util_test_normalize([0.75, 0.25], [0.18, 0.32], False)
    try:
        _ = data.create_dict_iterator(num_epochs=1).__next__()
    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Normalize: number of channels does not match the size of mean and std vectors" in str(e)


def test_normalize_exception_invalid_range():
    """
    Feature: Normalize op
    Description: Test Normalize with error input: value is not in range [0,1]
    Expectation: ValueError raised
    """
    logger.info("test_normalize_exception_invalid_range")
    try:
        _ = vision.Normalize([0.75, 1.25, 0.5], [0.1, 0.18, 1.32], False)
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input mean_value is not within the required interval of [0.0, 1.0]." in str(e)


def test_normalize_grayscale_md5_01():
    """
    Feature: Normalize Op
    Description: Test Normalize op with md5 check with len(mean)=len(std)=1 with 1 channel grayscale images
    Expectation: Passes the md5 check test
    """
    logger.info("test_normalize_grayscale_md5_01")
    data = util_test_normalize_grayscale(1, [0.5], [0.175])
    # check results with md5 comparison
    filename = "normalize_03_to_pil_result.npz"
    save_and_check_md5_pil(data, filename, generate_golden=GENERATE_GOLDEN)


def test_normalize_grayscale_md5_02():
    """
    Feature: Normalize Op
    Description: Test Normalize op with md5 check with len(mean)=len(std)=3 with 3 channels grayscale images
    Expectation: Passes the md5 check test
    """
    logger.info("test_normalize_grayscale_md5_02")
    data = util_test_normalize_grayscale(3, [0.5, 0.5, 0.5], [0.175, 0.235, 0.512])
    # check results with md5 comparison
    filename = "normalize_04_to_pil_result.npz"
    save_and_check_md5_pil(data, filename, generate_golden=GENERATE_GOLDEN)


def test_normalize_grayscale_exception():
    """
    Feature: Normalize Op
    Description: Test Normalize op with md5 check with len(mean)=len(std)=3 with 1 channel grayscale images
    Expectation: Error is raised as expected
    """
    logger.info("test_normalize_grayscale_exception")
    try:
        _ = util_test_normalize_grayscale(1, [0.5, 0.5, 0.5], [0.175, 0.235, 0.512])
    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input is not within the required range" in str(e)


def test_multiple_channels():
    """
    Feature: Normalize Op
    Description: Test Normalize op with multiple channels
    Expectation: Output is equal to the expected output
    """
    logger.info("test_multiple_channels")

    def util_test(item, mean, std):
        data = ds.NumpySlicesDataset([item], shuffle=False)
        data = data.map(vision.Normalize(mean, std, True))
        for d in data.create_tuple_iterator(num_epochs=1, output_numpy=True):
            actual = d[0]
            mean = np.array(mean, dtype=item.dtype)
            std = np.array(std, dtype=item.dtype)
            expected = item
            if len(item.shape) != 1 and len(mean) == 1:
                mean = [mean[0]] * expected.shape[-1]
                std = [std[0]] * expected.shape[-1]
            if len(item.shape) == 2:
                expected = np.expand_dims(expected, 2)
            for c in range(expected.shape[-1]):
                expected[:, :, c] = (expected[:, :, c] - mean[c]) / std[c]
            expected = expected.squeeze()

            np.testing.assert_almost_equal(actual, expected, decimal=6)

    util_test(np.ones(shape=[2, 2, 3]), mean=[0.5, 0.6, 0.7], std=[0.1, 0.2, 0.3])
    util_test(np.ones(shape=[20, 45, 3]) * 1.3, mean=[0.5, 0.6, 0.7], std=[0.1, 0.2, 0.3])
    util_test(np.ones(shape=[20, 45, 4]) * 1.3, mean=[0.5, 0.6, 0.7, 0.8], std=[0.1, 0.2, 0.3, 0.4])
    util_test(np.ones(shape=[2, 2]), mean=[0.5], std=[0.1])
    util_test(np.ones(shape=[2, 2, 5]), mean=[0.5], std=[0.1])
    util_test(np.ones(shape=[6, 6, 129]), mean=[0.5] * 129, std=[0.1] * 129)
    util_test(np.ones(shape=[6, 6, 129]), mean=[0.5], std=[0.1])


def test_normalize_eager_hwc():
    """
    Feature: Normalize op
    Description: Test eager support for Normalize Cpp implementation with HWC input
    Expectation: Receive non-None output image from op
    """
    img_in = Image.open("../data/dataset/apple.jpg").convert("RGB")
    mean_vec = [1, 100, 255]
    std_vec = [1, 20, 255]
    normalize_op = vision.Normalize(mean=mean_vec, std=std_vec)
    img_out = normalize_op(img_in)
    assert img_out is not None


def test_normalize_eager_chw():
    """
    Feature: Normalize op
    Description: Test eager support for Normalize Cpp implementation with CHW input
    Expectation: Receive non-None output image from op
    """
    img_in = Image.open("../data/dataset/apple.jpg").convert("RGB")
    img_in = vision.ToTensor()(img_in)
    mean_vec = [0.1, 0.5, 1.0]
    std_vec = [0.1, 0.4, 1.0]
    normalize_op = vision.Normalize(mean=mean_vec, std=std_vec, is_hwc=False)
    img_out = normalize_op(img_in)
    assert img_out is not None


def test_normalize_op_comp_chw():
    """
    Feature: Normalize op
    Description: Test Normalize with CHW input, Decode(to_pil=True) & ToTensor versus Decode(to_pil=False) & HWC2CHW
                 comparison.
    Expectation: Test succeeds. MSE difference is negligible.
    """
    logger.info("Test Normalize with CHW input")
    mean = [0.475, 0.45, 0.392]
    std = [0.275, 0.267, 0.278]
    # define map operations
    transforms = [
        vision.Decode(True),
        vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.Compose(transforms)
    normalize_op = vision.Normalize(mean, std, False)

    #  First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=transform, input_columns=["image"])
    data1 = data1.map(operations=normalize_op, input_columns=["image"])

    #  Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=vision.Decode(), input_columns=["image"])
    data2 = data2.map(operations=vision.HWC2CHW(), input_columns=["image"])
    data2 = data2.map(operations=vision.Normalize(mean, std, False), input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_de_normalized = item1["image"]
        image_np_normalized = item2["image"] / 255
        mse = diff_mse(image_de_normalized, image_np_normalized)
        logger.info("image_{}, mse: {}".format(num_iter + 1, mse))
        assert mse < 0.01
        num_iter += 1


if __name__ == "__main__":
    test_decode_op()
    test_decode_normalize_op()
    test_normalize_op_hwc(plot=True)
    test_normalize_op_chw(plot=True)
    test_normalize_op_video()
    test_normalize_op_5d()
    test_normalize_md5_01()
    test_normalize_md5_02()
    test_normalize_exception_unequal_size_1()
    test_normalize_exception_out_of_range()
    test_normalize_exception_unequal_size_2()
    test_normalize_exception_invalid_size()
    test_normalize_exception_invalid_range()
    test_normalize_grayscale_md5_01()
    test_normalize_grayscale_md5_02()
    test_normalize_grayscale_exception()
    test_multiple_channels()
    test_normalize_eager_hwc()
    test_normalize_eager_chw()
    test_normalize_op_comp_chw()
