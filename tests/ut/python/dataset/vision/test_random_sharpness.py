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
"""
Testing RandomSharpness op in DE
"""
import numpy as np
import os
import pytest
from PIL import Image

import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger
from util import (
    visualize_list,
    visualize_one_channel_dataset,
    diff_mse,
    save_and_check_md5,
    config_get_set_seed,
    config_get_set_num_parallel_workers,
)

DATA_DIR = "../data/dataset/testImageNetData/train/"
MNIST_DATA_DIR = "../data/dataset/testMnistData"
TEST_DATA_DATASET_FUNC ="../data/dataset/"
image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")

GENERATE_GOLDEN = False






def test_random_sharpness_c(degrees=(1.6, 1.6), plot=False):
    """
    Feature: RandomSharpness op
    Description: Test RandomSharpness with cpp op
    Expectation: The dataset is processed as expected
    """
    print(degrees)
    logger.info("Test RandomSharpness cpp op")

    # Original Images
    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transforms_original = [vision.Decode(),
                           vision.Resize((224, 224))]

    ds_original = data.map(operations=transforms_original, input_columns="image")

    ds_original = ds_original.batch(512)

    for idx, (image, _) in enumerate(ds_original.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        if idx == 0:
            images_original = image
        else:
            images_original = np.append(images_original,
                                        image,
                                        axis=0)

            # Random Sharpness Adjusted Images
    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    c_op = vision.RandomSharpness()
    if degrees is not None:
        c_op = vision.RandomSharpness(degrees)

    transforms_random_sharpness = [vision.Decode(),
                                   vision.Resize((224, 224)),
                                   c_op]

    ds_random_sharpness = data.map(operations=transforms_random_sharpness, input_columns="image")

    ds_random_sharpness = ds_random_sharpness.batch(512)

    for idx, (image, _) in enumerate(ds_random_sharpness.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        if idx == 0:
            images_random_sharpness = image
        else:
            images_random_sharpness = np.append(images_random_sharpness,
                                                image,
                                                axis=0)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_random_sharpness[i], images_original[i])

    logger.info("MSE= {}".format(str(np.mean(mse))))

    if plot:
        visualize_list(images_original, images_random_sharpness)


def test_random_sharpness_c_md5():
    """
    Feature: RandomSharpness op
    Description: Test RandomSharpness with cpp op with md5 comparison
    Expectation: The dataset is processed as expected
    """
    logger.info("Test RandomSharpness cpp op with md5 comparison")
    original_seed = config_get_set_seed(200)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # define map operations
    transforms = [
        vision.Decode(),
        vision.RandomSharpness((10.0, 15.0))
    ]

    #  Generate dataset
    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
    data = data.map(operations=transforms, input_columns=["image"])

    # check results with md5 comparison
    filename = "random_sharpness_cpp_01_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)




def test_random_sharpness_one_channel_c(degrees=(1.4, 1.4), plot=False):
    """
    Feature: RandomSharpness op
    Description: Test RandomSharpness with cpp op with one channel on MnistDataset (grayscale images)
    Expectation: The dataset is processed as expected
    """
    logger.info("Test RandomSharpness C Op With MNIST Dataset (Grayscale images)")

    c_op = vision.RandomSharpness()
    if degrees is not None:
        c_op = vision.RandomSharpness(degrees)
    # RandomSharpness Images
    data = ds.MnistDataset(dataset_dir=MNIST_DATA_DIR, num_samples=2, shuffle=False)
    ds_random_sharpness_c = data.map(operations=c_op, input_columns="image")
    # Original images
    data = ds.MnistDataset(dataset_dir=MNIST_DATA_DIR, num_samples=2, shuffle=False)

    images = []
    images_trans = []
    labels = []
    for _, (data_orig, data_trans) in enumerate(zip(data, ds_random_sharpness_c)):
        image_orig, label_orig = data_orig
        image_trans, _ = data_trans
        images.append(image_orig.asnumpy())
        labels.append(label_orig.asnumpy())
        images_trans.append(image_trans.asnumpy())

    if plot:
        visualize_one_channel_dataset(images, images_trans, labels)


def test_random_sharpness_invalid_params():
    """
    Feature: RandomSharpness op
    Description: Test RandomSharpness with invalid input parameters
    Expectation: Correct error is thrown as expected
    """
    logger.info("Test RandomSharpness with invalid input parameters.")
    try:
        data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data = data.map(operations=[vision.Decode(), vision.Resize((224, 224)),
                                    vision.RandomSharpness(10)], input_columns=["image"])
    except TypeError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "tuple" in str(error)

    try:
        data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data = data.map(operations=[vision.Decode(), vision.Resize((224, 224)),
                                    vision.RandomSharpness((-10, 10))], input_columns=["image"])
    except ValueError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "interval" in str(error)

    try:
        data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data = data.map(operations=[vision.Decode(), vision.Resize((224, 224)),
                                    vision.RandomSharpness((10, 5))], input_columns=["image"])
    except ValueError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "(min,max)" in str(error)


def test_random_sharpness_operation_01():
    """
    Feature: RandomSharpness operation
    Description: Testing the normal functionality of the RandomSharpness operator
    Expectation: The Output is equal to the expected output
    """
    # Test randomsharpness degrees=(0.1, 1)
    degrees = (0.1, 1)
    dataset = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=True)
    random_sharpness_op = vision.RandomSharpness(degrees=degrees)
    dataset = dataset.map(input_columns=["image"], operations=random_sharpness_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test randomsharpness degrees=(0, 1.5)
    degrees = (0, 1.5)
    dataset = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=True)
    random_sharpness_op = vision.RandomSharpness(degrees=degrees)
    dataset = dataset.map(input_columns=["image"], operations=random_sharpness_op)

    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test randomsharpness degrees=[0.5, 1.5]
    degrees = [0.5, 1.5]
    dataset = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=True)
    random_sharpness_op = vision.RandomSharpness(degrees=degrees)
    dataset = dataset.map(input_columns=["image"], operations=random_sharpness_op)

    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test randomsharpness degrees=[1, 1]
    degrees = [1, 1]
    dataset = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=True)
    random_sharpness_op = vision.RandomSharpness(degrees=degrees)
    dataset = dataset.map(input_columns=["image"], operations=random_sharpness_op)

    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test randomsharpness degrees = (0, 0.6)
    degrees = (0, 0.6)
    image = Image.open(image_jpg)
    random_sharpness_op = vision.RandomSharpness(degrees=degrees)
    out = random_sharpness_op(image)
    assert (np.array(image) != out).any()

    pil_op = vision.ToPIL()(image)
    out = random_sharpness_op(pil_op)
    assert (np.array(pil_op) != out).any()

    # Test randomsharpness degrees = (1.0, 1.0)
    degrees = (1.0, 1.0)
    image = Image.open(image_png)
    random_sharpness_op = vision.RandomSharpness(degrees=degrees)
    out = random_sharpness_op(image)
    assert (np.array(image) == out).all()

    pil_op = vision.ToPIL()(image)
    out = random_sharpness_op(pil_op)
    assert (np.array(pil_op) == out).all()

    # Test randomsharpness degrees = [0, 2.0]
    degrees = [0, 2.0]
    image = Image.open(image_bmp)
    random_sharpness_op = vision.RandomSharpness(degrees=degrees)
    _ = random_sharpness_op(image)


def test_random_sharpness_operation_02():
    """
    Feature: RandomSharpness operation
    Description: Testing the normal functionality of the RandomSharpness operator
    Expectation: The Output is equal to the expected output
    """
    # Test randomsharpness degrees=(0.6, 0.6)
    degrees = (0.6, 0.6)
    image = np.random.randn(256, 382, 1).astype(np.uint8)
    random_sharpness_op = vision.RandomSharpness(degrees=degrees)
    out = random_sharpness_op(image)
    assert (np.array(image) != out).any()

    # Test randomsharpness degrees = (0, 16777216)
    degrees = (0, 16777216)
    image = np.random.randn(256, 382, 1)
    random_sharpness_op = vision.RandomSharpness(degrees=degrees)
    out = random_sharpness_op(image)
    assert (np.array(image) != out).any()

    # Test randomsharpness image.shape = (500, 500)
    degrees = [3, 4]
    image = np.random.randint(0, 255, (500, 500)).astype(np.uint8)
    random_sharpness_op = vision.RandomSharpness(degrees=degrees)
    out = random_sharpness_op(image)
    assert (np.array(image) != out).any()

    pil_op = vision.ToPIL()(image)
    out = random_sharpness_op(pil_op)
    assert (np.array(pil_op) != out).any()

    # Test randomsharpness image.shape = (500, 500, 4)
    degrees = [0.01, 0.02]
    image = np.random.randint(0, 255, (500, 500, 4)).astype(np.uint8)
    random_sharpness_op = vision.RandomSharpness(degrees=degrees)
    out = random_sharpness_op(image)
    assert (np.array(image) != out).any()

    pil_op = vision.ToPIL()(image)
    out = random_sharpness_op(pil_op)
    assert (np.array(pil_op) != out).any()


def test_random_sharpness_exception_01():
    """
    Feature: RandomSharpness operation
    Description: Testing the RandomSharpness Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Test randomsharpness image = gif
    image = Image.open(image_gif)
    random_sharpness_op = vision.RandomSharpness()
    with pytest.raises(ValueError, match="cannot filter palette images"):
        _ = random_sharpness_op(image)

    # Test randomsharpness degrees=[-1, 2]
    degrees = [-1, 2]
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[0, 16777216\\]."):
        vision.RandomSharpness(degrees=degrees)

    # Test randomsharpness degrees=[1.5]
    degrees = [1.5]
    with pytest.raises(ValueError, match="degrees must be a sequence with length 2"):
        vision.RandomSharpness(degrees=degrees)

    # Test randomsharpness degrees=(0.1, 0.5, 1.9)
    degrees = (0.1, 0.5, 1.9)
    with pytest.raises(ValueError, match="degrees must be a sequence with length 2"):
        vision.RandomSharpness(degrees=degrees)

    # Test randomsharpness degrees=("0.1", "1.9")
    degrees = ("0.1", "1.9")
    with pytest.raises(TypeError, match=("Argument degrees\\[0\\] with value 0.1 is not of "
                                         "type \\[<class 'int'>, <class 'float'>\\]")):
        vision.RandomSharpness(degrees=degrees)

    # Test randomsharpness degrees=(1.5, 0.5)
    degrees = (1.5, 0.5)
    with pytest.raises(ValueError, match="degrees should be in \\(min,max\\) format. Got \\(max,min\\)"):
        vision.RandomSharpness(degrees=degrees)

    # Test randomsharpness image = 1d
    degrees = (0.01, 0.02)
    image = np.fromfile(image_jpg, dtype=np.uint8)
    random_sharpness_op = vision.RandomSharpness(degrees=degrees)
    with pytest.raises(RuntimeError, match="Sharpness: shape of input is not <H,W,C> or <H,W>, but got rank: 1"):
        random_sharpness_op(image)

    # Test randomsharpness image.shape = (500, 500, 3, 3)
    degrees = (0.01, 0.02)
    image = np.random.randint(0, 255, (500, 500, 3, 3)).astype(np.uint8)
    random_sharpness_op = vision.RandomSharpness(degrees=degrees)
    with pytest.raises(RuntimeError, match="Sharpness: shape of input is not <H,W,C> or <H,W>, but got rank: 4"):
        random_sharpness_op(image)

    # Test randomsharpness image = np
    degrees = (0.01, 0.02)
    image = np.random.randn(300, 300, 3).tolist()
    random_sharpness_op = vision.RandomSharpness(degrees=degrees)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        random_sharpness_op(image)

    # Test randomsharpness image = tuple
    degrees = (0.01, 0.02)
    image = tuple(np.random.randn(300, 300, 3).tolist())
    random_sharpness_op = vision.RandomSharpness(degrees=degrees)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'tuple'>."):
        random_sharpness_op(image)

    # Test randomsharpness image = Tensor
    degrees = (0.01, 0.02)
    image = ms.Tensor(np.random.randn(300, 300, 3))
    random_sharpness_op = vision.RandomSharpness(degrees=degrees)
    with pytest.raises(TypeError,
                       match="Input should be NumPy or PIL image, got <class 'mindspore.common.tensor.Tensor'>."):
        random_sharpness_op(image)

    # Test randomsharpness degrees={0.01, 0.02}
    degrees = {0.01, 0.02}
    with pytest.raises(TypeError, match="degrees must be either a tuple or a list."):
        vision.RandomSharpness(degrees=degrees)

    # Test randomsharpness degrees=np
    degrees = np.array([0.5, 0.6])
    with pytest.raises(TypeError, match="degrees must be either a tuple or a list."):
        vision.RandomSharpness(degrees=degrees)

    # Test randomsharpness degrees = 0.5
    degrees = 0.5
    with pytest.raises(TypeError, match="degrees must be either a tuple or a list."):
        vision.RandomSharpness(degrees=degrees)

    # Test randomsharpness degrees = None
    image = np.random.randn(300, 300, 3).astype(np.uint8)
    degrees = None
    with pytest.raises(TypeError, match="incompatible constructor arguments."):
        random_sharpness_op = vision.RandomSharpness(degrees=degrees)
        random_sharpness_op(image)

    with pytest.raises(TypeError, match="\'NoneType\' object is not subscriptable"):
        random_sharpness_op = vision.RandomSharpness(degrees=degrees)
        op_pil = vision.ToPIL()(image)
        random_sharpness_op(op_pil)

    # Test randomsharpness degrees = (10, 16777216.1)
    image = np.random.randn(300, 300, 3)
    degrees = (10, 16777216.1)
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[0, 16777216\\]."):
        random_sharpness_op = vision.RandomSharpness(degrees=degrees)
        random_sharpness_op(image)

    with pytest.raises(ValueError, match="Input is not within the required interval of \\[0, 16777216\\]."):
        random_sharpness_op = vision.RandomSharpness(degrees=degrees)
        op_pil = vision.ToPIL()(image)
        random_sharpness_op(op_pil)
