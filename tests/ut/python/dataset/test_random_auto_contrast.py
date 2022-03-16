# Copyright 2021 Huawei Technologies Co., Ltd
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
Testing RandomAutoContrast op in DE
"""
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as c_vision
from mindspore import log as logger
from util import visualize_list, visualize_image, diff_mse

image_file = "../data/dataset/testImageNetData/train/class1/1_1.jpg"
data_dir = "../data/dataset/testImageNetData/train/"


def test_random_auto_contrast_pipeline(plot=False):
    """
    Test RandomAutoContrast pipeline
    """
    logger.info("Test RandomAutoContrast pipeline")

    # Original Images
    data_set = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    transforms_original = [c_vision.Decode(), c_vision.Resize(size=[224, 224])]
    ds_original = data_set.map(operations=transforms_original, input_columns="image")
    ds_original = ds_original.batch(512)

    for idx, (image, _) in enumerate(ds_original):
        if idx == 0:
            images_original = image.asnumpy()
        else:
            images_original = np.append(images_original,
                                        image.asnumpy(),
                                        axis=0)

    # Randomly Automatically Contrasted Images
    data_set1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    transform_random_auto_contrast = [c_vision.Decode(),
                                      c_vision.Resize(size=[224, 224]),
                                      c_vision.RandomAutoContrast(prob=0.6)]
    ds_random_auto_contrast = data_set1.map(operations=transform_random_auto_contrast, input_columns="image")
    ds_random_auto_contrast = ds_random_auto_contrast.batch(512)
    for idx, (image, _) in enumerate(ds_random_auto_contrast):
        if idx == 0:
            images_random_auto_contrast = image.asnumpy()
        else:
            images_random_auto_contrast = np.append(images_random_auto_contrast,
                                                    image.asnumpy(),
                                                    axis=0)
    if plot:
        visualize_list(images_original, images_random_auto_contrast)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_random_auto_contrast[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))


def test_random_auto_contrast_eager():
    """
    Test RandomAutoContrast eager.
    """
    img = np.fromfile(image_file, dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img = c_vision.Decode()(img)
    img_auto_contrast = c_vision.AutoContrast(1.0, None)(img)
    img_random_auto_contrast = c_vision.RandomAutoContrast(1.0, None, 1.0)(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img_auto_contrast), img_random_auto_contrast.shape))

    assert img_auto_contrast.all() == img_random_auto_contrast.all()


def test_random_auto_contrast_comp(plot=False):
    """
    Test RandomAutoContrast op compared with AutoContrast op.
    """
    random_auto_contrast_op = c_vision.RandomAutoContrast(prob=1.0)
    auto_contrast_op = c_vision.AutoContrast()

    dataset1 = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    for item in dataset1.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = item['image']
    dataset1.map(operations=random_auto_contrast_op, input_columns=['image'])
    dataset2 = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    dataset2.map(operations=auto_contrast_op, input_columns=['image'])
    for item1, item2 in zip(dataset1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            dataset2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_random_auto_contrast = item1['image']
        image_auto_contrast = item2['image']

    mse = diff_mse(image_auto_contrast, image_random_auto_contrast)
    assert mse == 0
    logger.info("mse: {}".format(mse))
    if plot:
        visualize_image(image, image_random_auto_contrast, mse, image_auto_contrast)


def test_random_auto_contrast_invalid_prob():
    """
    Test RandomAutoContrast Op with invalid prob parameter.
    """
    logger.info("test_random_auto_contrast_invalid_prob")
    dataset = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    try:
        random_auto_contrast_op = c_vision.RandomAutoContrast(prob=1.5)
        dataset = dataset.map(operations=random_auto_contrast_op, input_columns=['image'])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input prob is not within the required interval of [0.0, 1.0]." in str(e)


def test_random_auto_contrast_invalid_ignore():
    """
    Test RandomAutoContrast Op with invalid ignore parameter.
    """
    logger.info("test_random_auto_contrast_invalid_ignore")
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data_set = data_set.map(operations=[c_vision.Decode(),
                                            c_vision.Resize((224, 224)),
                                            lambda img: np.array(img[:, :, 0])], input_columns=["image"])
        # invalid ignore
        data_set = data_set.map(operations=c_vision.RandomAutoContrast(ignore=255.5), input_columns="image")
    except TypeError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "Argument ignore with value 255.5 is not of type" in str(error)
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data_set = data_set.map(operations=[c_vision.Decode(), c_vision.Resize((224, 224)),
                                            lambda img: np.array(img[:, :, 0])], input_columns=["image"])
        # invalid ignore
        data_set = data_set.map(operations=c_vision.RandomAutoContrast(ignore=(10, 100)), input_columns="image")
    except TypeError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "Argument ignore with value (10,100) is not of type" in str(error)


def test_random_auto_contrast_invalid_cutoff():
    """
    Test RandomAutoContrast Op with invalid cutoff parameter.
    """
    logger.info("test_random_auto_contrast_invalid_cutoff")
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data_set = data_set.map(operations=[c_vision.Decode(),
                                            c_vision.Resize((224, 224)),
                                            lambda img: np.array(img[:, :, 0])], input_columns=["image"])
        # invalid cutoff
        data_set = data_set.map(operations=c_vision.RandomAutoContrast(cutoff=-10.0), input_columns="image")
    except ValueError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "Input cutoff is not within the required interval of [0, 50)." in str(error)
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data_set = data_set.map(operations=[c_vision.Decode(),
                                            c_vision.Resize((224, 224)),
                                            lambda img: np.array(img[:, :, 0])], input_columns=["image"])
        # invalid cutoff
        data_set = data_set.map(operations=c_vision.RandomAutoContrast(cutoff=120.0), input_columns="image")
    except ValueError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "Input cutoff is not within the required interval of [0, 50)." in str(error)


def test_random_auto_contrast_one_channel():
    """
    Feature: RandomAutoContrast
    Description: test with one channel images
    Expectation: raise errors as expected
    """
    logger.info("test_random_auto_contrast_one_channel")

    c_op = c_vision.RandomAutoContrast()

    try:
        data_set = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data_set = data_set.map(operations=[c_vision.Decode(), c_vision.Resize((224, 224)),
                                            lambda img: np.array(img[:, :, 0])], input_columns=["image"])

        data_set = data_set.map(operations=c_op, input_columns="image")

    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "image shape is incorrect, expected num of channels is 3." in str(e)


def test_random_auto_contrast_four_dim():
    """
    Feature: RandomAutoContrast
    Description: test with four dimension images
    Expectation: raise errors as expected
    """
    logger.info("test_random_auto_contrast_four_dim")

    c_op = c_vision.RandomAutoContrast()

    try:
        data_set = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data_set = data_set.map(operations=[c_vision.Decode(), c_vision.Resize((224, 224)),
                                            lambda img: np.array(img[2, 200, 10, 32])], input_columns=["image"])

        data_set = data_set.map(operations=c_op, input_columns="image")

    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "image shape is not <H,W,C>" in str(e)


def test_random_auto_contrast_invalid_input():
    """
    Feature: RandomAutoContrast
    Description: test with images in uint32 type
    Expectation: raise errors as expected
    """
    logger.info("test_random_invert_invalid_input")

    c_op = c_vision.RandomAutoContrast()

    try:
        data_set = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data_set = data_set.map(operations=[c_vision.Decode(), c_vision.Resize((224, 224)),
                                            lambda img: np.array(img[2, 32, 3], dtype=uint32)], input_columns=["image"])
        data_set = data_set.map(operations=c_op, input_columns="image")

    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Cannot convert from OpenCV type, unknown CV type" in str(e)


if __name__ == "__main__":
    test_random_auto_contrast_pipeline(plot=True)
    test_random_auto_contrast_eager()
    test_random_auto_contrast_comp(plot=True)
    test_random_auto_contrast_invalid_prob()
    test_random_auto_contrast_invalid_ignore()
    test_random_auto_contrast_invalid_cutoff()
    test_random_auto_contrast_one_channel()
    test_random_auto_contrast_four_dim()
    test_random_auto_contrast_invalid_input()
