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
Testing RandomAdjustSharpness in DE
"""
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as c_vision
from mindspore import log as logger
from util import visualize_list, visualize_image, diff_mse

image_file = "../data/dataset/testImageNetData/train/class1/1_1.jpg"
data_dir = "../data/dataset/testImageNetData/train/"


def test_random_adjust_sharpness_pipeline(plot=False):
    """
    Test RandomAdjustSharpness pipeline
    """
    logger.info("Test RandomAdjustSharpness pipeline")

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

    # Randomly Sharpness Adjusted Images
    data_set1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    transform_random_adjust_sharpness = [c_vision.Decode(),
                                         c_vision.Resize(size=[224, 224]),
                                         c_vision.RandomAdjustSharpness(2.0, 0.6)]
    ds_random_adjust_sharpness = data_set1.map(operations=transform_random_adjust_sharpness, input_columns="image")
    ds_random_adjust_sharpness = ds_random_adjust_sharpness.batch(512)
    for idx, (image, _) in enumerate(ds_random_adjust_sharpness):
        if idx == 0:
            images_random_adjust_sharpness = image.asnumpy()
        else:
            images_random_adjust_sharpness = np.append(images_random_adjust_sharpness,
                                                       image.asnumpy(),
                                                       axis=0)
    if plot:
        visualize_list(images_original, images_random_adjust_sharpness)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_random_adjust_sharpness[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))


def test_random_adjust_sharpness_eager():
    """
    Test RandomAdjustSharpness eager.
    """
    img = np.fromfile(image_file, dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img = c_vision.Decode()(img)
    img_sharped = c_vision.RandomSharpness((2.0, 2.0))(img)
    img_random_sharped = c_vision.RandomAdjustSharpness(2.0, 1.0)(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img_random_sharped), img_random_sharped.shape))

    assert img_random_sharped.all() == img_sharped.all()


def test_random_adjust_sharpness_comp(plot=False):
    """
    Test RandomAdjustSharpness op compared with Sharpness op.
    """
    random_adjust_sharpness_op = c_vision.RandomAdjustSharpness(degree=2.0, prob=1.0)
    sharpness_op = c_vision.RandomSharpness((2.0, 2.0))

    dataset1 = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    for item in dataset1.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = item['image']
    dataset1.map(operations=random_adjust_sharpness_op, input_columns=['image'])
    dataset2 = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    dataset2.map(operations=sharpness_op, input_columns=['image'])

    for item1, item2 in zip(dataset1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            dataset2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_random_sharpness = item1['image']
        image_sharpness = item2['image']

    mse = diff_mse(image_sharpness, image_random_sharpness)
    assert mse == 0
    logger.info("mse: {}".format(mse))
    if plot:
        visualize_image(image, image_random_sharpness, mse, image_sharpness)


def test_random_adjust_sharpness_invalid_prob():
    """
    Test invalid prob. prob out of range.
    """
    logger.info("test_random_adjust_sharpness_invalid_prob")
    dataset = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    try:
        random_adjust_sharpness_op = c_vision.RandomAdjustSharpness(2.0, 1.5)
        dataset = dataset.map(operations=random_adjust_sharpness_op, input_columns=['image'])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input prob is not within the required interval of [0.0, 1.0]." in str(e)


def test_random_adjust_sharpness_invalid_degree():
    """
    Test invalid prob. prob out of range.
    """
    logger.info("test_random_adjust_sharpness_invalid_prob")
    dataset = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    try:
        random_adjust_sharpness_op = c_vision.RandomAdjustSharpness(-1.0, 1.5)
        dataset = dataset.map(operations=random_adjust_sharpness_op, input_columns=['image'])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "interval" in str(e)


if __name__ == "__main__":
    test_random_adjust_sharpness_pipeline(plot=True)
    test_random_adjust_sharpness_eager()
    test_random_adjust_sharpness_comp(plot=True)
    test_random_adjust_sharpness_invalid_prob()
    test_random_adjust_sharpness_invalid_degree()
