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
Testing RandomEqualize op in DE
"""
import numpy as np

import mindspore.dataset as ds
from mindspore.dataset.vision.c_transforms import Decode, Resize, RandomEqualize, Equalize
from mindspore import log as logger
from util import visualize_list, visualize_image, diff_mse

image_file = "../data/dataset/testImageNetData/train/class1/1_1.jpg"
data_dir = "../data/dataset/testImageNetData/train/"


def test_random_equalize_pipeline(plot=False):
    """
    Test RandomEqualize pipeline
    """
    logger.info("Test RandomEqualize pipeline")

    # Original Images
    data_set = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    transforms_original = [Decode(), Resize(size=[224, 224])]
    ds_original = data_set.map(operations=transforms_original, input_columns="image")
    ds_original = ds_original.batch(512)

    for idx, (image, _) in enumerate(ds_original):
        if idx == 0:
            images_original = image.asnumpy()
        else:
            images_original = np.append(images_original,
                                        image.asnumpy(),
                                        axis=0)

    # Randomly Equalized Images
    data_set1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    transform_random_equalize = [Decode(), Resize(size=[224, 224]), RandomEqualize(0.6)]
    ds_random_equalize = data_set1.map(operations=transform_random_equalize, input_columns="image")
    ds_random_equalize = ds_random_equalize.batch(512)
    for idx, (image, _) in enumerate(ds_random_equalize):
        if idx == 0:
            images_random_equalize = image.asnumpy()
        else:
            images_random_equalize = np.append(images_random_equalize,
                                               image.asnumpy(),
                                               axis=0)
    if plot:
        visualize_list(images_original, images_random_equalize)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_random_equalize[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))


def test_random_equalize_eager():
    """
    Test RandomEqualize eager.
    """
    img = np.fromfile(image_file, dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img = Decode()(img)
    img_equalized = Equalize()(img)
    img_random_equalized = RandomEqualize(1.0)(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img_random_equalized), img_random_equalized.shape))

    assert img_random_equalized.all() == img_equalized.all()


def test_random_equalize_comp(plot=False):
    """
    Test RandomEqualize op compared with Equalize op.
    """
    random_equalize_op = RandomEqualize(prob=1.0)
    equalize_op = Equalize()

    dataset1 = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    for item in dataset1.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = item['image']
    dataset1.map(operations=random_equalize_op, input_columns=['image'])
    dataset2 = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    dataset2.map(operations=equalize_op, input_columns=['image'])
    for item1, item2 in zip(dataset1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            dataset2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_random_equalized = item1['image']
        image_equalized = item2['image']

    mse = diff_mse(image_equalized, image_random_equalized)
    assert mse == 0
    logger.info("mse: {}".format(mse))
    if plot:
        visualize_image(image, image_random_equalized, mse, image_equalized)


def test_random_equalize_invalid_prob():
    """
    Test eager. prob out of range.
    """
    logger.info("test_random_equalize_invalid_prob")
    dataset = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    try:
        random_equalize_op = RandomEqualize(1.5)
        dataset = dataset.map(operations=random_equalize_op, input_columns=['image'])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input prob is not within the required interval of [0.0, 1.0]." in str(e)


if __name__ == "__main__":
    test_random_equalize_pipeline(plot=True)
    test_random_equalize_eager()
    test_random_equalize_comp(plot=True)
    test_random_equalize_invalid_prob()
