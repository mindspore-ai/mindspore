# Copyright 2020 Huawei Technologies Co., Ltd
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

import matplotlib.pyplot as plt
import numpy as np

import mindspore.dataset.engine as de
import mindspore.dataset.transforms.vision.py_transforms as F
from mindspore import log as logger

DATA_DIR = "../data/dataset/testImageNetData/train/"


def visualize(image_original, image_random_sharpness):
    """
    visualizes the image using DE op and Numpy op
    """
    num = len(image_random_sharpness)
    for i in range(num):
        plt.subplot(2, num, i + 1)
        plt.imshow(image_original[i])
        plt.title("Original image")

        plt.subplot(2, num, i + num + 1)
        plt.imshow(image_random_sharpness[i])
        plt.title("DE Random Sharpness image")

    plt.show()


def test_random_sharpness(degrees=(0.1, 1.9), plot=False):
    """
    Test RandomSharpness
    """
    logger.info("Test RandomSharpness")

    # Original Images
    ds = de.ImageFolderDatasetV2(dataset_dir=DATA_DIR, shuffle=False)

    transforms_original = F.ComposeOp([F.Decode(),
                                       F.Resize((224, 224)),
                                       F.ToTensor()])

    ds_original = ds.map(input_columns="image",
                         operations=transforms_original())

    ds_original = ds_original.batch(512)

    for idx, (image, _) in enumerate(ds_original):
        if idx == 0:
            images_original = np.transpose(image, (0, 2, 3, 1))
        else:
            images_original = np.append(images_original,
                                        np.transpose(image, (0, 2, 3, 1)),
                                        axis=0)

            # Random Sharpness Adjusted Images
    ds = de.ImageFolderDatasetV2(dataset_dir=DATA_DIR, shuffle=False)

    transforms_random_sharpness = F.ComposeOp([F.Decode(),
                                               F.Resize((224, 224)),
                                               F.RandomSharpness(degrees=degrees),
                                               F.ToTensor()])

    ds_random_sharpness = ds.map(input_columns="image",
                                 operations=transforms_random_sharpness())

    ds_random_sharpness = ds_random_sharpness.batch(512)

    for idx, (image, _) in enumerate(ds_random_sharpness):
        if idx == 0:
            images_random_sharpness = np.transpose(image, (0, 2, 3, 1))
        else:
            images_random_sharpness = np.append(images_random_sharpness,
                                                np.transpose(image, (0, 2, 3, 1)),
                                                axis=0)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = np.mean((images_random_sharpness[i] - images_original[i]) ** 2)
    logger.info("MSE= {}".format(str(np.mean(mse))))

    if plot:
        visualize(images_original, images_random_sharpness)


if __name__ == "__main__":
    test_random_sharpness()
    test_random_sharpness(plot=True)
    test_random_sharpness(degrees=(0.5, 1.5), plot=True)
