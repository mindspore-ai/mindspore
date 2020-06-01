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


def visualize(image_original, image_equalize):
    """
    visualizes the image using DE op and Numpy op
    """
    num = len(image_equalize)
    for i in range(num):
        plt.subplot(2, num, i + 1)
        plt.imshow(image_original[i])
        plt.title("Original image")

        plt.subplot(2, num, i + num + 1)
        plt.imshow(image_equalize[i])
        plt.title("DE Color Equalized image")

    plt.show()


def test_equalize(plot=False):
    """
    Test Equalize
    """
    logger.info("Test Equalize")

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

            # Color Equalized Images
    ds = de.ImageFolderDatasetV2(dataset_dir=DATA_DIR, shuffle=False)

    transforms_equalize = F.ComposeOp([F.Decode(),
                                       F.Resize((224, 224)),
                                       F.Equalize(),
                                       F.ToTensor()])

    ds_equalize = ds.map(input_columns="image",
                         operations=transforms_equalize())

    ds_equalize = ds_equalize.batch(512)

    for idx, (image, _) in enumerate(ds_equalize):
        if idx == 0:
            images_equalize = np.transpose(image, (0, 2, 3, 1))
        else:
            images_equalize = np.append(images_equalize,
                                        np.transpose(image, (0, 2, 3, 1)),
                                        axis=0)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = np.mean((images_equalize[i] - images_original[i]) ** 2)
    logger.info("MSE= {}".format(str(np.mean(mse))))

    if plot:
        visualize(images_original, images_equalize)


if __name__ == "__main__":
    test_equalize(plot=True)
