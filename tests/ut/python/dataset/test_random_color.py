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
"""
Testing RandomColor op in DE
"""
import numpy as np

import mindspore.dataset.engine as de
import mindspore.dataset.transforms.vision.py_transforms as F
from mindspore import log as logger
from util import visualize_list

DATA_DIR = "../data/dataset/testImageNetData/train/"


def test_random_color(degrees=(0.1, 1.9), plot=False):
    """
    Test RandomColor
    """
    logger.info("Test RandomColor")

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

            # Random Color Adjusted Images
    ds = de.ImageFolderDatasetV2(dataset_dir=DATA_DIR, shuffle=False)

    transforms_random_color = F.ComposeOp([F.Decode(),
                                           F.Resize((224, 224)),
                                           F.RandomColor(degrees=degrees),
                                           F.ToTensor()])

    ds_random_color = ds.map(input_columns="image",
                             operations=transforms_random_color())

    ds_random_color = ds_random_color.batch(512)

    for idx, (image, _) in enumerate(ds_random_color):
        if idx == 0:
            images_random_color = np.transpose(image, (0, 2, 3, 1))
        else:
            images_random_color = np.append(images_random_color,
                                            np.transpose(image, (0, 2, 3, 1)),
                                            axis=0)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = np.mean((images_random_color[i] - images_original[i]) ** 2)
    logger.info("MSE= {}".format(str(np.mean(mse))))

    if plot:
        visualize_list(images_original, images_random_color)


if __name__ == "__main__":
    test_random_color()
    test_random_color(plot=True)
    test_random_color(degrees=(0.5, 1.5), plot=True)
