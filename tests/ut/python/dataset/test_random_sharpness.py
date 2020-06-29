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
Testing RandomSharpness op in DE
"""
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.engine as de
import mindspore.dataset.transforms.vision.py_transforms as F
from mindspore import log as logger
from util import visualize_list, diff_mse, save_and_check_md5, \
    config_get_set_seed, config_get_set_num_parallel_workers

DATA_DIR = "../data/dataset/testImageNetData/train/"

GENERATE_GOLDEN = False


def test_random_sharpness(degrees=(0.1, 1.9), plot=False):
    """
    Test RandomSharpness
    """
    logger.info("Test RandomSharpness")

    # Original Images
    data = de.ImageFolderDatasetV2(dataset_dir=DATA_DIR, shuffle=False)

    transforms_original = F.ComposeOp([F.Decode(),
                                       F.Resize((224, 224)),
                                       F.ToTensor()])

    ds_original = data.map(input_columns="image",
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
    data = de.ImageFolderDatasetV2(dataset_dir=DATA_DIR, shuffle=False)

    transforms_random_sharpness = F.ComposeOp([F.Decode(),
                                               F.Resize((224, 224)),
                                               F.RandomSharpness(degrees=degrees),
                                               F.ToTensor()])

    ds_random_sharpness = data.map(input_columns="image",
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
        mse[i] = diff_mse(images_random_sharpness[i], images_original[i])

    logger.info("MSE= {}".format(str(np.mean(mse))))

    if plot:
        visualize_list(images_original, images_random_sharpness)


def test_random_sharpness_md5():
    """
    Test RandomSharpness with md5 comparison
    """
    logger.info("Test RandomSharpness with md5 comparison")
    original_seed = config_get_set_seed(5)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # define map operations
    transforms = [
        F.Decode(),
        F.RandomSharpness((0.5, 1.5)),
        F.ToTensor()
    ]
    transform = F.ComposeOp(transforms)

    #  Generate dataset
    data = de.ImageFolderDatasetV2(dataset_dir=DATA_DIR, shuffle=False)
    data = data.map(input_columns=["image"], operations=transform())

    # check results with md5 comparison
    filename = "random_sharpness_01_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


if __name__ == "__main__":
    test_random_sharpness()
    test_random_sharpness(plot=True)
    test_random_sharpness(degrees=(0.5, 1.5), plot=True)
    test_random_sharpness_md5()
