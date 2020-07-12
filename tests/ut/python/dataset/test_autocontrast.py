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
Testing AutoContrast op in DE
"""
import numpy as np
import mindspore.dataset.engine as de
import mindspore.dataset.transforms.vision.py_transforms as F
import mindspore.dataset.transforms.vision.c_transforms as C
from mindspore import log as logger
from util import visualize_list, diff_mse, save_and_check_md5

DATA_DIR = "../data/dataset/testImageNetData/train/"

GENERATE_GOLDEN = False


def test_auto_contrast_py(plot=False):
    """
    Test AutoContrast
    """
    logger.info("Test AutoContrast Python Op")

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

            # AutoContrast Images
    ds = de.ImageFolderDatasetV2(dataset_dir=DATA_DIR, shuffle=False)

    transforms_auto_contrast = F.ComposeOp([F.Decode(),
                                            F.Resize((224, 224)),
                                            F.AutoContrast(),
                                            F.ToTensor()])

    ds_auto_contrast = ds.map(input_columns="image",
                              operations=transforms_auto_contrast())

    ds_auto_contrast = ds_auto_contrast.batch(512)

    for idx, (image, _) in enumerate(ds_auto_contrast):
        if idx == 0:
            images_auto_contrast = np.transpose(image, (0, 2, 3, 1))
        else:
            images_auto_contrast = np.append(images_auto_contrast,
                                             np.transpose(image, (0, 2, 3, 1)),
                                             axis=0)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_auto_contrast[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))

    # Compare with expected md5 from images
    filename = "autcontrast_01_result_py.npz"
    save_and_check_md5(ds_auto_contrast, filename, generate_golden=GENERATE_GOLDEN)

    if plot:
        visualize_list(images_original, images_auto_contrast)


def test_auto_contrast_c(plot=False):
    """
    Test AutoContrast C Op
    """
    logger.info("Test AutoContrast C Op")

    # AutoContrast Images
    ds = de.ImageFolderDatasetV2(dataset_dir=DATA_DIR, shuffle=False)
    ds = ds.map(input_columns=["image"],
                operations=[C.Decode(),
                            C.Resize((224, 224))])
    python_op = F.AutoContrast()
    c_op = C.AutoContrast()
    transforms_op = F.ComposeOp([lambda img: F.ToPIL()(img.astype(np.uint8)),
                                 python_op,
                                 np.array])()

    ds_auto_contrast_py = ds.map(input_columns="image",
                                 operations=transforms_op)

    ds_auto_contrast_py = ds_auto_contrast_py.batch(512)

    for idx, (image, _) in enumerate(ds_auto_contrast_py):
        if idx == 0:
            images_auto_contrast_py = image
        else:
            images_auto_contrast_py = np.append(images_auto_contrast_py,
                                                image,
                                                axis=0)

    ds = de.ImageFolderDatasetV2(dataset_dir=DATA_DIR, shuffle=False)
    ds = ds.map(input_columns=["image"],
                operations=[C.Decode(),
                            C.Resize((224, 224))])

    ds_auto_contrast_c = ds.map(input_columns="image",
                                operations=c_op)

    ds_auto_contrast_c = ds_auto_contrast_c.batch(512)

    for idx, (image, _) in enumerate(ds_auto_contrast_c):
        if idx == 0:
            images_auto_contrast_c = image
        else:
            images_auto_contrast_c = np.append(images_auto_contrast_c,
                                               image,
                                               axis=0)

    num_samples = images_auto_contrast_c.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_auto_contrast_c[i], images_auto_contrast_py[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))
    np.testing.assert_equal(np.mean(mse), 0.0)

    if plot:
        visualize_list(images_auto_contrast_c, images_auto_contrast_py, visualize_mode=2)


def test_auto_contrast_one_channel_c(plot=False):
    """
    Test AutoContrast C op with one channel
    """
    logger.info("Test AutoContrast C Op With One Channel Images")

    # AutoContrast Images
    ds = de.ImageFolderDatasetV2(dataset_dir=DATA_DIR, shuffle=False)
    ds = ds.map(input_columns=["image"],
                operations=[C.Decode(),
                            C.Resize((224, 224))])
    python_op = F.AutoContrast()
    c_op = C.AutoContrast()
    # not using F.ToTensor() since it converts to floats
    transforms_op = F.ComposeOp([lambda img: (np.array(img)[:, :, 0]).astype(np.uint8),
                                 F.ToPIL(),
                                 python_op,
                                 np.array])()

    ds_auto_contrast_py = ds.map(input_columns="image",
                                 operations=transforms_op)

    ds_auto_contrast_py = ds_auto_contrast_py.batch(512)

    for idx, (image, _) in enumerate(ds_auto_contrast_py):
        if idx == 0:
            images_auto_contrast_py = image
        else:
            images_auto_contrast_py = np.append(images_auto_contrast_py,
                                                image,
                                                axis=0)

    ds = de.ImageFolderDatasetV2(dataset_dir=DATA_DIR, shuffle=False)
    ds = ds.map(input_columns=["image"],
                operations=[C.Decode(),
                            C.Resize((224, 224)),
                            lambda img: np.array(img[:, :, 0])])

    ds_auto_contrast_c = ds.map(input_columns="image",
                                operations=c_op)

    ds_auto_contrast_c = ds_auto_contrast_c.batch(512)

    for idx, (image, _) in enumerate(ds_auto_contrast_c):
        if idx == 0:
            images_auto_contrast_c = image
        else:
            images_auto_contrast_c = np.append(images_auto_contrast_c,
                                               image,
                                               axis=0)

    num_samples = images_auto_contrast_c.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_auto_contrast_c[i], images_auto_contrast_py[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))
    np.testing.assert_equal(np.mean(mse), 0.0)

    if plot:
        visualize_list(images_auto_contrast_c, images_auto_contrast_py, visualize_mode=2)


def test_auto_contrast_invalid_input_c():
    """
    Test AutoContrast C Op with invalid params
    """
    logger.info("Test AutoContrast C Op with invalid params")
    try:
        ds = de.ImageFolderDatasetV2(dataset_dir=DATA_DIR, shuffle=False)
        ds = ds.map(input_columns=["image"],
                    operations=[C.Decode(),
                                C.Resize((224, 224)),
                                lambda img: np.array(img[:, :, 0])])
        # invalid ignore
        ds = ds.map(input_columns="image",
                    operations=C.AutoContrast(ignore=255.5))
    except TypeError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "Argument ignore with value 255.5 is not of type" in str(error)


if __name__ == "__main__":
    test_auto_contrast_py(plot=True)
    test_auto_contrast_c(plot=True)
    test_auto_contrast_one_channel_c(plot=True)
    test_auto_contrast_invalid_input_c()
