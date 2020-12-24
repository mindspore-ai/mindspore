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
import mindspore.dataset.transforms.py_transforms
import mindspore.dataset.vision.py_transforms as F
import mindspore.dataset.vision.c_transforms as C
from mindspore import log as logger
from util import visualize_list, visualize_one_channel_dataset, diff_mse, save_and_check_md5, \
    config_get_set_seed, config_get_set_num_parallel_workers

DATA_DIR = "../data/dataset/testImageNetData/train/"
MNIST_DATA_DIR = "../data/dataset/testMnistData"

GENERATE_GOLDEN = False


def test_random_sharpness_py(degrees=(0.7, 0.7), plot=False):
    """
    Test RandomSharpness python op
    """
    logger.info("Test RandomSharpness python op")

    # Original Images
    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transforms_original = mindspore.dataset.transforms.py_transforms.Compose([F.Decode(),
                                                                              F.Resize((224, 224)),
                                                                              F.ToTensor()])

    ds_original = data.map(operations=transforms_original, input_columns="image")

    ds_original = ds_original.batch(512)

    for idx, (image, _) in enumerate(ds_original.create_tuple_iterator(output_numpy=True)):
        if idx == 0:
            images_original = np.transpose(image, (0, 2, 3, 1))
        else:
            images_original = np.append(images_original,
                                        np.transpose(image, (0, 2, 3, 1)),
                                        axis=0)

    # Random Sharpness Adjusted Images
    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    py_op = F.RandomSharpness()
    if degrees is not None:
        py_op = F.RandomSharpness(degrees)

    transforms_random_sharpness = mindspore.dataset.transforms.py_transforms.Compose([F.Decode(),
                                                                                      F.Resize((224, 224)),
                                                                                      py_op,
                                                                                      F.ToTensor()])

    ds_random_sharpness = data.map(operations=transforms_random_sharpness, input_columns="image")

    ds_random_sharpness = ds_random_sharpness.batch(512)

    for idx, (image, _) in enumerate(ds_random_sharpness.create_tuple_iterator(output_numpy=True)):
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


def test_random_sharpness_py_md5():
    """
    Test RandomSharpness python op with md5 comparison
    """
    logger.info("Test RandomSharpness python op with md5 comparison")
    original_seed = config_get_set_seed(5)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # define map operations
    transforms = [
        F.Decode(),
        F.RandomSharpness((20.0, 25.0)),
        F.ToTensor()
    ]
    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)

    #  Generate dataset
    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
    data = data.map(operations=transform, input_columns=["image"])

    # check results with md5 comparison
    filename = "random_sharpness_py_01_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_sharpness_c(degrees=(1.6, 1.6), plot=False):
    """
    Test RandomSharpness cpp op
    """
    print(degrees)
    logger.info("Test RandomSharpness cpp op")

    # Original Images
    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transforms_original = [C.Decode(),
                           C.Resize((224, 224))]

    ds_original = data.map(operations=transforms_original, input_columns="image")

    ds_original = ds_original.batch(512)

    for idx, (image, _) in enumerate(ds_original.create_tuple_iterator(output_numpy=True)):
        if idx == 0:
            images_original = image
        else:
            images_original = np.append(images_original,
                                        image,
                                        axis=0)

            # Random Sharpness Adjusted Images
    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    c_op = C.RandomSharpness()
    if degrees is not None:
        c_op = C.RandomSharpness(degrees)

    transforms_random_sharpness = [C.Decode(),
                                   C.Resize((224, 224)),
                                   c_op]

    ds_random_sharpness = data.map(operations=transforms_random_sharpness, input_columns="image")

    ds_random_sharpness = ds_random_sharpness.batch(512)

    for idx, (image, _) in enumerate(ds_random_sharpness.create_tuple_iterator(output_numpy=True)):
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
    Test RandomSharpness cpp op with md5 comparison
    """
    logger.info("Test RandomSharpness cpp op with md5 comparison")
    original_seed = config_get_set_seed(200)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # define map operations
    transforms = [
        C.Decode(),
        C.RandomSharpness((10.0, 15.0))
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


def test_random_sharpness_c_py(degrees=(1.0, 1.0), plot=False):
    """
    Test Random Sharpness C and python Op
    """
    logger.info("Test RandomSharpness C and python Op")

    # RandomSharpness Images
    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
    data = data.map(operations=[C.Decode(), C.Resize((200, 300))], input_columns=["image"])

    python_op = F.RandomSharpness(degrees)
    c_op = C.RandomSharpness(degrees)

    transforms_op = mindspore.dataset.transforms.py_transforms.Compose([lambda img: F.ToPIL()(img.astype(np.uint8)),
                                                                        python_op,
                                                                        np.array])

    ds_random_sharpness_py = data.map(operations=transforms_op, input_columns="image")

    ds_random_sharpness_py = ds_random_sharpness_py.batch(512)

    for idx, (image, _) in enumerate(ds_random_sharpness_py.create_tuple_iterator(output_numpy=True)):
        if idx == 0:
            images_random_sharpness_py = image

        else:
            images_random_sharpness_py = np.append(images_random_sharpness_py,
                                                   image,
                                                   axis=0)

    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
    data = data.map(operations=[C.Decode(), C.Resize((200, 300))], input_columns=["image"])

    ds_images_random_sharpness_c = data.map(operations=c_op, input_columns="image")

    ds_images_random_sharpness_c = ds_images_random_sharpness_c.batch(512)

    for idx, (image, _) in enumerate(ds_images_random_sharpness_c.create_tuple_iterator(output_numpy=True)):
        if idx == 0:
            images_random_sharpness_c = image

        else:
            images_random_sharpness_c = np.append(images_random_sharpness_c,
                                                  image,
                                                  axis=0)

    num_samples = images_random_sharpness_c.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_random_sharpness_c[i], images_random_sharpness_py[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))
    if plot:
        visualize_list(images_random_sharpness_c, images_random_sharpness_py, visualize_mode=2)


def test_random_sharpness_one_channel_c(degrees=(1.4, 1.4), plot=False):
    """
    Test Random Sharpness cpp op with one channel
    """
    logger.info("Test RandomSharpness C Op With MNIST Dataset (Grayscale images)")

    c_op = C.RandomSharpness()
    if degrees is not None:
        c_op = C.RandomSharpness(degrees)
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
    Test RandomSharpness with invalid input parameters.
    """
    logger.info("Test RandomSharpness with invalid input parameters.")
    try:
        data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data = data.map(operations=[C.Decode(), C.Resize((224, 224)),
                                    C.RandomSharpness(10)], input_columns=["image"])
    except TypeError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "tuple" in str(error)

    try:
        data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data = data.map(operations=[C.Decode(), C.Resize((224, 224)),
                                    C.RandomSharpness((-10, 10))], input_columns=["image"])
    except ValueError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "interval" in str(error)

    try:
        data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data = data.map(operations=[C.Decode(), C.Resize((224, 224)),
                                    C.RandomSharpness((10, 5))], input_columns=["image"])
    except ValueError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "(min,max)" in str(error)


if __name__ == "__main__":
    test_random_sharpness_py(plot=True)
    test_random_sharpness_py(None, plot=True)  # Test with default values
    test_random_sharpness_py(degrees=(20.0, 25.0),
                             plot=True)  # Test with degree values that show more obvious transformation
    test_random_sharpness_py_md5()
    test_random_sharpness_c(plot=True)
    test_random_sharpness_c(None, plot=True)  # test with default values
    test_random_sharpness_c(degrees=[10, 15],
                            plot=True)  # Test with degrees values that show more obvious transformation
    test_random_sharpness_c_md5()
    test_random_sharpness_c_py(degrees=[1.5, 1.5], plot=True)
    test_random_sharpness_c_py(degrees=[1, 1], plot=True)
    test_random_sharpness_c_py(degrees=[10, 10], plot=True)
    test_random_sharpness_one_channel_c(degrees=[1.7, 1.7], plot=True)
    test_random_sharpness_one_channel_c(degrees=None, plot=True)  # Test with default values
    test_random_sharpness_invalid_params()
