# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
Test Caltech101 dataset operations
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image
from scipy.io import loadmat

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import log as logger

DATASET_DIR = "../data/dataset/testCaltech101Data"
WRONG_DIR = "../data/dataset/notExist"


def get_index_info():
    dataset_dir = os.path.realpath(DATASET_DIR)
    image_dir = os.path.join(dataset_dir, "101_ObjectCategories")
    classes = sorted(os.listdir(image_dir))
    if "BACKGROUND_Google" in classes:
        classes.remove("BACKGROUND_Google")
    name_map = {"Faces": "Faces_2",
                "Faces_easy": "Faces_3",
                "Motorbikes": "Motorbikes_16",
                "airplanes": "Airplanes_Side_2"}
    annotation_classes = [name_map[class_name] if class_name in name_map else class_name for class_name in classes]
    image_index = []
    image_label = []
    for i, c in enumerate(classes):
        sub_dir = os.path.join(image_dir, c)
        if not os.path.isdir(sub_dir) or not os.access(sub_dir, os.R_OK):
            continue
        num_images = len(os.listdir(sub_dir))
        image_index.extend(range(1, num_images + 1))
        image_label.extend(num_images * [i])
    return image_index, image_label, classes, annotation_classes


def load_caltech101(target_type="category", decode=False):
    """
    load Caltech101 data
    """
    dataset_dir = os.path.realpath(DATASET_DIR)
    image_dir = os.path.join(dataset_dir, "101_ObjectCategories")
    annotation_dir = os.path.join(dataset_dir, "Annotations")
    image_index, image_label, classes, annotation_classes = get_index_info()
    images, categories, annotations = [], [], []
    num_images = len(image_index)
    for i in range(num_images):
        image_file = os.path.join(image_dir, classes[image_label[i]], "image_{:04d}.jpg".format(image_index[i]))
        if not os.path.exists(image_file):
            raise ValueError("The image file {} does not exist or permission denied!".format(image_file))
        if decode:
            image = np.asarray(Image.open(image_file).convert("RGB"))
        else:
            image = np.fromfile(image_file, dtype=np.uint8)
        images.append(image)
    if target_type == "category":
        for i in range(num_images):
            categories.append(image_label[i])
        return images, categories
    for i in range(num_images):
        annotation_file = os.path.join(annotation_dir, annotation_classes[image_label[i]],
                                       "annotation_{:04d}.mat".format(image_index[i]))
        if not os.path.exists(annotation_file):
            raise ValueError("The annotation file {} does not exist or permission denied!".format(annotation_file))
        annotation = loadmat(annotation_file)["obj_contour"]
        annotations.append(annotation)
    if target_type == "annotation":
        return images, annotations
    for i in range(num_images):
        categories.append(image_label[i])
    return images, categories, annotations


def visualize_dataset(images, labels):
    """
    Helper function to visualize the dataset samples
    """
    num_samples = len(images)
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i].squeeze())
        plt.title(labels[i])
    plt.show()


def test_caltech101_content_check():
    """
    Feature: Caltech101Dataset
    Description: Check if the image data of caltech101 dataset is read correctly
    Expectation: The data is processed successfully
    """
    logger.info("Test Caltech101Dataset Op with content check")
    all_data = ds.Caltech101Dataset(DATASET_DIR, target_type="annotation", num_samples=4, shuffle=False, decode=True)
    images, annotations = load_caltech101(target_type="annotation", decode=True)
    num_iter = 0
    for i, data in enumerate(all_data.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(data["image"], images[i])
        np.testing.assert_array_equal(data["annotation"], annotations[i])
        num_iter += 1
    assert num_iter == 4

    all_data = ds.Caltech101Dataset(DATASET_DIR, target_type="all", num_samples=4, shuffle=False, decode=True)
    images, categories, annotations = load_caltech101(target_type="all", decode=True)
    num_iter = 0
    for i, data in enumerate(all_data.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(data["image"], images[i])
        np.testing.assert_array_equal(data["category"], categories[i])
        np.testing.assert_array_equal(data["annotation"], annotations[i])
        num_iter += 1
    assert num_iter == 4


def test_caltech101_basic():
    """
    Feature: Caltech101Dataset
    Description: Basic test of Caltech101Dataset
    Expectation: The data is processed successfully
    """
    logger.info("Test Caltech101Dataset Op")

    # case 1: test target_type
    all_data_1 = ds.Caltech101Dataset(DATASET_DIR, shuffle=False)
    all_data_2 = ds.Caltech101Dataset(DATASET_DIR, shuffle=False)

    num_iter = 0
    for item1, item2 in zip(all_data_1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            all_data_2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(item1["category"], item2["category"])
        num_iter += 1
    assert num_iter == 4

    # case 2: test decode
    all_data_1 = ds.Caltech101Dataset(DATASET_DIR, decode=True, shuffle=False)
    all_data_2 = ds.Caltech101Dataset(DATASET_DIR, decode=True, shuffle=False)

    num_iter = 0
    for item1, item2 in zip(all_data_1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            all_data_2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(item1["image"], item2["image"])
        num_iter += 1
    assert num_iter == 4

    # case 3: test num_samples
    all_data = ds.Caltech101Dataset(DATASET_DIR, num_samples=4)
    num_iter = 0
    for _ in all_data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 4

    # case 4: test repeat
    all_data = ds.Caltech101Dataset(DATASET_DIR, num_samples=4)
    all_data = all_data.repeat(2)
    num_iter = 0
    for _ in all_data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 8

    # case 5: test get_dataset_size, resize and batch
    all_data = ds.Caltech101Dataset(DATASET_DIR, num_samples=12)
    all_data = all_data.map(operations=[vision.Decode(), vision.Resize((120, 120))], input_columns=["image"],
                            num_parallel_workers=1)

    assert all_data.get_dataset_size() == 4
    assert all_data.get_batch_size() == 1
    # drop_remainder is default to be False
    all_data = all_data.batch(batch_size=4)
    assert all_data.get_batch_size() == 4
    assert all_data.get_dataset_size() == 1

    num_iter = 0
    for _ in all_data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 1

    # case 6: test get_class_indexing
    all_data = ds.Caltech101Dataset(DATASET_DIR, num_samples=4)
    class_indexing = all_data.get_class_indexing()
    assert class_indexing["Faces"] == 0
    assert class_indexing["yin_yang"] == 100


def test_caltech101_target_type():
    """
    Feature: Caltech101Dataset
    Description: Test Caltech101Dataset with target_type
    Expectation: The data is processed successfully
    """
    logger.info("Test Caltech101Dataset Op with target_type")
    all_data_1 = ds.Caltech101Dataset(DATASET_DIR, target_type="annotation", shuffle=False)
    all_data_2 = ds.Caltech101Dataset(DATASET_DIR, target_type="annotation", shuffle=False)
    num_iter = 0
    for item1, item2 in zip(all_data_1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            all_data_2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(item1["annotation"], item2["annotation"])
        num_iter += 1
    assert num_iter == 4
    all_data_1 = ds.Caltech101Dataset(DATASET_DIR, target_type="all", shuffle=False)
    all_data_2 = ds.Caltech101Dataset(DATASET_DIR, target_type="all", shuffle=False)
    num_iter = 0
    for item1, item2 in zip(all_data_1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            all_data_2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(item1["category"], item2["category"])
        np.testing.assert_array_equal(item1["annotation"], item2["annotation"])
        num_iter += 1
    assert num_iter == 4
    all_data_1 = ds.Caltech101Dataset(DATASET_DIR, target_type="category", shuffle=False)
    all_data_2 = ds.Caltech101Dataset(DATASET_DIR, target_type="category", shuffle=False)
    num_iter = 0
    for item1, item2 in zip(all_data_1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            all_data_2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(item1["category"], item2["category"])
        num_iter += 1
    assert num_iter == 4


def test_caltech101_sequential_sampler():
    """
    Feature: Caltech101Dataset
    Description: Test Caltech101Dataset with SequentialSampler
    Expectation: The data is processed successfully
    """
    logger.info("Test Caltech101Dataset Op with SequentialSampler")
    num_samples = 4
    sampler = ds.SequentialSampler(num_samples=num_samples)
    all_data_1 = ds.Caltech101Dataset(DATASET_DIR, sampler=sampler)
    all_data_2 = ds.Caltech101Dataset(DATASET_DIR, shuffle=False, num_samples=num_samples)
    label_list_1, label_list_2 = [], []
    num_iter = 0
    for item1, item2 in zip(all_data_1.create_dict_iterator(num_epochs=1),
                            all_data_2.create_dict_iterator(num_epochs=1)):
        label_list_1.append(item1["category"].asnumpy())
        label_list_2.append(item2["category"].asnumpy())
        num_iter += 1
    np.testing.assert_array_equal(label_list_1, label_list_2)
    assert num_iter == num_samples


def test_caltech101_exception():
    """
    Feature: Caltech101Dataset
    Description: Test error cases for Caltech101Dataset
    Expectation: Throw correct error and message
    """
    logger.info("Test error cases for Caltech101Dataset")
    error_msg_1 = "sampler and shuffle cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_1):
        ds.Caltech101Dataset(DATASET_DIR, shuffle=False, sampler=ds.SequentialSampler(1))

    error_msg_2 = "sampler and sharding cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_2):
        ds.Caltech101Dataset(DATASET_DIR, sampler=ds.SequentialSampler(1), num_shards=2, shard_id=0)

    error_msg_3 = "num_shards is specified and currently requires shard_id as well"
    with pytest.raises(RuntimeError, match=error_msg_3):
        ds.Caltech101Dataset(DATASET_DIR, num_shards=10)

    error_msg_4 = "shard_id is specified but num_shards is not"
    with pytest.raises(RuntimeError, match=error_msg_4):
        ds.Caltech101Dataset(DATASET_DIR, shard_id=0)

    error_msg_5 = "Input shard_id is not within the required interval"
    with pytest.raises(ValueError, match=error_msg_5):
        ds.Caltech101Dataset(DATASET_DIR, num_shards=5, shard_id=-1)

    with pytest.raises(ValueError, match=error_msg_5):
        ds.Caltech101Dataset(DATASET_DIR, num_shards=5, shard_id=5)

    with pytest.raises(ValueError, match=error_msg_5):
        ds.Caltech101Dataset(DATASET_DIR, num_shards=2, shard_id=5)

    error_msg_6 = "num_parallel_workers exceeds"
    with pytest.raises(ValueError, match=error_msg_6):
        ds.Caltech101Dataset(DATASET_DIR, shuffle=False, num_parallel_workers=0)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.Caltech101Dataset(DATASET_DIR, shuffle=False, num_parallel_workers=256)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.Caltech101Dataset(DATASET_DIR, shuffle=False, num_parallel_workers=-2)

    error_msg_7 = "Argument shard_id"
    with pytest.raises(TypeError, match=error_msg_7):
        ds.Caltech101Dataset(DATASET_DIR, num_shards=2, shard_id="0")

    error_msg_8 = "does not exist or is not a directory or permission denied!"
    with pytest.raises(ValueError, match=error_msg_8):
        all_data = ds.Caltech101Dataset(WRONG_DIR, WRONG_DIR)
        for _ in all_data.create_dict_iterator(num_epochs=1):
            pass

    error_msg_9 = "Input target_type is not within the valid set of \\['category', 'annotation', 'all'\\]."
    with pytest.raises(ValueError, match=error_msg_9):
        all_data = ds.Caltech101Dataset(DATASET_DIR, target_type="cate")
        for _ in all_data.create_dict_iterator(num_epochs=1):
            pass


def test_caltech101_visualize(plot=False):
    """
    Feature: Caltech101Dataset
    Description: Visualize Caltech101Dataset results
    Expectation: The data is processed successfully
    """
    logger.info("Test Caltech101Dataset visualization")

    all_data = ds.Caltech101Dataset(DATASET_DIR, num_samples=4, decode=True, shuffle=False)
    num_iter = 0
    image_list, category_list = [], []
    for item in all_data.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = item["image"]
        category = item["category"]
        image_list.append(image)
        category_list.append("label {}".format(category))
        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 3
        assert image.shape[-1] == 3
        assert image.dtype == np.uint8
        assert category.dtype == np.int64
        num_iter += 1
    assert num_iter == 4
    if plot:
        visualize_dataset(image_list, category_list)


if __name__ == '__main__':
    test_caltech101_content_check()
    test_caltech101_basic()
    test_caltech101_target_type()
    test_caltech101_sequential_sampler()
    test_caltech101_exception()
    test_caltech101_visualize(plot=True)
