# Copyright 2019-2025 Huawei Technologies Co., Ltd
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
"""Test ManifestDataset."""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from mindspore import log as logger
import mindspore.dataset as ds
from mindspore.dataset import vision
from mindspore.dataset import transforms

DATA_FILE = "../data/dataset/testManifestData/test.manifest"


def test_manifest_dataset_train():
    """
    Feature: ManifestDataset
    Description: Test ManifestDataset for train
    Expectation: The dataset is processed as expected
    """
    data = ds.ManifestDataset(DATA_FILE, decode=True)
    count = 0
    cat_count = 0
    dog_count = 0
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("item[image] is {}".format(item["image"]))
        count = count + 1
        if item["label"].size == 1 and item["label"] == 0:
            cat_count = cat_count + 1
        elif item["label"].size == 1 and item["label"] == 1:
            dog_count = dog_count + 1
    assert cat_count == 2
    assert dog_count == 1
    assert count == 4


def test_manifest_dataset_eval():
    """
    Feature: ManifestDataset
    Description: Test ManifestDataset for evaluation
    Expectation: The dataset is processed as expected
    """
    data = ds.ManifestDataset(DATA_FILE, "eval", decode=True)
    count = 0
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("item[image] is {}".format(item["image"]))
        count = count + 1
        if item["label"] != 0 and item["label"] != 1:
            assert 0
    assert count == 2


def test_manifest_dataset_class_index():
    """
    Feature: ManifestDataset
    Description: Test ManifestDataset with class_indexing parameter
    Expectation: The dataset is processed as expected
    """
    class_indexing = {"dog": 11}
    data = ds.ManifestDataset(DATA_FILE, decode=True, class_indexing=class_indexing)
    out_class_indexing = data.get_class_indexing()
    assert out_class_indexing == {"dog": 11}
    count = 0
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("item[image] is {}".format(item["image"]))
        count = count + 1
        if item["label"] != 11:
            assert 0
    assert count == 1


def test_manifest_dataset_get_class_index():
    """
    Feature: ManifestDataset
    Description: Test ManifestDataset get_class_indexing
    Expectation: The dataset is processed as expected
    """
    data = ds.ManifestDataset(DATA_FILE, decode=True)
    class_indexing = data.get_class_indexing()
    assert class_indexing == {"cat": 0, "dog": 1, "flower": 2}
    data = data.shuffle(4)
    class_indexing = data.get_class_indexing()
    assert class_indexing == {"cat": 0, "dog": 1, "flower": 2}
    count = 0
    for item in data.create_dict_iterator(num_epochs=1):
        logger.info("item[image] is {}".format(item["image"]))
        count = count + 1
    assert count == 4


def test_manifest_dataset_multi_label():
    """
    Feature: ManifestDataset
    Description: Test ManifestDataset with multi labels
    Expectation: The dataset is processed as expected
    """
    data = ds.ManifestDataset(DATA_FILE, decode=True, shuffle=False)
    count = 0
    expect_label = [1, 0, 0, [0, 2]]
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert item["label"].tolist() == expect_label[count]
        logger.info("item[image] is {}".format(item["image"]))
        count = count + 1
    assert count == 4


def multi_label_hot(x):
    result = np.zeros(x.size // x.ndim, dtype=int)
    if x.ndim > 1:
        for i in range(x.ndim):
            result = np.add(result, x[i])
    else:
        result = np.add(result, x)

    return result


def test_manifest_dataset_multi_label_onehot():
    """
    Feature: ManifestDataset
    Description: Test ManifestDataset with multi labels in OneHot
    Expectation: The dataset is processed as expected
    """
    data = ds.ManifestDataset(DATA_FILE, decode=True, shuffle=False)
    expect_label = [[[0, 1, 0], [1, 0, 0]], [[1, 0, 0], [1, 0, 1]]]
    one_hot_encode = transforms.OneHot(3)
    data = data.map(operations=one_hot_encode, input_columns=["label"])
    data = data.map(operations=multi_label_hot, input_columns=["label"])
    data = data.batch(2)
    count = 0
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert item["label"].tolist() == expect_label[count]
        logger.info("item[image] is {}".format(item["image"]))
        count = count + 1


def test_manifest_dataset_get_num_class():
    """
    Feature: ManifestDataset
    Description: Test ManifestDataset num_classes
    Expectation: The dataset is processed as expected
    """
    data = ds.ManifestDataset(DATA_FILE, decode=True, shuffle=False)
    assert data.num_classes() == 3

    padded_samples = [{"image": np.zeros(1, np.uint8), "label": np.array(1, np.int32)}]
    padded_ds = ds.PaddedDataset(padded_samples)

    data = data.repeat(2)
    padded_ds = padded_ds.repeat(2)

    data1 = data + padded_ds
    assert data1.num_classes() == 3


def test_manifest_dataset_exception():
    """
    Feature: ManifestDataset
    Description: Test error cases on ManifestDataset
    Expectation: Correct error is thrown as expected
    """

    def exception_func(item):
        raise Exception("Error occur!")

    try:
        data = ds.ManifestDataset(DATA_FILE)
        data = data.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    try:
        data = ds.ManifestDataset(DATA_FILE)
        data = data.map(operations=vision.Decode(), input_columns=["image"], num_parallel_workers=1)
        data = data.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    try:
        data = ds.ManifestDataset(DATA_FILE)
        data = data.map(operations=exception_func, input_columns=["label"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    no_source_data_file = "../data/dataset/testManifestData/invalidNoSource.manifest"
    try:
        data = ds.ManifestDataset(no_source_data_file)
        for _ in data.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "Invalid manifest file, 'source' is missing in" in str(e)

    no_usage_data_file = "../data/dataset/testManifestData/invalidNoUsage.manifest"
    try:
        data = ds.ManifestDataset(no_usage_data_file)
        for _ in data.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "Invalid manifest file, 'usage' is missing in" in str(e)


def test_manifest_dataset_error_sample_empty():
    """
    Feature: ManifestDataset
    Description: Test ManifestDataset with sample error of empty sample
    Expectation: Error is raised as expected
    """
    error_sample_data_file = "../data/dataset/testManifestData/errorSample1_empty.manifest"
    data = ds.ManifestDataset(error_sample_data_file, "eval", num_samples=3, decode=True)
    with pytest.raises(RuntimeError) as error_info:
        for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass
    assert "Invalid manifest file" in str(error_info.value)


def test_manifest_dataset_error_sample_sourceop():
    """
    Feature: ManifestDataset
    Description: Test ManifestDataset with sample errors. Include just source op in pipeline.
    Expectation: The dataset is processed as expected.
        Note: No error is raised from the invalid JPEG file.
    """

    def test_config(my_error_sample_data_file, my_num_rows):
        # For ManifestDataset, use decode=False
        data1 = ds.ManifestDataset(
            my_error_sample_data_file, "eval", num_samples=3, num_parallel_workers=1, decode=False
        )
        count = 0
        for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            count += 1
        assert count == my_num_rows

        # For ManifestDataset, use decode=True
        data2 = ds.ManifestDataset(
            my_error_sample_data_file, "eval", num_samples=3, num_parallel_workers=1, decode=True
        )
        count = 0
        for _ in data2.create_dict_iterator(num_epochs=1, output_numpy=True):
            count += 1
        assert count == my_num_rows

    # Test corrupt sample
    test_config("../data/dataset/testManifestData/errorSample2_corrupt1midsamp.manifest", 3)
    test_config("../data/dataset/testManifestData/errorSample4_corrupt1midsamp.manifest", 2)
    # Test text sample, instead of image sample
    test_config("../data/dataset/testManifestData/errorSample3_text.manifest", 2)


def test_manifest_dataset_error_sample_mapop_c():
    """
    Feature: ManifestDataset
    Description: Test ManifestDataset with sample errors. Include map op in pipeline with C++ implemented ops
    Expectation: The dataset is processed as expected.
        Note: No error is raised from the invalid JPEG file.
    """

    def test_config(my_error_sample_data_file, my_num_rows):
        # For ManifestDataset, use decode=True
        data3 = ds.ManifestDataset(
            my_error_sample_data_file, "eval", num_samples=3, num_parallel_workers=1, decode=True
        )
        # Add map op to the pipeline
        data3 = data3.map(operations=[vision.HorizontalFlip()], input_columns=["image"], num_parallel_workers=1)
        count = 0
        for _ in data3.create_dict_iterator(num_epochs=1, output_numpy=True):
            count += 1
        assert count == my_num_rows

        # For ManifestDataset, use decode=False
        data4 = ds.ManifestDataset(
            my_error_sample_data_file, "eval", num_samples=3, num_parallel_workers=1, decode=False
        )
        # Add map op to the pipeline
        data4 = data4.map(
            operations=[vision.Decode(), vision.HorizontalFlip()], input_columns=["image"], num_parallel_workers=1
        )
        count = 0
        for _ in data4.create_dict_iterator(num_epochs=1, output_numpy=True):
            count += 1
        assert count == my_num_rows

    # Test corrupt sample
    test_config("../data/dataset/testManifestData/errorSample2_corrupt1midsamp.manifest", 3)
    test_config("../data/dataset/testManifestData/errorSample4_corrupt1midsamp.manifest", 2)
    # Test text sample, instead of image sample
    test_config("../data/dataset/testManifestData/errorSample3_text.manifest", 2)


def test_manifest_dataset_error_sample_mapop_py():
    """
    Feature: ManifestDataset
    Description: Test ManifestDataset with sample errors. Include map op in pipeline with Python implemented ops
    Expectation: The dataset is processed as expected.
        Note: No error is raised from the invalid JPEG file.
    """

    def test_config(my_error_sample_data_file, my_num_rows):
        # For ManifestDataset, use decode=True
        data3 = ds.ManifestDataset(
            my_error_sample_data_file, "eval", num_samples=3, num_parallel_workers=1, decode=True
        )
        # Add map op to the pipeline
        data3 = data3.map(operations=[vision.RandomVerticalFlip(0.8)], input_columns=["image"], num_parallel_workers=1)
        count = 0
        for _ in data3.create_dict_iterator(num_epochs=1, output_numpy=True):
            count += 1
        assert count == my_num_rows

        # For ManifestDataset, use decode=False
        data4 = ds.ManifestDataset(
            my_error_sample_data_file, "eval", num_samples=3, num_parallel_workers=1, decode=False
        )
        # Add map op to the pipeline
        data4 = data4.map(
            operations=[vision.Decode(to_pil=True), vision.RandomVerticalFlip(0.2)],
            input_columns=["image"],
            num_parallel_workers=1,
        )
        count = 0
        for _ in data4.create_dict_iterator(num_epochs=1, output_numpy=True):
            count += 1
        assert count == my_num_rows

    # Test corrupt sample
    test_config("../data/dataset/testManifestData/errorSample2_corrupt1midsamp.manifest", 3)
    test_config("../data/dataset/testManifestData/errorSample4_corrupt1midsamp.manifest", 2)
    # Test text sample, instead of image sample
    test_config("../data/dataset/testManifestData/errorSample3_text.manifest", 2)


def test_manifest_dataset_raise_error_with_invalid_utf8():
    """
    Feature: ManifestDataset.
    Description: Test ManifestDataset with wrong dataset file causing the error containing invalid UTF-8.
    Expectation: No UnicodeDecodeError is raised.
    """
    # Run the test logic in a separate Python subprocess and decode the non-UTF8 characters
    # in the C++ error message before returning to the current pytest process.
    code = textwrap.dedent(
        """
        import numpy as np
        import pytest
        import mindspore.dataset as ds

        def run():
            dataset = ds.ManifestDataset("../data/dataset/testManifestData/train/1.JPEG")
            iterator = dataset.create_dict_iterator(num_epochs=1, output_numpy=True)
            with pytest.raises(RuntimeError, match="Invalid manifest file"):
                next(iterator)

        if __name__ == "__main__":
            run()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="backslashreplace",
        check=False,
    )
    if result.returncode != 0:
        if result.stderr:
            raise RuntimeError(result.stderr)
    assert result.returncode == 0


if __name__ == "__main__":
    test_manifest_dataset_train()
    test_manifest_dataset_eval()
    test_manifest_dataset_class_index()
    test_manifest_dataset_get_class_index()
    test_manifest_dataset_multi_label()
    test_manifest_dataset_multi_label_onehot()
    test_manifest_dataset_get_num_class()
    test_manifest_dataset_exception()
    test_manifest_dataset_error_sample_empty()
    test_manifest_dataset_error_sample_sourceop()
    test_manifest_dataset_error_sample_mapop_c()
    test_manifest_dataset_error_sample_mapop_py()
    test_manifest_dataset_raise_error_with_invalid_utf8()
