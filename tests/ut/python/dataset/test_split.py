# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
""" Test split operation """
import numpy as np
import pytest
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from util import config_get_set_num_parallel_workers, config_get_set_seed

# test5trainimgs.json contains 5 images whose un-decoded shape is [83554, 54214, 65512, 54214, 64631]
# the label of each image is [0,0,0,1,1] each image can be uniquely identified
# via the following lookup table (dict){(83554, 0): 0, (54214, 0): 1, (54214, 1): 2, (65512, 0): 3, (64631, 1): 4}
manifest_file = "../data/dataset/testManifestData/test5trainimgs.json"
manifest_map = {(172876, 0): 0, (54214, 0): 1, (54214, 1): 2, (173673, 0): 3, (64631, 1): 4}

text_file_dataset_path = "../data/dataset/testTextFileDataset/*"
text_file_data = ["This is a text file.", "Another file.", "Be happy every day.",
                  "End of file.", "Good luck to everyone."]


def split_with_invalid_inputs(d):
    with pytest.raises(ValueError) as info:
        _, _ = d.split([])
    assert "sizes cannot be empty" in str(info.value)

    with pytest.raises(ValueError) as info:
        _, _ = d.split([5, 0.6])
    assert "sizes should be list of int or list of float" in str(info.value)

    with pytest.raises(ValueError) as info:
        _, _ = d.split([-1, 6])
    assert "there should be no negative or zero numbers" in str(info.value)

    with pytest.raises(RuntimeError) as info:
        _, _ = d.split([3, 1])
    assert "Sum of split sizes 4 is not equal to dataset size 5" in str(info.value)

    with pytest.raises(RuntimeError) as info:
        _, _ = d.split([5, 1])
    assert "Sum of split sizes 6 is not equal to dataset size 5" in str(info.value)

    with pytest.raises(RuntimeError) as info:
        _, _ = d.split([0.15, 0.15, 0.15, 0.15, 0.15, 0.25])
    assert "Sum of calculated split sizes 6 is not equal to dataset size 5" in str(info.value)

    with pytest.raises(ValueError) as info:
        _, _ = d.split([-0.5, 0.5])
    assert "there should be no numbers outside the range (0, 1]" in str(info.value)

    with pytest.raises(ValueError) as info:
        _, _ = d.split([1.5, 0.5])
    assert "there should be no numbers outside the range (0, 1]" in str(info.value)

    with pytest.raises(ValueError) as info:
        _, _ = d.split([0.5, 0.6])
    assert "percentages do not sum up to 1" in str(info.value)

    with pytest.raises(ValueError) as info:
        _, _ = d.split([0.3, 0.6])
    assert "percentages do not sum up to 1" in str(info.value)

    with pytest.raises(RuntimeError) as info:
        _, _ = d.split([0.05, 0.95])
    assert "percentage 0.05 is too small" in str(info.value)


def test_unmappable_invalid_input():
    """
    Feature: Split op
    Description: Test split op using unmappable dataset (TextFileDataset)
        with various invalid inputs and applying split op on sharded dataset
    Expectation: Correct error is raised as expected
    """
    d = ds.TextFileDataset(text_file_dataset_path)
    split_with_invalid_inputs(d)

    d = ds.TextFileDataset(text_file_dataset_path, num_shards=2, shard_id=0)
    with pytest.raises(RuntimeError) as info:
        _, _ = d.split([4, 1])
    assert "Dataset should not be sharded before split" in str(info.value)


def test_unmappable_split():
    """
    Feature: Split op
    Description: Test split op using unmappable dataset (TextFileDataset)
        with absolute rows, exact percentages, and fuzzy percentages as input
    Expectation: Output is equal to the expected output
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(4)

    d = ds.TextFileDataset(text_file_dataset_path, shuffle=False)
    s1, s2 = d.split([4, 1], randomize=False)

    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(item["text"])

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(item["text"])

    assert s1_output == text_file_data[0:4]
    assert s2_output == text_file_data[4:]

    # exact percentages
    s1, s2 = d.split([0.8, 0.2], randomize=False)

    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(item["text"])

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(item["text"])

    assert s1_output == text_file_data[0:4]
    assert s2_output == text_file_data[4:]

    # fuzzy percentages
    s1, s2 = d.split([0.33, 0.67], randomize=False)

    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(item["text"])

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(item["text"])

    assert s1_output == text_file_data[0:2]
    assert s2_output == text_file_data[2:]

    # Restore configuration num_parallel_workers
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_unmappable_randomize_deterministic():
    """
    Feature: Split op
    Description: Test split op using unmappable dataset (TextFileDataset) with randomization
    Expectation: Output is equal to the expected output
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(4)

    # the labels outputted by ShuffleOp for seed 53 is [0, 2, 1, 4, 3]
    original_seed = config_get_set_seed(53)

    d = ds.TextFileDataset(text_file_dataset_path, shuffle=False)
    s1, s2 = d.split([0.8, 0.2])

    for _ in range(10):
        s1_output = []
        for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
            s1_output.append(item["text"])

        s2_output = []
        for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
            s2_output.append(item["text"])

        # note no overlap
        assert s1_output == [text_file_data[0], text_file_data[2], text_file_data[1], text_file_data[4]]
        assert s2_output == [text_file_data[3]]

    # Restore configuration num_parallel_workers
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_unmappable_randomize_repeatable():
    """
    Feature: Split op
    Description: Test split op using unmappable dataset (TextFileDataset) with randomization followed by repeat op
    Expectation: Output is equal to the expected output
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(4)

    # the labels outputted by ShuffleOp for seed 53 is [0, 2, 1, 4, 3]
    original_seed = config_get_set_seed(53)

    d = ds.TextFileDataset(text_file_dataset_path, shuffle=False)
    s1, s2 = d.split([0.8, 0.2])

    num_epochs = 5
    s1 = s1.repeat(num_epochs)
    s2 = s2.repeat(num_epochs)

    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(item["text"])

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(item["text"])

    # note no overlap
    assert s1_output == [text_file_data[0], text_file_data[2], text_file_data[1], text_file_data[4]] * num_epochs
    assert s2_output == [text_file_data[3]] * num_epochs

    # Restore configuration num_parallel_workers
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_unmappable_get_dataset_size():
    """
    Feature: Split op
    Description: Test split op using unmappable dataset (TextFileDataset) followed by get_dataset_size
    Expectation: Output is equal to the expected output
    """
    d = ds.TextFileDataset(text_file_dataset_path, shuffle=False)
    s1, s2 = d.split([0.8, 0.2])

    assert d.get_dataset_size() == 5
    assert s1.get_dataset_size() == 4
    assert s2.get_dataset_size() == 1


def test_unmappable_multi_split():
    """
    Feature: Split op
    Description: Test split op using unmappable dataset (TextFileDataset)
        with randomization followed by deterministic split or another randomized split
    Expectation: Output is equal to the expected output
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(4)

    # the labels outputted by ShuffleOp for seed 53 is [0, 2, 1, 4, 3]
    original_seed = config_get_set_seed(53)

    d = ds.TextFileDataset(text_file_dataset_path, shuffle=False)
    s1, s2 = d.split([4, 1])

    s1_correct_output = [text_file_data[0], text_file_data[2], text_file_data[1], text_file_data[4]]

    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(item["text"])
    assert s1_output == s1_correct_output

    # no randomize in second split
    s1s1, s1s2, s1s3 = s1.split([1, 2, 1], randomize=False)

    s1s1_output = []
    for item in s1s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1s1_output.append(item["text"])

    s1s2_output = []
    for item in s1s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1s2_output.append(item["text"])

    s1s3_output = []
    for item in s1s3.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1s3_output.append(item["text"])

    assert s1s1_output == [s1_correct_output[0]]
    assert s1s2_output == [s1_correct_output[1], s1_correct_output[2]]
    assert s1s3_output == [s1_correct_output[3]]

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(item["text"])
    assert s2_output == [text_file_data[3]]

    # randomize in second split
    # the labels outputted by the ShuffleOp for seed 53 is [2, 3, 1, 0]
    shuffled_ids = [2, 3, 1, 0]

    s1s1, s1s2, s1s3 = s1.split([1, 2, 1])

    s1s1_output = []
    for item in s1s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1s1_output.append(item["text"])

    s1s2_output = []
    for item in s1s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1s2_output.append(item["text"])

    s1s3_output = []
    for item in s1s3.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1s3_output.append(item["text"])

    assert s1s1_output == [s1_correct_output[shuffled_ids[0]]]
    assert s1s2_output == [s1_correct_output[shuffled_ids[1]], s1_correct_output[shuffled_ids[2]]]
    assert s1s3_output == [s1_correct_output[shuffled_ids[3]]]

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(item["text"])
    assert s2_output == [text_file_data[3]]

    # Restore configuration num_parallel_workers
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_mappable_invalid_input():
    """
    Feature: Split op
    Description: Test split op using mappable dataset (ManifestDataset) with invalid inputs and
        applying split op on sharded dataset
    Expectation: Error is raised as expected
    """
    d = ds.ManifestDataset(manifest_file)
    split_with_invalid_inputs(d)

    d = ds.ManifestDataset(manifest_file, num_shards=2, shard_id=0)
    with pytest.raises(RuntimeError) as info:
        _, _ = d.split([4, 1])
    assert "Dataset should not be sharded before split" in str(info.value)


def test_mappable_split_general():
    """
    Feature: Split op
    Description: Test split op using mappable dataset (ManifestDataset)
        with absolute rows, exact percentages, and fuzzy percentages
    Expectation: Output is equal to the expected output
    """
    d = ds.ManifestDataset(manifest_file, shuffle=False)
    d = d.take(5)

    # absolute rows
    s1, s2 = d.split([4, 1], randomize=False)

    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))

    assert s1_output == [0, 1, 2, 3]
    assert s2_output == [4]

    # exact percentages
    s1, s2 = d.split([0.8, 0.2], randomize=False)

    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))

    assert s1_output == [0, 1, 2, 3]
    assert s2_output == [4]

    # fuzzy percentages
    s1, s2 = d.split([0.33, 0.67], randomize=False)

    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))

    assert s1_output == [0, 1]
    assert s2_output == [2, 3, 4]


def test_mappable_split_optimized():
    """
    Feature: Split op
    Description: Test optimized split op using mappable dataset (ManifestDataset)
        with absolute rows, exact percentages, and fuzzy percentages
    Expectation: Output is equal to the expected output
    """
    d = ds.ManifestDataset(manifest_file, shuffle=False)

    # absolute rows
    s1, s2 = d.split([4, 1], randomize=False)

    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))

    assert s1_output == [0, 1, 2, 3]
    assert s2_output == [4]

    # exact percentages
    s1, s2 = d.split([0.8, 0.2], randomize=False)

    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))

    assert s1_output == [0, 1, 2, 3]
    assert s2_output == [4]

    # fuzzy percentages
    s1, s2 = d.split([0.33, 0.67], randomize=False)

    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))

    assert s1_output == [0, 1]
    assert s2_output == [2, 3, 4]


def test_mappable_randomize_deterministic():
    """
    Feature: Split op
    Description: Test split op using mappable dataset (ManifestDataset) with randomization
    Expectation: Output is equal to the expected output
    """
    # the labels outputted by ManifestDataset for seed 53 is [0, 1, 3, 4, 2]
    original_seed = config_get_set_seed(53)

    d = ds.ManifestDataset(manifest_file, shuffle=False)
    s1, s2 = d.split([0.8, 0.2])

    for _ in range(10):
        s1_output = []
        for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
            s1_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))

        s2_output = []
        for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
            s2_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))

        # note no overlap
        assert s1_output == [0, 1, 3, 4]
        assert s2_output == [2]

    ds.config.set_seed(original_seed)


def test_mappable_randomize_repeatable():
    """
    Feature: Split op
    Description: Test split op using mappable dataset (ManifestDataset) followed by repeat op
    Expectation: Output is equal to the expected output
    """
    # the labels outputted by ManifestDataset for seed 53 is [0, 1, 3, 4, 2]
    original_seed = config_get_set_seed(53)

    d = ds.ManifestDataset(manifest_file, shuffle=False)
    s1, s2 = d.split([0.8, 0.2])

    num_epochs = 5
    s1 = s1.repeat(num_epochs)
    s2 = s2.repeat(num_epochs)

    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))

    # note no overlap
    assert s1_output == [0, 1, 3, 4] * num_epochs
    assert s2_output == [2] * num_epochs

    ds.config.set_seed(original_seed)


def test_mappable_sharding():
    """
    Feature: Split op
    Description: Test split op using mappable dataset (ManifestDataset) followed by sharding the dataset after split
    Expectation: Output is equal to the expected output
    """
    # set arbitrary seed for repeatability for shard after split
    # the labels outputted by ManifestDataset for seed 53 is [0, 1, 3, 4, 2]
    original_seed = config_get_set_seed(53)

    num_epochs = 5
    first_split_num_rows = 4

    d = ds.ManifestDataset(manifest_file, shuffle=False)
    s1, s2 = d.split([first_split_num_rows, 1])

    distributed_sampler = ds.DistributedSampler(2, 0)
    s1.use_sampler(distributed_sampler)

    s1 = s1.repeat(num_epochs)

    # testing sharding, second dataset to simulate another instance
    d2 = ds.ManifestDataset(manifest_file, shuffle=False)
    d2s1, d2s2 = d2.split([first_split_num_rows, 1])

    distributed_sampler = ds.DistributedSampler(2, 1)
    d2s1.use_sampler(distributed_sampler)

    d2s1 = d2s1.repeat(num_epochs)

    # shard 0
    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))

    # shard 1
    d2s1_output = []
    for item in d2s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        d2s1_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))

    rows_per_shard_per_epoch = 2
    assert len(s1_output) == rows_per_shard_per_epoch * num_epochs
    assert len(d2s1_output) == rows_per_shard_per_epoch * num_epochs

    # verify each epoch that
    #   1. shards contain no common elements
    #   2. the data was split the same way, and that the union of shards equal the split
    correct_sorted_split_result = [0, 1, 3, 4]
    for i in range(num_epochs):
        combined_data = []
        for j in range(rows_per_shard_per_epoch):
            combined_data.append(s1_output[i * rows_per_shard_per_epoch + j])
            combined_data.append(d2s1_output[i * rows_per_shard_per_epoch + j])

        assert sorted(combined_data) == correct_sorted_split_result

    # test other split
    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))

    d2s2_output = []
    for item in d2s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        d2s2_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))

    assert s2_output == [2]
    assert d2s2_output == [2]

    ds.config.set_seed(original_seed)


def test_mappable_get_dataset_size():
    """
    Feature: Split op
    Description: Test split op using mappable dataset (ManifestDataset) followed by get_dataset_size
    Expectation: Output is equal to the expected output
    """
    d = ds.ManifestDataset(manifest_file, shuffle=False)
    s1, s2 = d.split([4, 1])

    assert d.get_dataset_size() == 5
    assert s1.get_dataset_size() == 4
    assert s2.get_dataset_size() == 1


def test_mappable_multi_split():
    """
    Feature: Split op
    Description: Test randomized split op using mappable dataset (ManifestDataset) followed by
        another split op with and without randomization
    Expectation: Output is equal to the expected output
    """
    # the labels outputted by ManifestDataset for seed 53 is [0, 1, 3, 4, 2]
    original_seed = config_get_set_seed(53)

    d = ds.ManifestDataset(manifest_file, shuffle=False)
    s1, s2 = d.split([4, 1])

    s1_correct_output = [0, 1, 3, 4]

    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))
    assert s1_output == s1_correct_output

    # no randomize in second split
    s1s1, s1s2, s1s3 = s1.split([1, 2, 1], randomize=False)

    s1s1_output = []
    for item in s1s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1s1_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))

    s1s2_output = []
    for item in s1s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1s2_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))

    s1s3_output = []
    for item in s1s3.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1s3_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))

    assert s1s1_output == [s1_correct_output[0]]
    assert s1s2_output == [s1_correct_output[1], s1_correct_output[2]]
    assert s1s3_output == [s1_correct_output[3]]

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))
    assert s2_output == [2]

    # randomize in second split
    # the labels outputted by the RandomSampler for seed 53 is [3, 1, 2, 0]
    random_sampler_ids = [3, 1, 2, 0]

    s1s1, s1s2, s1s3 = s1.split([1, 2, 1])

    s1s1_output = []
    for item in s1s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1s1_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))

    s1s2_output = []
    for item in s1s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1s2_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))

    s1s3_output = []
    for item in s1s3.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1s3_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))

    assert s1s1_output == [s1_correct_output[random_sampler_ids[0]]]
    assert s1s2_output == [s1_correct_output[random_sampler_ids[1]], s1_correct_output[random_sampler_ids[2]]]
    assert s1s3_output == [s1_correct_output[random_sampler_ids[3]]]

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))
    assert s2_output == [2]

    ds.config.set_seed(original_seed)


def test_rounding():
    """
    Feature: Split op
    Description: Test split op using mappable dataset (ManifestDataset) with under rounding and over rounding arg
    Expectation: Output is equal to the expected output
    """
    d = ds.ManifestDataset(manifest_file, shuffle=False)

    # under rounding
    s1, s2 = d.split([0.5, 0.5], randomize=False)

    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))

    assert s1_output == [0, 1, 2]
    assert s2_output == [3, 4]

    # over rounding
    s1, s2, s3 = d.split([0.15, 0.55, 0.3], randomize=False)

    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))

    s3_output = []
    for item in s3.create_dict_iterator(num_epochs=1, output_numpy=True):
        s3_output.append(manifest_map.get((item["image"].shape[0], item["label"].item())))

    assert s1_output == [0]
    assert s2_output == [1, 2]
    assert s3_output == [3, 4]


# Run this test in separate process since this test updates shared memory config
@pytest.mark.forked
def test_split_numpyslices_num_workers():
    """
    Feature: Split op
    Description: Test split op when using NumpySlicesDataset(..., num_parallel_workers=2, ...)
    Expectation: Error is raised as expected
    """

    # Note: Since NumpySlicesDataset is derived from GeneratorDataset and GeneratorDataset has
    # python_multiprocessing=True as default, need to disable shared memory when running this test in CI
    # since NumpySlicesDataset using num_parallel_workers > 1.
    # Reduce memory required by disabling the shared memory optimization
    mem_original = ds.config.get_enable_shared_mem()
    ds.config.set_enable_shared_mem(False)

    # construct data and label
    data1 = np.array(np.random.sample(size=(300, 300, 3)) * 255, dtype=np.uint8)
    data2 = np.array(np.random.sample(size=(300, 300, 3)) * 255, dtype=np.uint8)
    data3 = np.array(np.random.sample(size=(300, 300, 3)) * 255, dtype=np.uint8)
    data4 = np.array(np.random.sample(size=(300, 300, 3)) * 255, dtype=np.uint8)

    label = [1, 2, 3, 4]

    # load the data and label by NumpySlicesDataset
    dataset = ds.NumpySlicesDataset(([data1, data2, data3, data4], label), ["data", "label"], num_parallel_workers=2)

    dataset_train, dataset_val = dataset.split([0.5, 0.5])

    # apply the transform to data
    dataset_train = dataset_train.map(operations=vision.RandomCrop(size=(250, 250)), input_columns="data")

    # batch
    dataset_train = dataset_train.batch(batch_size=2)

    # create iterator
    epochs = 2
    ds_iter = dataset_train.create_dict_iterator(output_numpy=True, num_epochs=epochs)
    count = 0
    for _ in range(epochs):
        for item in ds_iter:
            assert item["data"].shape == (2, 250, 250, 3)
            count += 1
    assert count == 2

    # create val iterator
    epochs = 2
    ds_iter = dataset_val.create_dict_iterator(output_numpy=True, num_epochs=epochs)
    count = 0
    for _ in range(epochs):
        for item in ds_iter:
            assert item["data"].shape == (300, 300, 3)
            count += 1
    assert count == 4

    # Restore configuration
    ds.config.set_enable_shared_mem(mem_original)


if __name__ == '__main__':
    test_unmappable_invalid_input()
    test_unmappable_split()
    test_unmappable_randomize_deterministic()
    test_unmappable_randomize_repeatable()
    test_unmappable_get_dataset_size()
    test_unmappable_multi_split()
    test_mappable_invalid_input()
    test_mappable_split_general()
    test_mappable_split_optimized()
    test_mappable_randomize_deterministic()
    test_mappable_randomize_repeatable()
    test_mappable_sharding()
    test_mappable_get_dataset_size()
    test_mappable_multi_split()
    test_rounding()
    test_split_numpyslices_num_workers()
