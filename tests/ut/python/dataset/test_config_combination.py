# Copyright 2023 Huawei Technologies Co., Ltd
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
Test combination of different configuration manager settings
"""

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from tests.dataset_mock import celeba_source

ORIGINAL_SEED = 0  # Seed will be set internally in debug mode. Save original seed value to restore.


def setup_function():
    # Save the original seed, since debug mode sets the seed (for deterministic results) if default seed is detected.
    global ORIGINAL_SEED
    ORIGINAL_SEED = ds.config.get_seed()


def teardown_function():
    # Restore the original seed
    ds.config.set_seed(ORIGINAL_SEED)


def create_celeba_pipeline(offload=False, to_pil=False):
    """ Create simple CelebADataset pipeline, suitable for offloading. """
    data1 = celeba_source("../data/dataset/testCelebAData/")
    data1 = data1.map(operations=[vision.Decode(to_pil=to_pil)], input_columns=["image"])
    data1 = data1.map(operations=[vision.Resize((30, 30))], input_columns=["image"])
    data1 = data1.map(operations=[vision.Rescale(1.0 / 255.0, -1.0),
                                  vision.RandomHorizontalFlip(0.9)],
                      input_columns=["image"], offload=offload)
    data1 = data1.batch(batch_size=2)
    return data1


def iterate_celeba_pipeline(data1):
    """ Iterate over data pipeline created from create_celeba_pipeline. """
    row_count = 0
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 2
        assert item[0].shape == (2, 30, 30, 3)
        assert item[1].shape == (2, 40)
        row_count += 1
    assert row_count == 2


def create_imagefolder_pipeline(data_dir, offload=False, to_pil=False):
    """ Create simple ImageFolderDataset pipeline, suitable for offloading. """
    data1 = ds.ImageFolderDataset(data_dir, num_samples=None, num_parallel_workers=1, shuffle=False)
    data1 = data1.map(operations=[vision.Decode(to_pil=to_pil)], input_columns=["image"])
    data1 = data1.map(operations=[vision.Resize((40, 50))], input_columns=["image"])
    data1 = data1.map(operations=[vision.RandomVerticalFlip(0.8)], input_columns=["image"], offload=offload)
    return data1


def iterate_imagefolder_pipeline(data1):
    """ Iterate over data pipeline created from create_imagefolder_pipeline. """
    row_count = 0
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        row_count += 1
    assert row_count == 6


def run_imagefolder_pyop_pipeline(python_multiprocessing=False, num_epoch=2):
    """ Create and execute ImageFolderDataset pipeline with Python implemented ops. """
    # Define dataset pipeline
    data = ds.ImageFolderDataset("../data/dataset/testImageNetData4/train", decode=True)
    num_repeat = 3
    sample_row = 7
    data = data.repeat(num_repeat)
    # Apply map ops with Python implemented ops
    data = data.map(operations=[(lambda x: x + 1)], input_columns=["image"],
                    python_multiprocessing=python_multiprocessing)
    data = data.map(operations=[lambda image: image.copy()], input_columns=["image"],
                    python_multiprocessing=python_multiprocessing)

    # Create and execute iterator
    epoch_count = 0
    sample_count = 0
    iter1 = data.create_dict_iterator(num_epochs=num_epoch)
    for _ in range(num_epoch):
        num_rows = 0
        for row_item in iter1:
            assert len(row_item) == 2
            assert row_item["image"].shape == (384, 682, 3)
            num_rows += 1
        assert num_rows == sample_row * num_repeat
        sample_count += num_rows
        epoch_count += 1
    assert epoch_count == num_epoch
    assert sample_count == num_repeat * num_epoch * sample_row
