# Copyright 2021 Huawei Technologies Co., Ltd
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
# ============================================================================

"""ntsnet dataset"""
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as vision
from mindspore.dataset.vision import Inter


def create_dataset_train(train_path, batch_size):
    """create train dataset"""
    train_data_set = ds.ImageFolderDataset(train_path, shuffle=True)
    # define map operations
    transform_img = [
        vision.Decode(),
        vision.Resize([448, 448], Inter.LINEAR),
        vision.RandomHorizontalFlip(),
        vision.HWC2CHW()
    ]
    train_data_set = train_data_set.map(input_columns="image", num_parallel_workers=8, operations=transform_img,
                                        output_columns="image")
    train_data_set = train_data_set.map(input_columns="image", num_parallel_workers=8,
                                        operations=lambda x: (x / 255).astype("float32"))
    train_data_set = train_data_set.batch(batch_size)
    return train_data_set


def create_dataset_test(test_path, batch_size):
    """create test dataset"""
    test_data_set = ds.ImageFolderDataset(test_path, shuffle=False)
    # define map operations
    transform_img = [
        vision.Decode(),
        vision.Resize([448, 448], Inter.LINEAR),
        vision.HWC2CHW()
    ]
    test_data_set = test_data_set.map(input_columns="image", num_parallel_workers=8, operations=transform_img,
                                      output_columns="image")
    test_data_set = test_data_set.map(input_columns="image", num_parallel_workers=8,
                                      operations=lambda x: (x / 255).astype("float32"))
    test_data_set = test_data_set.batch(batch_size)
    return test_data_set
