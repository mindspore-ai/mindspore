# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
Test Compose op in Dataset
"""
import numpy as np
import mindspore.dataset as ds
from mindspore.dataset import vision

from util import visualize_list, save_and_check_md5_pil, config_get_set_seed, config_get_set_num_parallel_workers

GENERATE_GOLDEN = False








def test_c_py_compose_vision_module(plot=False, run_golden=True):
    """
    Feature: Compose Op
    Description: Test Compose op combining Python and C++ vision transforms
    Expectation: Dataset pipeline runs successfully, results are visually verified and md5 results are verified
    """
    original_seed = config_get_set_seed(10)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    def test_config(plot, file_name, op_list):
        data_dir = "../data/dataset/testImageNetData/train/"
        data1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data1 = data1.map(operations=op_list, input_columns=["image"])
        data2 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data2 = data2.map(operations=vision.Decode(), input_columns=["image"])
        original_images = []
        transformed_images = []

        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            transformed_images.append(item["image"])
        for item in data2.create_dict_iterator(num_epochs=1, output_numpy=True):
            original_images.append(item["image"])

        if run_golden:
            # Compare with expected md5 from images
            save_and_check_md5_pil(
                data1, file_name, generate_golden=GENERATE_GOLDEN)

        if plot:
            visualize_list(original_images, transformed_images)

    test_config(op_list=[vision.Decode(),
                         vision.ToPIL(),
                         vision.Resize((224, 224)),
                         vision.ToNumpy()],
                plot=plot, file_name="compose_c_py_1.npz")

    test_config(op_list=[vision.Decode(),
                         vision.Resize((224, 244)),
                         vision.ToPIL(),
                         vision.ToNumpy(),
                         vision.Resize((24, 24))],
                plot=plot, file_name="compose_c_py_2.npz")

    test_config(op_list=[vision.Decode(True),
                         vision.Resize((224, 224)),
                         np.array,
                         vision.RandomColor()],
                plot=plot, file_name="compose_c_py_3.npz")

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers((original_num_parallel_workers))




def test_compose_with_custom_function():
    """
    Feature: Compose op
    Description: Test Python Compose op with custom function
    Expectation: Output is equal to the expected value
    """

    def custom_function(x):
        return (x, x * x)

    # First dataset
    op_list = [
        lambda x: x * 3,
        custom_function,
        # convert two column output to one
        lambda *images: np.stack(images)
    ]

    data = ds.NumpySlicesDataset([[1, 2]], column_names=[
        "col0"], shuffle=False)
    data = data.map(input_columns=["col0"], operations=op_list)
    #

    res = []
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        res.append(i["col0"].tolist())
    assert res == [[[3, 6], [9, 36]]]
