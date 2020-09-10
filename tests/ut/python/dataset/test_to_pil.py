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
Testing ToPIL op in DE
"""
import mindspore.dataset as ds
import mindspore.dataset.transforms.py_transforms
import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore import log as logger
from util import save_and_check_md5

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_to_pil_01():
    """
    Test ToPIL Op with md5 comparison: input is already PIL image
    Expected to pass
    """
    logger.info("test_to_pil_01")

    # Generate dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        py_vision.Decode(),
        # If input is already PIL image.
        py_vision.ToPIL(),
        py_vision.CenterCrop(375),
        py_vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
    data1 = data1.map(operations=transform, input_columns=["image"])

    # Compare with expected md5 from images
    filename = "to_pil_01_result.npz"
    save_and_check_md5(data1, filename, generate_golden=GENERATE_GOLDEN)

def test_to_pil_02():
    """
    Test ToPIL Op with md5 comparison: input is not PIL image
    Expected to pass
    """
    logger.info("test_to_pil_02")

    # Generate dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = c_vision.Decode()
    transforms = [
        # If input type is not PIL.
        py_vision.ToPIL(),
        py_vision.CenterCrop(375),
        py_vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=transform, input_columns=["image"])

    # Compare with expected md5 from images
    filename = "to_pil_02_result.npz"
    save_and_check_md5(data1, filename, generate_golden=GENERATE_GOLDEN)

if __name__ == "__main__":
    test_to_pil_01()
    test_to_pil_02()
