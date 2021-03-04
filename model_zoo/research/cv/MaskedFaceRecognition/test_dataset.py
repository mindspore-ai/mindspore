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
"""
create train or eval dataset.
"""
import mindspore.common.dtype as mstype
import mindspore.dataset.engine as de
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from config import config
from dataset.Dataset import Dataset


def create_dataset(data_dir, p=16, k=8):
    """
    create a train or eval dataset

    Args:
        dataset_path(string): the path of dataset.
        p(int): randomly choose p classes from all classes.
        k(int): randomly choose k images from each of the chosen p classes.
                p * k is the batchsize.

    Returns:
        dataset
    """
    dataset = Dataset(data_dir)
    de_dataset = de.GeneratorDataset(dataset, ["image", "label1", "label2"])

    resize_height = config.image_height
    resize_width = config.image_width
    rescale = 1.0 / 255.0
    shift = 0.0

    resize_op = CV.Resize((resize_height, resize_width))
    rescale_op = CV.Rescale(rescale, shift)
    normalize_op = CV.Normalize([0.486, 0.459, 0.408], [0.229, 0.224, 0.225])

    change_swap_op = CV.HWC2CHW()

    trans = []

    trans += [resize_op, rescale_op, normalize_op, change_swap_op]

    type_cast_op_label1 = C.TypeCast(mstype.int32)
    type_cast_op_label2 = C.TypeCast(mstype.float32)

    de_dataset = de_dataset.map(input_columns="label1", operations=type_cast_op_label1)
    de_dataset = de_dataset.map(input_columns="label2", operations=type_cast_op_label2)
    de_dataset = de_dataset.map(input_columns="image", operations=trans)
    de_dataset = de_dataset.batch(p*k, drop_remainder=False)

    return de_dataset
