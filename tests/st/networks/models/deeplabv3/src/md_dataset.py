# Copyright 2020-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# httpwww.apache.orglicensesLICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Dataset module."""
import numpy as np
from PIL import Image
import mindspore.dataset as de
import mindspore.dataset.vision as C

from .ei_dataset import HwVocRawDataset
from .utils import custom_transforms as tr


class DataTransform:
    """Transform dataset for DeepLabV3."""

    def __init__(self, args, usage):
        self.args = args
        self.usage = usage

    def __call__(self, image, label):
        if self.usage == "train":
            return self._train(image, label)
        if self.usage == "eval":
            return self._eval(image, label)
        return None

    def _train(self, image, label):
        """
        Process training data.

        Args:
            image (list): Image data.
            label (list): Dataset label.
        """
        image = Image.fromarray(image)
        label = Image.fromarray(label)

        rsc_tr = tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size)
        image, label = rsc_tr(image, label)

        rhf_tr = tr.RandomHorizontalFlip()
        image, label = rhf_tr(image, label)

        image = np.array(image).astype(np.float32)
        label = np.array(label).astype(np.float32)

        return image, label

    def _eval(self, image, label):
        """
        Process eval data.

        Args:
            image (list): Image data.
            label (list): Dataset label.
        """
        image = Image.fromarray(image)
        label = Image.fromarray(label)

        fsc_tr = tr.FixScaleCrop(crop_size=self.args.crop_size)
        image, label = fsc_tr(image, label)

        image = np.array(image).astype(np.float32)
        label = np.array(label).astype(np.float32)

        return image, label


def create_dataset(args, data_url, epoch_num=1, batch_size=1, usage="train", shuffle=True):
    """
    Create Dataset for DeepLabV3.

    Args:
        args (dict): Train parameters.
        data_url (str): Dataset path.
        epoch_num (int): Epoch of dataset (default=1).
        batch_size (int): Batch size of dataset (default=1).
        usage (str): Whether is use to train or eval (default='train').

    Returns:
        Dataset.
    """
    # create iter dataset
    dataset = HwVocRawDataset(data_url, usage=usage)

    # wrapped with GeneratorDataset
    dataset = de.GeneratorDataset(dataset, ["image", "label"], sampler=None)
    dataset = dataset.map(operations=DataTransform(args, usage=usage), input_columns=["image", "label"])

    channelswap_op = C.HWC2CHW()
    dataset = dataset.map(operations=channelswap_op, input_columns="image")

    # 1464 samples / batch_size 8 = 183 batches
    # epoch_num is num of steps
    # 3658 steps / 183 = 20 epochs
    if usage == "train" and shuffle:
        dataset = dataset.shuffle(1464)
    dataset = dataset.batch(batch_size, drop_remainder=(usage == "train"))
    dataset = dataset.repeat(count=epoch_num)

    return dataset
