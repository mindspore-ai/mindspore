# Copyright 2020 Huawei Technologies Co., Ltd
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
from PIL import Image
import mindspore.dataset as de
import mindspore.dataset.transforms.vision.c_transforms as C

from .ei_dataset import HwVocManifestDataset, HwVocRawDataset
from .utils import custom_transforms as tr


class DataTransform(object):
    """Transform dataset for DeepLabV3."""

    def __init__(self, args, usage):
        self.args = args
        self.usage = usage

    def __call__(self, image, label):
        if "train" == self.usage:
            return self._train(image, label)
        elif "eval" == self.usage:
            return self._eval(image, label)

    def _train(self, image, label):
        image = Image.fromarray(image)
        label = Image.fromarray(label)

        rsc_tr = tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size)
        image, label = rsc_tr(image, label)

        rhf_tr = tr.RandomHorizontalFlip()
        image, label = rhf_tr(image, label)

        nor_tr = tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        image, label = nor_tr(image, label)

        return image, label

    def _eval(self, image, label):
        image = Image.fromarray(image)
        label = Image.fromarray(label)

        fsc_tr = tr.FixScaleCrop(crop_size=self.args.crop_size)
        image, label = fsc_tr(image, label)

        nor_tr = tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        image, label = nor_tr(image, label)

        return image, label


def create_dataset(args, data_url, epoch_num=1, batch_size=1, usage="train"):
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
    if data_url.endswith(".manifest"):
        dataset = HwVocManifestDataset(data_url, usage=usage)
    else:
        dataset = HwVocRawDataset(data_url, usage=usage)
    dataset_len = len(dataset)

    # wrapped with GeneratorDataset
    dataset = de.GeneratorDataset(dataset, ["image", "label"], sampler=None)
    dataset.set_dataset_size(dataset_len)
    dataset = dataset.map(input_columns=["image", "label"], operations=DataTransform(args, usage=usage))

    channelswap_op = C.HWC2CHW()
    dataset = dataset.map(input_columns="image", operations=channelswap_op)

    # 1464 samples / batch_size 8 = 183 batches
    # epoch_num is num of steps
    # 3658 steps / 183 = 20 epochs
    if usage == "train":
        dataset = dataset.shuffle(1464)
    dataset = dataset.batch(batch_size, drop_remainder=(usage == usage))
    dataset = dataset.repeat(count=epoch_num)
    dataset.map_model = 4

    dataset.__loop_size__ = 1
    return dataset
