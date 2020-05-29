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
"""Process Dataset."""
import abc
import os
import time

from .utils.adapter import get_raw_samples, read_image


class BaseDataset:
    """
    Create dataset.

    Args:
        data_url (str): The path of data.
        usage (str): Whether to use train or eval (default='train').

    Returns:
        Dataset.
    """
    def __init__(self, data_url, usage):
        self.data_url = data_url
        self.usage = usage
        self.cur_index = 0
        self.samples = []
        _s_time = time.time()
        self._load_samples()
        _e_time = time.time()
        print(f"load samples success~, time cost = {_e_time - _s_time}")

    def __getitem__(self, item):
        sample = self.samples[item]
        return self._next_data(sample)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _next_data(sample):
        image_path = sample[0]
        mask_image_path = sample[1]

        image = read_image(image_path)
        mask_image = read_image(mask_image_path)
        return [image, mask_image]

    @abc.abstractmethod
    def _load_samples(self):
        pass


class HwVocRawDataset(BaseDataset):
    """
    Create dataset with raw data.

    Args:
        data_url (str): The path of data.
        usage (str): Whether to use train or eval (default='train').

    Returns:
        Dataset.
    """
    def __init__(self, data_url, usage="train"):
        super().__init__(data_url, usage)

    def _load_samples(self):
        try:
            self.samples = get_raw_samples(os.path.join(self.data_url, self.usage))
        except Exception as e:
            print("load HwVocRawDataset failed!!!")
            raise e
