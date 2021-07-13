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
The Augment operator.
"""

import random

import mindspore.dataset.vision.py_transforms as py_trans

from .third_party.policies import good_policies
from .ops import OperatorClasses, RandomCutout


class Augment:
    """
    Augment acts as a single transformation operator and applies policies found
    by AutoAugment.

    Args:
        index (int or None): If index is not None, the indexed policy would
            always be used. Otherwise, a policy would be randomly chosen from
            the policies set for each image.
        policies (policies found by AutoAugment or None): A set of policies
            to sample from. When the given policies is None, good policies found
            on cifar10 would be used.
        enable_basic (bool): Whether to apply basic augmentations after
                             auto-augment or not. Note that basic augmentations
                             include RandomFlip, RandomCrop, and RandomCutout.
        from_pil (bool): Whether the image passed to the operator is already a
                         PIL image.
        as_pil (bool): Whether the returned image should be kept as a PIL image.
        mean, std (list): Per-channel mean and std used to normalize the output
                          image. Only applicable when as_pil is False.
    """

    def __init__(
            self, index=None, policies=None, enable_basic=True,
            from_pil=False, as_pil=False,
            mean=None, std=None,
    ):
        self.index = index
        if policies is None:
            self.policies = good_policies()
        else:
            self.policies = policies

        self.oc = OperatorClasses()
        self.to_pil = py_trans.ToPIL()
        self.to_tensor = py_trans.ToTensor()

        self.enable_basic = enable_basic
        self.random_crop = self.oc.RandomCrop(None)
        self.random_flip = self.oc.RandomHorizontalFlip(None)
        self.cutout = RandomCutout(size=16, value=(0, 0, 0))

        self.from_pil = from_pil
        self.as_pil = as_pil
        self.normalize = None
        if mean is not None and std is not None:
            self.normalize = py_trans.Normalize(mean, std)

    def _apply(self, name, prob, level, img):
        if random.random() > prob:
            # Untouched
            return img
        # Apply the operator
        return getattr(self.oc, name)(level)(img)

    def __call__(self, img):
        """
        Call method.

        Args:
            img (raw image or PIL image): Image to be auto-augmented.

        Returns:
            img (PIL image or Tensor), Auto-augmented image.
        """
        if self.index is None:
            policy = random.choice(self.policies)
        else:
            policy = self.policies[self.index]

        if not self.from_pil:
            img = self.to_pil(img)

        for name, prob, level in policy:
            img = self._apply(name, prob, level, img)

        if self.enable_basic:
            img = self.random_crop(img)
            img = self.random_flip(img)
            img = self.cutout(img)

        if not self.as_pil:
            img = self.to_tensor(img)
            if self.normalize is not None:
                img = self.normalize(img)
        return img
