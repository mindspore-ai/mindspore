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
RandomCrop operator.
"""

from mindspore.dataset.vision import py_transforms
from mindspore.dataset.vision import py_transforms_util
from mindspore.dataset.vision import utils


class RandomCrop(py_transforms.RandomCrop):
    """
    RandomCrop inherits from py_transforms.RandomCrop but derives/uses the
    original image size as the output size.

    Please refer to py_transforms.RandomCrop for argument specifications.
    """

    def __init__(self, padding=4, pad_if_needed=False,
                 fill_value=0, padding_mode=utils.Border.CONSTANT):
        # Note the `1` for the size argument is only set for passing the check.
        super(RandomCrop, self).__init__(1, padding=padding, pad_if_needed=pad_if_needed,
                                         fill_value=fill_value, padding_mode=padding_mode)

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to be padded and then randomly cropped back
                             to the same size.

        Returns:
            img (PIL image), Randomly cropped image.
        """
        if not py_transforms_util.is_pil(img):
            raise TypeError(
                py_transforms_util.augment_error_message.format(type(img)))

        return py_transforms_util.random_crop(
            img, img.size, self.padding, self.pad_if_needed,
            self.fill_value, self.padding_mode,
        )
