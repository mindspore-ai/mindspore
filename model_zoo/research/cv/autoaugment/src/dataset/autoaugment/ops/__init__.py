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
Package initialization for custom PIL operators.
"""

from mindspore.dataset.vision import py_transforms

from .crop import RandomCrop
from .cutout import RandomCutout
from .effect import (
    Posterize,
    Solarize,
)
from .enhance import (
    Brightness,
    Color,
    Contrast,
    Sharpness,
)
from .transform import (
    Rotate,
    ShearX,
    ShearY,
    TranslateX,
    TranslateY,
)


class OperatorClasses:
    """OperatorClasses gathers all unary-image transformations listed in the
    Table 6 of https://arxiv.org/abs/1805.09501 and uses discrte levels for
    these transformations (The Sample Pairing transformation is an
    exception, which involes multiple images from a single mini-batch and
    is not exploited in this implementation.)

    Additionally, there are RandomHorizontalFlip and RandomCrop.
    """

    def __init__(self):
        self.Rotate = self.decorate(Rotate, max_val=30, rounding=True)
        self.ShearX = self.decorate(ShearX, max_val=0.3)
        self.ShearY = self.decorate(ShearY, max_val=0.3)
        self.TranslateX = self.decorate(TranslateX, max_val=10, rounding=True)
        self.TranslateY = self.decorate(TranslateY, max_val=10, rounding=True)

        self.AutoContrast = self.decorate(py_transforms.AutoContrast)
        self.Invert = self.decorate(py_transforms.Invert)
        self.Equalize = self.decorate(py_transforms.Equalize)

        self.Solarize = self.decorate(
            Solarize, max_val=256, rounding=True, post=lambda x: 256 - x)
        self.Posterize = self.decorate(
            Posterize, max_val=4, rounding=True, post=lambda x: 4 - x)

        def post(x):
            """Post operation to avoid 0 value."""
            return x + 0.1
        self.Brightness = self.decorate(Brightness, max_val=1.8, post=post)
        self.Color = self.decorate(Color, max_val=1.8, post=post)
        self.Contrast = self.decorate(Contrast, max_val=1.8, post=post)
        self.Sharpness = self.decorate(Sharpness, max_val=1.8, post=post)

        self.Cutout = self.decorate(RandomCutout, max_val=20, rounding=True)

        self.RandomHorizontalFlip = self.decorate(
            py_transforms.RandomHorizontalFlip)
        self.RandomCrop = self.decorate(RandomCrop)

    def vars(self):
        """vars returns all available operators as a dictionary."""
        return vars(self)

    def decorate(self, op, max_val=None, rounding=False, post=None):
        """
        decorate interprets discrete levels for the given operator when
        applicable.

        Args:
            op (Augmentation Operator): Operator to be decorated.
            max_val (int or float): Maximum value level-10 corresponds to.
            rounding (bool): Whether the corresponding value should be rounded
                             to an integer.
            post (function): User-defined post-processing value function.

        Returns:
            Decorated operator.
        """
        if max_val is None:
            def no_arg_fn(_):
                """Decorates an operator without level parameter."""
                return op()
            return no_arg_fn

        def fn(level):
            """Decorates an operator with level parameter."""
            val = max_val * level / 10
            if rounding:
                val = int(val)
            if post is not None:
                val = post(val)
            return op(val)
        return fn
