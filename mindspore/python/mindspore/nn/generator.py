# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Generator"""
import os

import numpy as np

from mindspore import context
from mindspore.common.parameter import Parameter
from mindspore.nn.cell import Cell
from mindspore.ops.operations import Assign, AssignAdd, Depend


class Generator(Cell):
    """
    A generator that manages the state of random numbers and provides seed and offset for random functions.
    When the seed and offset are fixed, the random function generates the same random sequence.

    Inputs:
        - **step** (int) - Set the step size for offset update.

    Outputs:
        Tuple consisting of the seed and offset of generator.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore.nn import Generator
        >>> import numpy as np
        >>> np.random.seed(10)
        >>> ms.set_context(mode=1)
        >>> generator = Generator()
        >>> print(generator.get_state())
        (Tensor(shape=[], dtype=Int32, value= 0), Tensor(shape=[], dtype=Int32, value= 0))
        >>> print(generator(12))
        (0, 0)
        >>> print(generator.get_state())
        (Tensor(shape=[], dtype=Int32, value= 0), Tensor(shape=[], dtype=Int32, value= 12))
        >>> generator.manual_seed(20)
        >>> print(generator.get_state())
        (Tensor(shape=[], dtype=Int32, value= 20), Tensor(shape=[], dtype=Int32, value= 0))
        >>> print(generator.seed())
        1165313289
        >>> print(generator.initial_seed())
        1165313289
    """

    def __init__(self):
        super(Generator, self).__init__()
        self._assign = Assign().set_device("CPU")
        self._assign_add = AssignAdd().set_device("CPU")
        self._depend = Depend()
        self._seed = Parameter(0, name="seed", requires_grad=False)
        self._offset = Parameter(0, name="offset", requires_grad=False)
        self._seed_val = 0
        self._offset_val = 0

    def set_state(self, seed, offset=None):  # pylint: disable=redefined-outer-name
        """
        Sets the generator state.

        Args:
            seed (int): Seed of the generator.
            offset (int, optional): Offset of the generator, default: ``None`` , means ``0``.
        """
        self._seed_val = int(seed)
        self._assign(self._seed, self._seed_val)
        if offset is None:
            offset = 0
        self._offset_val = int(offset)
        self._assign(self._offset, self._offset_val)

    def get_state(self):
        """
        Get the generator state.

        Returns:
            Tuple consisting of the seed and offset of generator.
        """
        return self._seed.value(), self._offset.value()

    def seed(self):  # pylint: disable=redefined-outer-name
        """
        Generate random seeds that can be used as seeds for generator.

        Returns:
            Tensor, randomly generated seeds.
        """
        seed_ = np.random.randint(np.iinfo(np.int32).min, np.iinfo(np.int32).max)
        self.set_state(seed_)
        return self._seed.value()

    def manual_seed(self, seed):  # pylint: disable=redefined-outer-name
        """
        Sets the generator seed.

        Args:
            seed (int): Sets the generator seed.

        Returns:
            The generator self.
        """
        self.set_state(seed)
        return self

    def initial_seed(self):
        """
        Return the initial seed of generator.

        Returns:
            The initial seed of generator.
        """
        return self._seed.value()

    def construct(self, step):
        """
        Update the value of offset, and return the seed and the previous offset.

        Args:
            step (int): Update offset by step.

        Returns:
            Seed and offset before update.
        """
        offset = self._offset.value()
        step = self._depend(step, offset)
        self._assign_add(self._offset, step)
        return self._seed.value(), offset

    def __call__(self, step):
        if os.getenv("MS_JIT") != '0' and context.get_context("mode") == context.GRAPH_MODE:
            return super().__call__(step)

        offset_val = self._offset_val
        self._offset_val += step
        self._offset.set_data(self._offset_val)
        return self._seed_val, offset_val


default_generator_ = None


def _init_default_generator():
    global default_generator_
    default_generator_ = Generator()
    default_generator_.seed()


def default_generator():
    """
    Return the default generator object.

    When the user does not specify generator, the random operator invokes default generator to generate random numbers.

    Returns:
        The default generator.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore.nn import default_generator
        >>> default_gen = default_generator()
        >>> print(type(default_gen))
        <class 'mindspore.nn.generator.Generator'>
    """
    if default_generator_ is None:
        _init_default_generator()
    return default_generator_


def seed():  # pylint: disable=redefined-outer-name
    """
    Generate random seeds that can be used as seeds for default generator.

    Returns:
        Randomly generated seeds.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore.nn import seed
        >>> np.random.seed(20)
        >>> print(seed())
        1663920602
    """
    if default_generator_ is None:
        _init_default_generator()
    return default_generator_.seed()


def manual_seed(seed):  # pylint: disable=redefined-outer-name
    """
    Sets the default generator seed.

    Args:
        seed (int): Sets the default generator seed.

    Returns:
        The default generator self.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore.nn import manual_seed, initial_seed
        >>> manual_seed(13)
        >>> print(initial_seed())
        13
    """
    if default_generator_ is None:
        _init_default_generator()
    default_generator_.manual_seed(seed)


def initial_seed():
    """
    Return the initial seed of default generator.

    Returns:
        The initial seed of default generator.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore.nn import manual_seed, initial_seed
        >>> manual_seed(14)
        >>> print(initial_seed())
        14
    """
    if default_generator_ is None:
        _init_default_generator()
    return default_generator_.initial_seed()


def get_rng_state():
    """
    Get the default generator state.

    Returns:
        Tuple consisting of the seed and offset of default generator.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore.nn import get_rng_state
        >>> np.random.seed(20)
        >>> print(get_rng_state())
        (Tensor(shape=[], dtype=Int32, value= 378518883), Tensor(shape=[], dtype=Int32, value= 0))
    """
    if default_generator_ is None:
        _init_default_generator()
    return default_generator_.get_state()


def set_rng_state(seed, offset=None):  # pylint: disable=redefined-outer-name
    """
    Sets the default generator state.

    Args:
        seed (int): Seed of the default generator.
        offset (int, optional): Offset of the default generator, default: ``None`` , means ``0``.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore.nn import set_rng_state, get_rng_state
        >>> set_rng_state(10)
        >>> print(get_rng_state())
        (Tensor(shape=[], dtype=Int32, value= 10), Tensor(shape=[], dtype=Int32, value= 0))
    """
    if default_generator_ is None:
        _init_default_generator()
    default_generator_.set_state(seed, offset)


__all__ = ["Generator", "default_generator", "seed", "manual_seed", "initial_seed", "set_rng_state", "get_rng_state"]
