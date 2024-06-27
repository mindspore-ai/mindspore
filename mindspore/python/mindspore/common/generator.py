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
from mindspore.common.parameter import Parameter
from mindspore.common import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops.operations._inner_ops import Generator as GeneratorOp
from mindspore.ops.operations._sequence_ops import TensorToScalar
from mindspore.ops.vm_impl_registry import vm_impl_registry as vm_impl_getters
from mindspore._c_expression import _random_seeded_generator


# pylint: disable=unused-argument
@vm_impl_getters.register(GeneratorOp)
def vm_impl_generator(self):
    """
    Generate vm_impl function for Generator.

    The default_generator is initialized and called during import.
    For this reason a vm_impl is required for ut compilation.
    """
    def vm_impl(cmd, inputs):
        return 0
    return vm_impl


STEP = 0
SEED = 1
GET_STATE = 2
SET_STATE = 3
MANUAL_SEED = 4
INITIAL_SEED = 5


def jit_class(cls):
    """Make class recognizable in graph mode"""
    setattr(cls, '__ms_class__', True)
    return cls


@jit_class
class Generator:
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
        >>> from mindspore import Generator
        >>> generator = Generator()
        >>> generator.manual_seed(5)
        >>> print(generator.initial_seed())
        5
        >>> state = generator.get_state()
        >>> generator.seed()
        >>> generator.set_state(state)
        >>> print(generator.initial_seed())
        5
    """

    def __init__(self):
        self._seed = Parameter(Tensor(0, mstype.int64),
                               name="seed", requires_grad=False)
        self._offset = Parameter(
            Tensor(0, mstype.int64), name="offset", requires_grad=False)

        self._generator = GeneratorOp().set_device("CPU")
        self._to_scalar = TensorToScalar()

    def set_state(self, state):
        """
        Sets the generator state.

        Args:
            state (tensor): target state of the generator.
        """
        self._generator(SET_STATE, (self._seed, self._offset, state))

    def get_state(self):
        """
        Get the generator state.

        Returns:
            Tensor, generator state.
        """
        return self._generator(GET_STATE, (self._seed, self._offset))[2]

    def seed(self):  # pylint: disable=redefined-outer-name
        """
        Seed generator with random number.

        Returns:
            Randomly generated seeds, the type is int.
        """
        current_seed = self._generator(
            SEED, (self._seed, self._offset))[0]
        return self._to_scalar(current_seed)

    def manual_seed(self, seed):  # pylint: disable=redefined-outer-name
        """
        Set the generator seed.

        Args:
            seed (int): Set the generator seed.

        Returns:
            Generator, the generator instance.
        """
        seed = Tensor(seed, mstype.int64)
        self._generator(MANUAL_SEED, (self._seed, self._offset, seed))
        return self

    def initial_seed(self):
        """
        Return the initial seed of generator.

        Returns:
            The initial seed of generator.
        """
        current_seed = self._generator(INITIAL_SEED, (self._seed,))[0]
        return self._to_scalar(current_seed)


    def _step(self, step):
        """
        Return current seed and offset, and update offset for the next call.

        Args:
            step (Tensor): Update offset by step.

        Returns:
            Current seed and offset.
        """
        return self._generator(STEP, (self._seed, self._offset, step,))[:2]


default_generator = _random_seeded_generator()


def seed():  # pylint: disable=redefined-outer-name
    """
    Seed the default generator with random number.

    Returns:
        Randomly generated seeds, the type is int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import seed
        >>> print(seed())
        1663920602
    """
    return default_generator.seed()


def manual_seed(seed):  # pylint: disable=redefined-outer-name
    """
    Set the default generator seed.

    Args:
        seed (int): Set the default generator seed.

    Returns:
        Generator, the default generator.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import manual_seed, initial_seed
        >>> manual_seed(13)
        >>> print(initial_seed())
        13
    """
    default_generator.manual_seed(seed)


def initial_seed():
    """
    Return the initial seed of the default generator.

    Returns:
        The initial seed of the default generator.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import manual_seed, initial_seed
        >>> manual_seed(14)
        >>> print(initial_seed())
        14
    """
    return default_generator.initial_seed()


def get_rng_state():
    """
    Get the state of the default generator.

    Returns:
        Tensor, generator state.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import get_rng_state
        >>> state = get_rng_state()
    """
    return default_generator.get_state()


def set_rng_state(state):  # pylint: disable=redefined-outer-name
    """
    Set the state of the default generator.

    Args:
        state (Tensor): the target state

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import set_rng_state, get_rng_state
        >>> state = get_rng_state()
        >>> set_rng_state(state)
    """
    default_generator.set_state(state)


__all__ = ["Generator", "default_generator", "seed", "manual_seed", "initial_seed", "set_rng_state", "get_rng_state"]
