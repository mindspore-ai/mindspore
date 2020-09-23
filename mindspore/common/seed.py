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
# ============================================================================
"""Provide random seed api."""
import numpy as np
import mindspore.dataset as de

# constants
_MAXINT32 = 2**31 - 1
keyConstant = [3528531795, 2654435769, 3449720151, 3144134277]

# set global RNG seed
_GLOBAL_SEED = None
_KERNEL_SEED = {}

def set_seed(seed):
    """
    Set global random seed.

    Note:
        The global seed is used by numpy.random, mindspore.common.Initializer, mindspore.ops.composite.random_ops and
        mindspore.nn.probability.distribution.

        If global seed is not set, these packages will use their own default seed independently, numpy.random and
        mindspore.common.Initializer will choose a random seed, mindspore.ops.composite.random_ops and
        mindspore.nn.probability.distribution will use zero.

        Seed set by numpy.random.seed() only used by numpy.random, while seed set by this API will also used by
        numpy.random, so just set all seed by this API is recommended.

    Args:
        seed (int): The seed to be set.

    Raises:
        ValueError: If seed is invalid (< 0).
        TypeError: If seed isn't a int.
    """
    if not isinstance(seed, int):
        raise TypeError("The seed must be type of int.")
    if seed < 0:
        raise ValueError("The seed must be greater or equal to 0.")
    np.random.seed(seed)
    de.config.set_seed(seed)
    _reset_op_seed()
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed


def get_seed():
    """
    Get global random seed.
    """
    return _GLOBAL_SEED


def _truncate_seed(seed):
    """
    Truncate the seed with MAXINT32.

    Args:
        seed (int): The seed to be truncated.
    """
    return seed % _MAXINT32  # Truncate to fit into 32-bit integer


def _update_seeds(op_seed, kernel_name):
    """
    Update the seed every time when a random op is called.

    Args:
        seed (int): The op-seed to be updated.
        kernel_name (string): The random op kernel.
    """
    global _GLOBAL_SEED
    global _KERNEL_SEED
    if _GLOBAL_SEED is not None:
        _GLOBAL_SEED += keyConstant[1] + keyConstant[3] * (2**8)
    if op_seed is not None:
        _KERNEL_SEED[(kernel_name, op_seed)] = _KERNEL_SEED[(kernel_name, op_seed)] + (keyConstant[0] ^ keyConstant[2])


def _get_op_seed(op_seed, kernel_name):
    """
    Get op seed which is relating to the specific kernel.
    If the seed does not exist, add it into the kernel's dictionary.

    Args:
        seed (int): The op-seed to be updated.
        kernel_name (string): The random op kernel.
    """
    if (kernel_name, op_seed) not in _KERNEL_SEED:
        _KERNEL_SEED[(kernel_name, op_seed)] = op_seed
    return _KERNEL_SEED[(kernel_name, op_seed)]

def _reset_op_seed():
    """
    Reset op seeds in the kernel's dictionary.
    """
    for (kernel_name, op_seed) in _KERNEL_SEED:
        _KERNEL_SEED[(kernel_name, op_seed)] = op_seed
