# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
from __future__ import absolute_import

import numpy as np
from mindspore import _checkparam as Validator

# constants
DEFAULT_GRAPH_SEED = 87654321
_MAXINT32 = 2**31 - 1
keyConstant = [3528531795, 2654435769, 3449720151, 3144134277]

# set global RNG seed
_GLOBAL_SEED = None
_KERNEL_SEED = {}


def _reset_op_seed():
    """
    Reset op seeds in the kernel's dictionary.
    """
    for (kernel_name, op_seed) in list(_KERNEL_SEED):
        _KERNEL_SEED[(kernel_name, op_seed)] = op_seed


def set_seed(seed):
    """
    Set global seed.

    Note:
        - The global seed is used by numpy.random, mindspore.common.Initializer and random operators that combine
          global and operator-level seeds, such as :func:`mindspore.ops.uniform` and
          :func:`mindspore.ops.standard_laplace`.

        - If global seed is not set, these packages will use their own default seed independently, numpy.random and
          mindspore.common.Initializer will choose a random seed.

        - For operators that combine global and operator-level seeds, an unset global seed contributes ``0``;
          an operator seed of ``None`` is also treated as ``0``. If both are zero, the seed pair ``(0, 0)`` lets
          the kernel choose a random seed. It does not request a reproducible sequence seeded with zero.
          If only a positive operator seed is specified, the global component remains zero and the operator's
          seed counter advances on successive calls. Repeating the same call sequence in a fresh process can
          reproduce the results on a backend that supports reproducibility, under the same execution conditions.

        - Explicitly calling ``set_seed(0)`` differs from leaving the global seed unset: for these operators,
          zero is mapped to an internal fixed default graph seed, while :func:`mindspore.get_seed` still returns
          ``0``. Operators using a ``Generator`` follow their own generator state; refer to their documentation.

        - Seed set by numpy.random.seed() is only used by numpy.random, while seed set by this API will also be used by
          numpy.random, so it is recommended to set all seeds using this API.

        - In semi_auto_parallel/auto_parallel mode, when using set_seed, weights with same shape and same sharding
          strategy on the same device would be initialized to the same result, otherwise, they would be initialized to
          the different results.

    Args:
        seed (int): The seed to be set.

    Raises:
        ValueError: If seed is invalid (< 0).
        TypeError: If seed isn't an integer.

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Tensor, set_seed, Parameter, ops
        >>> from mindspore.common.initializer import initializer
        >>> # Each numbered case below describes an independent scenario. Cases 1, 3 and 6 assume
        >>> # a fresh process in which set_seed() has not been called; do not run all cases sequentially.
        >>> # Note: (1) Please make sure the code is running in PYNATIVE MODE;
        >>> # (2) Because Composite-level ops need parameters to be Tensors, for below examples,
        >>> # when using ops.uniform operator, minval and maxval are initialized as:
        >>> minval = Tensor(1.0, ms.float32)
        >>> maxval = Tensor(2.0, ms.float32)
        >>>
        >>> # 1. If global seed is not set, numpy.random and initializer will choose a random seed:
        >>> np_1 = np.random.normal(0, 1, [1]).astype(np.float32) # A1
        >>> np_1 = np.random.normal(0, 1, [1]).astype(np.float32) # A2
        >>> w1 = Parameter(initializer("uniform", [2, 2], ms.float32), name="w1") # W1
        >>> w1 = Parameter(initializer("uniform", [2, 2], ms.float32), name="w1") # W2
        >>> # Rerunning the program will get different results:
        >>> np_1 = np.random.normal(0, 1, [1]).astype(np.float32) # A3
        >>> np_1 = np.random.normal(0, 1, [1]).astype(np.float32) # A4
        >>> w1 = Parameter(initializer("uniform", [2, 2], ms.float32), name="w1") # W3
        >>> w1 = Parameter(initializer("uniform", [2, 2], ms.float32), name="w1") # W4
        >>>
        >>> # 2. If global seed is set, numpy.random and initializer will use it:
        >>> set_seed(1234)
        >>> np_1 = np.random.normal(0, 1, [1]).astype(np.float32) # A1
        >>> np_1 = np.random.normal(0, 1, [1]).astype(np.float32) # A2
        >>> w1 = Parameter(initializer("uniform", [2, 2], ms.float32), name="w1") # W1
        >>> w1 = Parameter(initializer("uniform", [2, 2], ms.float32), name="w1") # W2
        >>> # Rerun the program will get the same results:
        >>> set_seed(1234)
        >>> np_1 = np.random.normal(0, 1, [1]).astype(np.float32) # A1
        >>> np_1 = np.random.normal(0, 1, [1]).astype(np.float32) # A2
        >>> w1 = Parameter(initializer("uniform", [2, 2], ms.float32), name="w1") # W1
        >>> w1 = Parameter(initializer("uniform", [2, 2], ms.float32), name="w1") # W2
        >>>
        >>> # 3. With neither global seed nor op seed set, ops.uniform passes (0, 0) to the kernel,
        >>> # which chooses a random seed rather than using zero as a reproducible seed:
        >>> c1 = ops.uniform((1, 4), minval, maxval) # C1
        >>> c2 = ops.uniform((1, 4), minval, maxval) # C2
        >>> # Rerun the program will get different results:
        >>> c1 = ops.uniform((1, 4), minval, maxval) # C3
        >>> c2 = ops.uniform((1, 4), minval, maxval) # C4
        >>>
        >>> # 4. If global seed is set, but op seed is not set, mindspore.ops.function.random_func
        >>> # will calculate a seed according to global seed and default op seed.
        >>> # Each call will change the default op seed, thus each call gets different
        >>> # results.
        >>> set_seed(1234)
        >>> c1 = ops.uniform((1, 4), minval, maxval) # C1
        >>> c2 = ops.uniform((1, 4), minval, maxval) # C2
        >>> # Rerun the program will get the same results:
        >>> set_seed(1234)
        >>> c1 = ops.uniform((1, 4), minval, maxval) # C1
        >>> c2 = ops.uniform((1, 4), minval, maxval) # C2
        >>>
        >>> # 5. If both global seed and op seed are set, mindspore.ops.function.random_func
        >>> # will calculate a seed according to global seed and op seed counter.
        >>> # Each call will change the op seed counter, thus each call gets different
        >>> # results.
        >>> set_seed(1234)
        >>> c1 = ops.uniform((1, 4), minval, maxval, seed=2) # C1
        >>> c2 = ops.uniform((1, 4), minval, maxval, seed=2) # C2
        >>> # Rerun the program will get the same results:
        >>> set_seed(1234)
        >>> c1 = ops.uniform((1, 4), minval, maxval, seed=2) # C1
        >>> c2 = ops.uniform((1, 4), minval, maxval, seed=2) # C2
        >>>
        >>> # 6. In a fresh process, with a positive op seed but no global seed, the global component
        >>> # is 0 and the op seed counter advances. This differs from explicitly calling set_seed(0).
        >>> # On a reproducible backend, the same call sequence in another fresh process is repeatable.
        >>> c1 = ops.uniform((1, 4), minval, maxval, seed=2) # C1
        >>> c2 = ops.uniform((1, 4), minval, maxval, seed=2) # C2
        >>> # Further calls in this process advance the sequence:
        >>> c1 = ops.uniform((1, 4), minval, maxval, seed=2) # C3
        >>> c2 = ops.uniform((1, 4), minval, maxval, seed=2) # C4
        >>>
        >>> # 7. Recall set_seed() in the program will reset numpy seed and op seed counter of
        >>> # mindspore.ops.function.random_func.
        >>> set_seed(1234)
        >>> np_1 = np.random.normal(0, 1, [1]).astype(np.float32) # A1
        >>> c1 = ops.uniform((1, 4), minval, maxval, seed=2) # C1
        >>> set_seed(1234)
        >>> np_2 = np.random.normal(0, 1, [1]).astype(np.float32) # still get A1
        >>> c2 = ops.uniform((1, 4), minval, maxval, seed=2) # still get C1
    """
    if not isinstance(seed, int):
        raise TypeError("The argument 'seed' must be type of int, but got {}.".format(type(seed)))
    Validator.check_non_negative_int(seed, "seed", "global_seed")
    import mindspore.dataset as de
    np.random.seed(seed)
    de.config.set_seed(seed)
    _reset_op_seed()
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed


def get_seed():
    """
    Get global seed.

    Returns:
        Integer. The global seed.

    Examples:
        >>> import mindspore as ms
        >>> ms.set_seed(1234)
        >>> seed = ms.get_seed()
        >>> print(seed)
        1234
    """
    return _GLOBAL_SEED


def _truncate_seed(seed):
    """
    Truncate the seed with MAXINT32.

    Args:
        seed (int): The seed to be truncated.

    Returns:
        Integer. The seed with MAXINT32.
    """
    return seed % _MAXINT32


def _update_seeds(op_seed, kernel_name):
    """
    Update the seed every time when the op seed is called.

    Args:
        op_seed (int): The op seed to be updated.
        kernel_name (str): The random op kernel.
    """
    if op_seed is not None:
        _KERNEL_SEED[(kernel_name, op_seed)] = _KERNEL_SEED.get((kernel_name, op_seed)) + \
                                               (keyConstant[0] ^ keyConstant[2])


def _get_op_seed(op_seed, kernel_name):
    """
    Get op seed which is relating to the specific kernel.
    If the seed does not exist, add it into the kernel's dictionary.

    Args:
        op_seed (int): The op seed to be updated.
        kernel_name (str): The random op kernel.
    """
    if (kernel_name, op_seed) not in _KERNEL_SEED:
        _KERNEL_SEED[(kernel_name, op_seed)] = op_seed
    return _KERNEL_SEED[(kernel_name, op_seed)]


def _get_graph_seed(op_seed, kernel_name):
    """
    Get the global and operator-level seed pair for a random operator.
    An unset global seed (None) becomes 0, whereas an explicitly set global seed of 0 becomes DEFAULT_GRAPH_SEED.
    These are mutually exclusive branches. An unset operator seed (None) becomes 0.
    If both normalized components are 0, return (0, 0) to let the kernel choose a random seed.
    Otherwise, truncate the global component and the current operator seed counter, then advance the counter.

    Note:
        For each seed, either op-seed or graph-seed, a random sequence will be generated relating to this seed.
        So, the state of the seed regarding to this op should be recorded.
        A simple illustration should be:
          If a random op is called twice within one program, the two results should be different:
          minval = Tensor(1.0, mstype.float32)
          maxval = Tensor(2.0, mstype.float32)
          print(C.uniform((1, 4), minval, maxval, seed=1))  # generates 'A1'
          print(C.uniform((1, 4), minval, maxval, seed=1))  # generates 'A2'
          If the same program runs again, it repeat the results:
          print(C.uniform((1, 4), minval, maxval, seed=1))  # generates 'A1'
          print(C.uniform((1, 4), minval, maxval, seed=1))  # generates 'A2'

    Returns:
        Tuple[int, int]. The global seed component and the operator seed component.

    Examples:
        >>> print(_get_graph_seed(0, 'normal'))
        (0, 0)
    """
    global_seed = get_seed()
    if global_seed == 0:
        global_seed = DEFAULT_GRAPH_SEED
    elif global_seed is None:
        global_seed = 0
    if op_seed is None:
        op_seed = 0
    # neither global seed or op seed is set, return (0, 0) to let kernel choose random seed.
    if global_seed == 0 and op_seed == 0:
        seeds = 0, 0
    else:
        Validator.check_non_negative_int(op_seed, "seed", kernel_name)
        temp_seed = _get_op_seed(op_seed, kernel_name)
        seeds = _truncate_seed(global_seed), _truncate_seed(temp_seed)
        _update_seeds(op_seed, kernel_name)
    return seeds
