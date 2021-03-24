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
from mindspore._checkparam import Validator

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
    for (kernel_name, op_seed) in _KERNEL_SEED:
        _KERNEL_SEED[(kernel_name, op_seed)] = op_seed


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

    Examples:
        >>> from mindspore.ops import composite as C
        >>> from mindspore import Tensor
        >>>
        >>> # Note: (1) Please make sure the code is running in PYNATIVE MODE;
        >>> # (2) Because Composite-level ops need parameters to be Tensors, for below examples,
        >>> # when using C.uniform operator, minval and maxval are initialised as:
        >>> minval = Tensor(1.0, ms.float32)
        >>> maxval = Tensor(2.0, ms.float32)
        >>>
        >>> # 1. If global seed is not set, numpy.random and initializer will choose a random seed:
        >>> np_1 = np.random.normal(0, 1, [1]).astype(np.float32) # A1
        >>> np_1 = np.random.normal(0, 1, [1]).astype(np.float32) # A2
        >>> w1 = Parameter(initializer("uniform", [2, 2], ms.float32), name="w1") # W1
        >>> w1 = Parameter(initializer("uniform", [2, 2], ms.float32), name="w1") # W2
        >>> # Rerun the program will get different results:
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
        >>> # 3. If neither global seed nor op seed is set, mindspore.ops.composite.random_ops and
        >>> # mindspore.nn.probability.distribution will choose a random seed:
        >>> c1 = C.uniform((1, 4), minval, maxval) # C1
        >>> c2 = C.uniform((1, 4), minval, maxval) # C2
        >>> # Rerun the program will get different results:
        >>> c1 = C.uniform((1, 4), minval, maxval) # C3
        >>> c2 = C.uniform((1, 4), minval, maxval) # C4
        >>>
        >>> # 4. If global seed is set, but op seed is not set, mindspore.ops.composite.random_ops and
        >>> # mindspore.nn.probability.distribution will calculate a seed according to global seed and
        >>> # default op seed. Each call will change the default op seed, thus each call get different
        >>> # results.
        >>> set_seed(1234)
        >>> c1 = C.uniform((1, 4), minval, maxval) # C1
        >>> c2 = C.uniform((1, 4), minval, maxval) # C2
        >>> # Rerun the program will get the same results:
        >>> set_seed(1234)
        >>> c1 = C.uniform((1, 4), minval, maxval) # C1
        >>> c2 = C.uniform((1, 4), minval, maxval) # C2
        >>>
        >>> # 5. If both global seed and op seed are set, mindspore.ops.composite.random_ops and
        >>> # mindspore.nn.probability.distribution will calculate a seed according to global seed and
        >>> # op seed counter. Each call will change the op seed counter, thus each call get different
        >>> # results.
        >>> set_seed(1234)
        >>> c1 = C.uniform((1, 4), minval, maxval, seed=2) # C1
        >>> c2 = C.uniform((1, 4), minval, maxval, seed=2) # C2
        >>> # Rerun the program will get the same results:
        >>> set_seed(1234)
        >>> c1 = C.uniform((1, 4), minval, maxval, seed=2) # C1
        >>> c2 = C.uniform((1, 4), minval, maxval, seed=2) # C2
        >>>
        >>> # 6. If op seed is set but global seed is not set, 0 will be used as global seed. Then
        >>> # mindspore.ops.composite.random_ops and mindspore.nn.probability.distribution act as in
        >>> # condition 5.
        >>> c1 = C.uniform((1, 4), minval, maxval, seed=2) # C1
        >>> c2 = C.uniform((1, 4), minval, maxval, seed=2) # C2
        >>> # Rerun the program will get the same results:
        >>> c1 = C.uniform((1, 4), minval, maxval, seed=2) # C1
        >>> c2 = C.uniform((1, 4), minval, maxval, seed=2) # C2
        >>>
        >>> # 7. Recall set_seed() in the program will reset numpy seed and op seed counter of
        >>> # mindspore.ops.composite.random_ops and mindspore.nn.probability.distribution.
        >>> set_seed(1234)
        >>> np_1 = np.random.normal(0, 1, [1]).astype(np.float32) # A1
        >>> c1 = C.uniform((1, 4), minval, maxval, seed=2) # C1
        >>> set_seed(1234)
        >>> np_2 = np.random.normal(0, 1, [1]).astype(np.float32) # still get A1
        >>> c2 = C.uniform((1, 4), minval, maxval, seed=2) # still get C1
    """
    if not isinstance(seed, int):
        raise TypeError("The seed must be type of int.")
    Validator.check_non_negative_int(seed, "seed", "global_seed")
    # We put import here to solve an ut/cpp core issue
    import mindspore.dataset as de
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
    global _KERNEL_SEED
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


def _get_global_and_op_seed():
    """Get global_seed and op_seed."""
    global_seed = get_seed()
    op_seed = get_seed()
    if global_seed == 0:
        global_seed = DEFAULT_GRAPH_SEED
    elif global_seed is None:
        global_seed = 0
    if op_seed is None:
        op_seed = 0
    Validator.check_non_negative_int(op_seed, "seed", "init")
    temp_seed = _get_op_seed(op_seed, "init")
    seeds = _truncate_seed(global_seed), _truncate_seed(temp_seed)
    return seeds


def _get_graph_seed(op_seed, kernel_name):
    """
    Get the graph-level seed.
    Graph-level seed is used as a global variable, that can be used in different ops in case op-level seed is not set.
    If op-level seed is 0, use graph-level seed; if graph-level seed is also 0, the system would generate a
    random seed.

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
        Integer. The current graph-level seed.

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
