# Adapted from https://github.com/numpy/numpy/blob/main/numpy/random/bit_generator.pyx

# BitGenerator base class and SeedSequence used to seed the BitGenerators.
#
# SeedSequence is derived from Melissa E. O'Neill's C++11 `std::seed_seq`
# implementation, as it has a lot of nice properties that we want.
#
# https://gist.github.com/imneme/540829265469e673d045
# https://www.pcg-random.org/posts/developing-a-seed_seq-alternative.html
#
# The MIT License (MIT)
#
# Copyright (c) 2015 Melissa E. O'Neill
# Copyright (c) 2019 NumPy Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Bit generator."""

import numpy as np


DEFAULT_POOL_SIZE = 4
INIT_A = 0x43B0D7E5
MULT_A = 0x931E8875
INIT_B = 0x8B51F9DD
MULT_B = 0x58F38DED
MIX_MULT_L = 0xCA01F9DD
MIX_MULT_R = 0x4973F715
XSHIFT = np.dtype(np.uint32).itemsize * 8 // 2
MASK32 = 0xFFFFFFFF


def hashmix(value, hash_const):
    """
    Mix the given value and hash constant.

    Args:
        value (int): The value to mix.
        hash_const (int): The hash constant to mix.

    Returns:
        int, the mixed value.
    """
    value = (value ^ hash_const) & MASK32
    hash_const = (hash_const * MULT_A) & MASK32
    value = (value * hash_const) & MASK32
    value = (value ^ (value >> XSHIFT)) & MASK32
    return value, hash_const


def mix(x, y):
    """
    Mix the given values.

    Args:
        x (int): The first value to mix.
        y (int): The second value to mix.

    Returns:
        int, the mixed value.
    """
    result = (((MIX_MULT_L * x) & MASK32) - ((MIX_MULT_R * y) & MASK32)) & MASK32
    result = (result ^ (result >> XSHIFT)) & MASK32
    return result


def mix_entropy(entropy_array):
    """
    Mix in the given entropy to mixer.

    Args:
        entropy_array (list): The entropy array to mix in.

    Returns:
        np.ndarray, the mixed entropy.
    """
    hash_const_a = INIT_A
    mixer = [0] * DEFAULT_POOL_SIZE

    # Add in the entropy up to the pool size.
    for i, _ in enumerate(mixer):
        mixer[i], hash_const_a = hashmix(entropy_array[i], hash_const_a)

    # Mix all bits together so late bits can affect earlier bits.
    for i_src, _ in enumerate(mixer):
        for i_dst, _ in enumerate(mixer):
            if i_src != i_dst:
                value, hash_const_a = hashmix(mixer[i_src], hash_const_a)
                mixer[i_dst] = mix(mixer[i_dst], value)

    return mixer


def generate_state(pool):
    """
    Return the requested number of words for PRNG seeding.

    Args:
        pool (np.ndarray): The pool of words to mix.

    Returns:
        np.ndarray, the words for seeding.
    """
    hash_const_b = INIT_B

    state = [0] * DEFAULT_POOL_SIZE
    for i_dst, data_val in enumerate(pool):
        data_val = (data_val ^ hash_const_b) & MASK32
        hash_const_b = (hash_const_b * MULT_B) & MASK32
        data_val = (data_val * hash_const_b) & MASK32
        data_val = (data_val ^ (data_val >> XSHIFT)) & MASK32
        state[i_dst] = data_val
    return state


def seed_sequence(base_seed, worker_id):
    """
    This function generates an array of uint32 as the seed for
    `numpy.random`, in order to prevent state collision due to same
    seed and algorithm for `numpy.random` and `random` modules.
    """
    entropy_array = [worker_id, base_seed & MASK32, base_seed >> 32, 0]
    pool = mix_entropy(entropy_array)
    return generate_state(pool)
