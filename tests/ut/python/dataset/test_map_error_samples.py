# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
Test Map operation's handling of rows with errors
"""

import mindspore.dataset as ds

# Set global variable
TOTAL_SIZE = 100


def my_generator(ds_size):
    def generator_func():
        yield from range(ds_size)

    return generator_func


def raise_none(x):
    return x


def raise_all(x):
    raise ZeroDivisionError


def raise_first(x):
    if x == 0:
        raise ZeroDivisionError
    return x


def raise_first_10(x):
    if x < 10:
        raise ZeroDivisionError
    return x


def raise_first_100(x):
    if x < 100:
        raise ZeroDivisionError
    return x


def raise_first_101(x):
    if x < 101:
        raise ZeroDivisionError
    return x


def raise_first_n(x):
    if x < TOTAL_SIZE // 2 - 2:
        raise ZeroDivisionError
    return x


def raise_first_m(x):
    if x < TOTAL_SIZE // 2 + 2:
        raise ZeroDivisionError
    return x


def raise_all_but_last(x):
    if x < TOTAL_SIZE - 1:
        raise ZeroDivisionError
    return x


def raise_all_but_first(x):
    if x > 0:
        raise ZeroDivisionError
    return x


def raise_last_n(x):
    if x > TOTAL_SIZE // 2 + 2:
        raise ZeroDivisionError
    return x


def raise_last_m(x):
    if x > TOTAL_SIZE // 2 - 2:
        raise ZeroDivisionError
    return x


def raise_all_odds(x):
    if x % 2 != 0:
        raise ZeroDivisionError
    return x


def raise_all_3_remainders(x):
    if x % 3 != 0:
        raise ZeroDivisionError
    return x


def run_replace_test(transforms, dataset_size, num_parallel_workers, python_multiprocessing, expected=None, epochs=1):
    """ Function to run test replace error samples mode based on input configuration. """
    data1 = ds.GeneratorDataset(my_generator(dataset_size), ["data"])
    data1 = data1.map(operations=transforms,
                      num_parallel_workers=num_parallel_workers,
                      python_multiprocessing=python_multiprocessing)

    global TOTAL_SIZE
    TOTAL_SIZE = dataset_size

    itr = data1.create_dict_iterator(num_epochs=epochs, output_numpy=True)
    count = 0
    result = []
    for _ in range(epochs):
        for _, data in enumerate(itr):
            count += 1
            if expected is not None:
                result.append(data["data"].item(0))
    assert count == dataset_size * epochs
    if expected is not None:
        assert result == expected


def run_skip_test(transforms, dataset_size, num_parallel_workers, python_multiprocessing, expected=None, epochs=1):
    """ Function to run test skip error samples mode based on input configuration. """
    data1 = ds.GeneratorDataset(my_generator(dataset_size), ["data"])
    data1 = data1.map(operations=transforms,
                      num_parallel_workers=num_parallel_workers,
                      python_multiprocessing=python_multiprocessing)

    global TOTAL_SIZE
    TOTAL_SIZE = dataset_size

    itr = data1.create_dict_iterator(num_epochs=epochs, output_numpy=True)
    count = 0
    result = []
    for _ in range(epochs):
        for _, data in enumerate(itr):
            count += 1
            if expected is not None:
                result.append(data["data"].item(0))

    if expected is not None:
        assert count == len(expected) * epochs
        assert result == expected
