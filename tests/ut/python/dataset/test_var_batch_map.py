# Copyright 2019-2023 Huawei Technologies Co., Ltd
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
""" Test batch variations including per_batch_map """
import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore import log as logger


def test_batch_corner_cases():
    """
    Feature: Batch op
    Description: Test batch variations using corner cases:
        - where batch_size is greater than the entire epoch, with drop equals to both val
        - where Batch op is done before Repeat op with different drop
    Expectation: Output is equal to the expected output
    """
    def gen(num):
        for i in range(num):
            yield (np.array([i]),)

    def test_repeat_batch(gen_num, repeats, batch_size, drop, res):
        data1 = ds.GeneratorDataset((lambda: gen(gen_num)), ["num"]).repeat(repeats).batch(batch_size, drop)
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            res.append(item["num"])

    def test_batch_repeat(gen_num, repeats, batch_size, drop, res):
        data1 = ds.GeneratorDataset((lambda: gen(gen_num)), ["num"]).batch(batch_size, drop).repeat(repeats)
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            res.append(item["num"])

    tst1, tst2, tst3, tst4 = [], [], [], []
    # case 1 & 2, where batch_size is greater than the entire epoch, with drop equals to both val
    test_repeat_batch(gen_num=2, repeats=4, batch_size=7, drop=False, res=tst1)
    np.testing.assert_array_equal(np.array([[0], [1], [0], [1], [0], [1], [0]]), tst1[0], "\nATTENTION BATCH FAILED\n")
    np.testing.assert_array_equal(np.array([[1]]), tst1[1], "\nATTENTION TEST BATCH FAILED\n")
    assert len(tst1) == 2, "\nATTENTION TEST BATCH FAILED\n"
    test_repeat_batch(gen_num=2, repeats=4, batch_size=5, drop=True, res=tst2)
    np.testing.assert_array_equal(np.array([[0], [1], [0], [1], [0]]), tst2[0], "\nATTENTION BATCH FAILED\n")
    assert len(tst2) == 1, "\nATTENTION TEST BATCH FAILED\n"
    # case 3 & 4, batch before repeat with different drop
    test_batch_repeat(gen_num=5, repeats=2, batch_size=4, drop=True, res=tst3)
    np.testing.assert_array_equal(np.array([[0], [1], [2], [3]]), tst3[0], "\nATTENTION BATCH FAILED\n")
    np.testing.assert_array_equal(tst3[0], tst3[1], "\nATTENTION BATCH FAILED\n")
    assert len(tst3) == 2, "\nATTENTION BATCH FAILED\n"
    test_batch_repeat(gen_num=5, repeats=2, batch_size=4, drop=False, res=tst4)
    np.testing.assert_array_equal(np.array([[0], [1], [2], [3]]), tst4[0], "\nATTENTION BATCH FAILED\n")
    np.testing.assert_array_equal(tst4[0], tst4[2], "\nATTENTION BATCH FAILED\n")
    np.testing.assert_array_equal(tst4[1], np.array([[4]]), "\nATTENTION BATCH FAILED\n")
    np.testing.assert_array_equal(tst4[1], tst4[3], "\nATTENTION BATCH FAILED\n")
    assert len(tst4) == 4, "\nATTENTION BATCH FAILED\n"


# Run this test in separate process since this test updates shared memory config
@pytest.mark.forked
def test_variable_size_batch():
    """
    Feature: Batch
    Description: Test batch variations with repeat and with/without per_batch_map.
        Each sub-test is tested with same parameters except that
        - the second test uses per_batch_map which passes each row a pyfunc and makes a deep copy of the row
        - the third test (if it exists) uses per_batch_map and python multiprocessing
    Expectation: Results are the same, independent of per_batch_map or python_multiprocessing settings
    """

    def check_res(arr1, arr2):
        for ind, _ in enumerate(arr1):
            if not np.array_equal(arr1[ind], np.array(arr2[ind])):
                return False
        return len(arr1) == len(arr2)

    def gen(num):
        for i in range(num):
            yield (np.array([i]),)

    def add_one_by_batch_num(batch_info):
        return batch_info.get_batch_num() + 1

    def add_one_by_epoch(batch_info):
        return batch_info.get_epoch_num() + 1

    def simple_copy(col_list, batch_info):
        _ = batch_info
        return ([np.copy(arr) for arr in col_list],)

    def test_repeat_batch(gen_num, r, drop, func, res):
        data1 = ds.GeneratorDataset((lambda: gen(gen_num)), ["num"]).repeat(r).batch(batch_size=func,
                                                                                     drop_remainder=drop)
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            res.append(item["num"])

    # same as test_repeat_batch except each row is passed through via a map which makes a copy of each element
    def test_repeat_batch_with_copy_map(gen_num, r, drop, func):
        res = []
        data1 = ds.GeneratorDataset((lambda: gen(gen_num)), ["num"]).repeat(r) \
            .batch(batch_size=func, drop_remainder=drop, input_columns=["num"], per_batch_map=simple_copy)
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            res.append(item["num"])
        return res

    def test_batch_repeat(gen_num, r, drop, func, res):
        data1 = ds.GeneratorDataset((lambda: gen(gen_num)), ["num"]).batch(batch_size=func, drop_remainder=drop).repeat(
            r)
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            res.append(item["num"])

    # same as test_batch_repeat except each row is passed through via a map which makes a copy of each element
    def test_batch_repeat_with_copy_map(gen_num, r, drop, func):
        res = []
        data1 = ds.GeneratorDataset((lambda: gen(gen_num)), ["num"]) \
            .batch(batch_size=func, drop_remainder=drop, input_columns=["num"], per_batch_map=simple_copy).repeat(r)
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            res.append(item["num"])
        return res

    # same as test_batch_repeat_with_copy_map except with python multiprocessing enabled
    def test_batch_repeat_with_copy_map_multiproc(gen_num, r, drop, func, num_workers, my_maxrowsize):
        # Reduce memory required by disabling the shared memory optimization
        mem_original = ds.config.get_enable_shared_mem()
        ds.config.set_enable_shared_mem(False)

        res = []
        data1 = ds.GeneratorDataset((lambda: gen(gen_num)), ["num"], num_parallel_workers=num_workers,
                                    python_multiprocessing=True, max_rowsize=my_maxrowsize) \
            .batch(batch_size=func, drop_remainder=drop, input_columns=["num"], per_batch_map=simple_copy,
                   num_parallel_workers=num_workers, python_multiprocessing=True,
                   max_rowsize=my_maxrowsize).repeat(r)
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            res.append(item["num"])

        ds.config.set_enable_shared_mem(mem_original)
        return res

    tst1, tst2, tst3, tst4, tst5, tst6, tst7 = [], [], [], [], [], [], []

    # no repeat, simple var size, based on batch_num
    test_repeat_batch(7, 1, True, add_one_by_batch_num, tst1)
    assert check_res(tst1, [[[0]], [[1], [2]], [[3], [4], [5]]]), "\nATTENTION VAR BATCH FAILED\n"
    assert check_res(tst1, test_repeat_batch_with_copy_map(7, 1, True, add_one_by_batch_num)), "\nMAP FAILED\n"
    test_repeat_batch(9, 1, False, add_one_by_batch_num, tst2)
    assert check_res(tst2, [[[0]], [[1], [2]], [[3], [4], [5]], [[6], [7], [8]]]), "\nATTENTION VAR BATCH FAILED\n"
    assert check_res(tst2, test_repeat_batch_with_copy_map(9, 1, False, add_one_by_batch_num)), "\nMAP FAILED\n"
    # batch after repeat, cross epoch batch
    test_repeat_batch(7, 2, False, add_one_by_batch_num, tst3)
    assert check_res(tst3, [[[0]], [[1], [2]], [[3], [4], [5]], [[6], [0], [1], [2]],
                            [[3], [4], [5], [6]]]), "\nATTENTION VAR BATCH FAILED\n"
    assert check_res(tst3, test_repeat_batch_with_copy_map(7, 2, False, add_one_by_batch_num)), "\nMAP FAILED\n"
    # repeat after batch, no cross epoch batch, remainder dropped
    test_batch_repeat(9, 7, True, add_one_by_batch_num, tst4)
    assert check_res(tst4, [[[0]], [[1], [2]], [[3], [4], [5]]] * 7), "\nATTENTION VAR BATCH FAILED\n"
    assert check_res(tst4, test_batch_repeat_with_copy_map(9, 7, True, add_one_by_batch_num)), "\nAMAP FAILED\n"
    # repeat after batch, no cross epoch batch, remainder kept
    test_batch_repeat(9, 3, False, add_one_by_batch_num, tst5)
    assert check_res(tst5, [[[0]], [[1], [2]], [[3], [4], [5]], [[6], [7], [8]]] * 3), "\nATTENTION VAR BATCH FAILED\n"
    assert check_res(tst5, test_batch_repeat_with_copy_map(9, 3, False, add_one_by_batch_num)), "\nMAP FAILED\n"
    # batch_size based on epoch number, drop
    test_batch_repeat(4, 4, True, add_one_by_epoch, tst6)
    assert check_res(tst6, [[[0]], [[1]], [[2]], [[3]], [[0], [1]], [[2], [3]], [[0], [1], [2]],
                            [[0], [1], [2], [3]]]), "\nATTENTION VAR BATCH FAILED\n"
    assert check_res(tst6, test_batch_repeat_with_copy_map(4, 4, True, add_one_by_epoch)), "\nMAP FAILED\n"
    # batch_size based on epoch number, no drop
    test_batch_repeat(4, 4, False, add_one_by_epoch, tst7)
    assert check_res(tst7, [[[0]], [[1]], [[2]], [[3]], [[0], [1]], [[2], [3]], [[0], [1], [2]], [[3]],
                            [[0], [1], [2], [3]]]), "\nATTENTION VAR BATCH FAILED\n" + str(tst7)
    assert check_res(tst7, test_batch_repeat_with_copy_map(4, 4, False, add_one_by_epoch)), "\nMAP FAILED\n"
    assert check_res(tst7, test_batch_repeat_with_copy_map_multiproc(
        4, 4, False, add_one_by_epoch, 4, 1)), "\nMULTIPROC1 MAP FAILED\n"
    assert check_res(tst7, test_batch_repeat_with_copy_map_multiproc(
        4, 4, False, add_one_by_epoch, 2, 2)), "\nMULTIPROC2 MAP FAILED\n"


def test_get_batchsize_on_callable_batchsize():
    """
    Feature: Test get_batch_size() func which can get batch size of pipeline.
    Description: Use get_batch_size() func on dataset pipeline with dynamic batch size.
    Expectation: Return -1 which means the batch size is unknown.
    """
    def gen(num):
        for i in range(num):
            yield (np.array([i]),)

    def add_one_by_batch_num(batch_info):
        return batch_info.get_batch_num() + 1

    data1 = ds.GeneratorDataset((lambda: gen(10)), ["num"])
    data1 = data1.batch(batch_size=add_one_by_batch_num, drop_remainder=True)

    batchsize = data1.get_batch_size()
    assert batchsize == -1


def test_basic_batch_map():
    """
    Feature: Batch op
    Description: Test basic map Batch op with per_batch_map
    Expectation: Output is equal to the expected output
    """
    def check_res(arr1, arr2):
        for ind, _ in enumerate(arr1):
            if not np.array_equal(arr1[ind], np.array(arr2[ind])):
                return False
        return len(arr1) == len(arr2)

    def gen(num):
        for i in range(num):
            yield (np.array([i]),)

    def invert_sign_per_epoch(col_list, batch_info):
        return ([np.copy(((-1) ** batch_info.get_epoch_num()) * arr) for arr in col_list],)

    def invert_sign_per_batch(col_list, batch_info):
        return ([np.copy(((-1) ** batch_info.get_batch_num()) * arr) for arr in col_list],)

    def batch_map_config(num, r, batch_size, func, res):
        data1 = ds.GeneratorDataset((lambda: gen(num)), ["num"]) \
            .batch(batch_size=batch_size, input_columns=["num"], per_batch_map=func).repeat(r)
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            res.append(item["num"])

    tst1, tst2, = [], []
    batch_map_config(4, 2, 2, invert_sign_per_epoch, tst1)
    assert check_res(tst1, [[[0], [1]], [[2], [3]], [[0], [-1]], [[-2], [-3]]]), "\nATTENTION MAP BATCH FAILED\n" + str(
        tst1)
    # each batch, the sign of a row is changed, test map is corrected performed according to its batch_num
    batch_map_config(4, 2, 2, invert_sign_per_batch, tst2)
    assert check_res(tst2,
                     [[[0], [1]], [[-2], [-3]], [[0], [1]], [[-2], [-3]]]), "\nATTENTION MAP BATCH FAILED\n" + str(tst2)


def test_batch_multi_col_map():
    """
    Feature: Batch op
    Description: Test map Batch op with multiple columns input with per_batch_map
    Expectation: Output is equal to the expected output
    """
    def check_res(arr1, arr2):
        for ind, _ in enumerate(arr1):
            if not np.array_equal(arr1[ind], np.array(arr2[ind])):
                return False
        return len(arr1) == len(arr2)

    def gen(num):
        for i in range(num):
            yield (np.array([i]), np.array([i ** 2]))

    def col1_col2_add_num(col1, col2, batch_info):
        _ = batch_info
        return ([[np.copy(arr + 100) for arr in col1],
                 [np.copy(arr + 300) for arr in col2]])

    def invert_sign_per_batch(col_list, batch_info):
        return ([np.copy(((-1) ** batch_info.get_batch_num()) * arr) for arr in col_list],)

    def invert_sign_per_batch_multi_col(col1, col2, batch_info):
        return ([np.copy(((-1) ** batch_info.get_batch_num()) * arr) for arr in col1],
                [np.copy(((-1) ** batch_info.get_batch_num()) * arr) for arr in col2])

    def batch_map_config(num, r, batch_size, func, col_names, res):
        data1 = ds.GeneratorDataset((lambda: gen(num)), ["num", "num_square"]) \
            .batch(batch_size=batch_size, input_columns=col_names, per_batch_map=func).repeat(r)
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            res.append(np.array([item["num"], item["num_square"]]))

    tst1, tst2, tst3, tst4 = [], [], [], []
    batch_map_config(4, 2, 2, invert_sign_per_batch, ["num_square"], tst1)
    assert check_res(tst1, [[[[0], [1]], [[0], [1]]], [[[2], [3]], [[-4], [-9]]], [[[0], [1]], [[0], [1]]],
                            [[[2], [3]], [[-4], [-9]]]]), "\nATTENTION MAP BATCH FAILED\n" + str(tst1)

    batch_map_config(4, 2, 2, invert_sign_per_batch_multi_col, ["num", "num_square"], tst2)
    assert check_res(tst2, [[[[0], [1]], [[0], [1]]], [[[-2], [-3]], [[-4], [-9]]], [[[0], [1]], [[0], [1]]],
                            [[[-2], [-3]], [[-4], [-9]]]]), "\nATTENTION MAP BATCH FAILED\n" + str(tst2)

    # the two tests below verify the order of the map.
    # num_square column adds 100, num column adds 300.
    batch_map_config(4, 3, 2, col1_col2_add_num, ["num_square", "num"], tst3)
    assert check_res(tst3, [[[[300], [301]], [[100], [101]]],
                            [[[302], [303]], [[104], [109]]]] * 3), "\nATTENTION MAP BATCH FAILED\n" + str(tst3)
    # num column adds 100, num_square column adds 300.
    batch_map_config(4, 3, 2, col1_col2_add_num, ["num", "num_square"], tst4)
    assert check_res(tst4, [[[[100], [101]], [[300], [301]]],
                            [[[102], [103]], [[304], [309]]]] * 3), "\nATTENTION MAP BATCH FAILED\n" + str(tst4)


def test_var_batch_multi_col_map():
    """
    Feature: Batch op
    Description: Test Batch op with a function arg for batch_size using multiple columns input with per_batch_map
    Expectation: Output is equal to the expected output
    """
    def check_res(arr1, arr2):
        for ind, _ in enumerate(arr1):
            if not np.array_equal(arr1[ind], np.array(arr2[ind])):
                return False
        return len(arr1) == len(arr2)

    # gen 3 columns
    # first column: 0, 3, 6, 9 ... ...
    # second column:1, 4, 7, 10 ... ...
    # third column: 2, 5, 8, 11 ... ...
    def gen_3_cols(num):
        for i in range(num):
            yield (np.array([i * 3]), np.array([i * 3 + 1]), np.array([i * 3 + 2]))

    # first epoch batch_size per batch: 1, 2 ,3 ... ...
    # second epoch batch_size per batch: 2, 4, 6 ... ...
    # third epoch batch_size per batch: 3, 6 ,9 ... ...
    def batch_func(batch_info):
        return (batch_info.get_batch_num() + 1) * (batch_info.get_epoch_num() + 1)

    # multiply first col by batch_num, multiply second col by -batch_num
    def map_func(col1, col2, batch_info):
        return ([np.copy((1 + batch_info.get_batch_num()) * arr) for arr in col1],
                [np.copy(-(1 + batch_info.get_batch_num()) * arr) for arr in col2])

    def batch_map_config(num, r, fbatch, fmap, col_names, res):
        data1 = ds.GeneratorDataset((lambda: gen_3_cols(num)), ["col1", "col2", "col3"]) \
            .batch(batch_size=fbatch, input_columns=col_names, per_batch_map=fmap).repeat(r)
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            res.append(np.array([item["col1"], item["col2"], item["col3"]]))

    tst1 = []
    tst1_res = [[[[0]], [[-1]], [[2]]], [[[6], [12]], [[-8], [-14]], [[5], [8]]],
                [[[27], [36], [45]], [[-30], [-39], [-48]], [[11], [14], [17]]],
                [[[72], [84], [96], [108]], [[-76], [-88], [-100], [-112]], [[20], [23], [26], [29]]]]
    batch_map_config(10, 1, batch_func, map_func, ["col1", "col2"], tst1)
    assert check_res(tst1, tst1_res), "test_var_batch_multi_col_map FAILED"


def test_var_batch_var_resize():
    """
    Feature: Batch op
    Description: Test Batch op with a function arg for batch_size with resize as per_batch_map
    Expectation: Output is equal to the expected output
    """
    # fake resize image according to its batch number, if it's 5-th batch, resize to (5^2, 5^2) = (25, 25)
    def np_psedo_resize(col, batch_info):
        s = (batch_info.get_batch_num() + 1) ** 2
        return ([np.copy(c[0:s, 0:s, :]) for c in col],)

    def add_one(batch_info):
        return batch_info.get_batch_num() + 1

    data1 = ds.ImageFolderDataset("../data/dataset/testPK/data/", num_parallel_workers=4, decode=True)
    data1 = data1.batch(batch_size=add_one, drop_remainder=True, input_columns=["image"], per_batch_map=np_psedo_resize)
    # i-th batch has shape [i, i^2, i^2, 3]
    i = 1
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert item["image"].shape == (i, i ** 2, i ** 2, 3), "\ntest_var_batch_var_resize FAILED\n"
        i += 1


def test_exception():
    """
    Feature: Batch op
    Description: Test Batch op with bad batch size and bad map function
    Expectation: Error is raised as expected
    """
    def gen(num):
        for i in range(num):
            yield (np.array([i]),)

    def bad_batch_size(batch_info):
        raise StopIteration
        # Note: return batch_info.get_batch_num()

    def bad_map_func(col, batch_info):
        raise StopIteration
        # Note: return (col,)

    data1 = ds.GeneratorDataset((lambda: gen(100)), ["num"]).batch(bad_batch_size)
    try:
        for _ in data1.create_dict_iterator(num_epochs=1):
            pass
        assert False
    except RuntimeError:
        pass

    data2 = ds.GeneratorDataset((lambda: gen(100)), ["num"]).batch(4, input_columns=["num"], per_batch_map=bad_map_func)
    try:
        for _ in data2.create_dict_iterator(num_epochs=1):
            pass
        assert False
    except RuntimeError:
        pass


def test_multi_col_map():
    """
    Feature: Batch op
    Description: Test Batch op with multiple columns with various per_batch_map args with valid and invalid inputs
    Expectation: Output is equal to the expected output for valid input and error is raised otherwise
    """
    def gen_2_cols(num):
        for i in range(1, 1 + num):
            yield (np.array([i]), np.array([i ** 2]))

    def split_col(col, batch_info):
        return ([np.copy(arr) for arr in col], [np.copy(-arr) for arr in col])

    def merge_col(col1, col2, batch_info):
        merged = []
        for k, v in enumerate(col1):
            merged.append(np.array(v + col2[k]))
        return (merged,)

    def swap_col(col1, col2, batch_info):
        return ([np.copy(a) for a in col2], [np.copy(b) for b in col1])

    def batch_map_config(num, s, f, in_nms, out_nms, col_order=None):
        try:
            dst = ds.GeneratorDataset((lambda: gen_2_cols(num)), ["col1", "col2"])
            dst = dst.batch(batch_size=s, input_columns=in_nms, output_columns=out_nms, per_batch_map=f)
            if col_order is not None:
                dst = dst.project(col_order)
            res = []
            for row in dst.create_dict_iterator(num_epochs=1, output_numpy=True):
                res.append(row)
            return res
        except (ValueError, RuntimeError, TypeError) as e:
            return str(e)

    # split 1 col into 2 cols
    res = batch_map_config(2, 2, split_col, ["col2"], ["col_x", "col_y"])[0]
    assert np.array_equal(res["col1"], [[1], [2]])
    assert np.array_equal(res["col_x"], [[1], [4]]) and np.array_equal(res["col_y"], [[-1], [-4]])

    # merge 2 cols into 1 col
    res = batch_map_config(4, 4, merge_col, ["col1", "col2"], ["merged"])[0]
    assert np.array_equal(res["merged"], [[2], [6], [12], [20]])

    # swap once
    res = batch_map_config(3, 3, swap_col, ["col1", "col2"], ["col1", "col2"])[0]
    assert np.array_equal(res["col1"], [[1], [4], [9]]) and np.array_equal(res["col2"], [[1], [2], [3]])

    # swap twice
    res = batch_map_config(3, 3, swap_col, ["col1", "col2"], ["col2", "col1"])[0]
    assert np.array_equal(res["col2"], [[1], [4], [9]]) and np.array_equal(res["col1"], [[1], [2], [3]])

    # test project after map
    res = batch_map_config(2, 2, split_col, ["col2"], ["col_x", "col_y"], ["col_x", "col_y", "col1"])[0]
    assert list(res.keys()) == ["col_x", "col_y", "col1"]

    # test the insertion order is maintained
    res = batch_map_config(2, 2, split_col, ["col2"], ["col_x", "col_y"], ["col1", "col_x", "col_y"])[0]
    assert list(res.keys()) == ["col1", "col_x", "col_y"]

    # test exceptions
    assert "output_columns with value 233 is not of type" in batch_map_config(2, 2, split_col, ["col2"], 233)
    assert "columns that are not involved in 'per_batch_map' should not be in output_columns" \
           in batch_map_config(2, 2, split_col, ["col2"], ["col1"])
    assert "the number of columns returned in 'per_batch_map' function should be 3" \
           in batch_map_config(2, 2, split_col, ["col2"], ["col3", "col4", "col5"])
    assert "'col-1' of 'input_columns' doesn't exist" \
           in batch_map_config(2, 2, split_col, ["col-1"], ["col_x", "col_y"])


def test_multi_col_concat_map():
    """
    Feature: Batch op
    Description: Test Batch op with multiple columns with concat per_batch_map args with valid inputs
    Expectation: Output is equal to the expected output for valid input
    """
    def gen_2_cols(num):
        for i in range(1, 1 + num):
            yield np.array([i]), np.array([i * 2]), np.array([i ** 2])

    def concat(args):
        arg_list = []
        for arg in args:
            rows = []
            for value in arg:
                rows.append(value)
            arg_list.append(np.array(np.concatenate(rows, axis=0)))
        return arg_list

    def append_batch_array(args):
        arg_list = []
        for arg in args:
            rows = []
            for value in arg:
                rows.append(value)
            arg_list.append(np.array(rows))
        return arg_list

    def append_batch_list(args):
        arg_list = []
        for arg in args:
            rows = []
            for value in arg:
                rows.append(value)
            arg_list.append(rows)
        return arg_list

    def concat_all_col(col1, col2, col3, batch_info):
        return tuple(concat([col1, col2, col3]))

    def concat_less_col(col1, col2, batch_info):
        return tuple(concat([col1, col2]))

    def concat_more_col(col1, col2, batch_info):
        return tuple(concat([col1, col2, col1]))

    def append_all_col_list(col1, col2, col3, batch_info):
        return tuple(append_batch_list([col1, col2, col3]))

    def append_less_col_list(col1, col2, batch_info):
        return tuple(append_batch_list([col1, col2]))

    def append_more_col_list(col1, col2, batch_info):
        return tuple(append_batch_list([col1, col2, col1]))

    def append_all_col_array(col1, col2, col3, batch_info):
        return tuple(append_batch_array([col1, col2, col3]))

    def append_less_col_array(col1, col2, batch_info):
        return tuple(append_batch_array([col1, col2]))

    def append_more_col_array(col1, col2, batch_info):
        return tuple(append_batch_array([col1, col2, col1]))

    def batch_map(num, all_col, batch_size, icol, ocol, pfunc, res):
        dst = ds.GeneratorDataset((lambda: gen_2_cols(num)), all_col)
        dst = dst.batch(batch_size=batch_size, input_columns=icol, output_columns=ocol,
                        per_batch_map=pfunc)
        for row in dst.create_dict_iterator(num_epochs=1, output_numpy=True):
            res.append(row)

    # test all column
    res = []
    batch_map(3, ["col1", "col2", "col3"], 3, ["col1", "col2", "col3"], ["col1", "col2", "col3"], concat_all_col, res)
    assert np.array_equal(res[0]["col1"], [1, 2, 3]) and np.array_equal(res[0]["col2"], [2, 4, 6]) and \
           np.array_equal(res[0]["col3"], [1, 4, 9])
    # test less column
    res = []
    batch_map(6, ["col1", "col2", "col3"], 3, ["col1", "col2"], ["col1", "col2"], concat_less_col, res)
    assert np.array_equal(res[1]["col1"], [4, 5, 6]) and np.array_equal(res[1]["col2"], [8, 10, 12]) and \
           np.array_equal(res[1]["col3"], [[16], [25], [36]])
    # test more column
    res = []
    batch_map(8, ["col1", "col2", "col3"], 3, ["col1", "col2"], ["col1", "col2", "col4"], concat_more_col, res)
    assert np.array_equal(res[0]["col1"], [1, 2, 3]) and np.array_equal(res[0]["col2"], [2, 4, 6]) and \
           np.array_equal(res[0]["col3"], [[1], [4], [9]]) and np.array_equal(res[0]["col4"], [1, 2, 3])
    assert np.array_equal(res[2]["col1"], [7, 8]) and np.array_equal(res[2]["col2"], [14, 16]) and \
           np.array_equal(res[2]["col3"], [[49], [64]]) and np.array_equal(res[2]["col4"], [7, 8])

    # test all column
    res = []
    batch_map(3, ["col1", "col2", "col3"], 3, ["col1", "col2", "col3"], ["col1", "col2", "col3"],
              append_all_col_array, res)
    assert np.array_equal(res[0]["col1"], [[1], [2], [3]]) and np.array_equal(res[0]["col2"], [[2], [4], [6]]) and \
           np.array_equal(res[0]["col3"], [[1], [4], [9]])
    list_res = []
    batch_map(3, ["col1", "col2", "col3"], 3, ["col1", "col2", "col3"], ["col1", "col2", "col3"],
              append_all_col_list, list_res)
    assert np.array_equal(res[0]["col1"], list_res[0]["col1"]) and \
           np.array_equal(res[0]["col2"], list_res[0]["col2"]) \
           and np.array_equal(res[0]["col3"], list_res[0]["col3"])

    # test less column
    res = []
    batch_map(6, ["col1", "col2", "col3"], 3, ["col1", "col2"], ["col1", "col2"], append_less_col_array, res)
    assert np.array_equal(res[1]["col1"], [[4], [5], [6]]) and np.array_equal(res[1]["col2"], [[8], [10], [12]]) and \
           np.array_equal(res[1]["col3"], [[16], [25], [36]])
    list_res = []
    batch_map(6, ["col1", "col2", "col3"], 3, ["col1", "col2"], ["col1", "col2"], append_less_col_list, list_res)
    assert np.array_equal(res[1]["col1"], list_res[1]["col1"]) and \
           np.array_equal(res[1]["col2"], list_res[1]["col2"]) \
           and np.array_equal(res[1]["col3"], list_res[1]["col3"])

    # test more column
    res = []
    batch_map(8, ["col1", "col2", "col3"], 3, ["col1", "col2"], ["col1", "col2", "col4"], append_more_col_array, res)
    assert np.array_equal(res[0]["col1"], [[1], [2], [3]]) and np.array_equal(res[0]["col2"], [[2], [4], [6]]) and \
           np.array_equal(res[0]["col3"], [[1], [4], [9]]) and np.array_equal(res[0]["col4"], [[1], [2], [3]])
    assert np.array_equal(res[2]["col1"], [[7], [8]]) and np.array_equal(res[2]["col2"], [[14], [16]]) and \
           np.array_equal(res[2]["col3"], [[49], [64]]) and np.array_equal(res[2]["col4"], [[7], [8]])
    list_res = []
    batch_map(8, ["col1", "col2", "col3"], 3, ["col1", "col2"], ["col1", "col2", "col4"], append_more_col_list,
              list_res)
    assert np.array_equal(res[0]["col1"], list_res[0]["col1"]) and \
           np.array_equal(res[0]["col2"], list_res[0]["col2"]) \
           and np.array_equal(res[0]["col3"], list_res[0]["col3"])
    assert np.array_equal(res[2]["col1"], list_res[2]["col1"]) and \
           np.array_equal(res[2]["col2"], list_res[2]["col2"]) \
           and np.array_equal(res[2]["col3"], list_res[2]["col3"])


def test_exceptions_2():
    """
    Feature: Batch op
    Description: Test Batch op with invalid column name and invalid per_batch_map function argument
    Expectation: Error is raised as expected
    """
    def gen(num):
        for i in range(num):
            yield (np.array([i]),)

    def simple_copy(col_list, batch_info):
        return ([np.copy(arr) for arr in col_list],)

    def concat_copy(col_list, batch_info):
        # this will duplicate the number of rows returned, which would be wrong!
        return ([np.copy(arr) for arr in col_list] * 2,)

    def shrink_copy(col_list, batch_info):
        # this will duplicate the number of rows returned, which would be wrong!
        return ([np.copy(arr) for arr in col_list][0:int(len(col_list) / 2)],)

    def test_exceptions_config(gen_num, batch_size, in_cols, per_batch_map):
        data1 = ds.GeneratorDataset((lambda: gen(gen_num)), ["num"]).batch(batch_size, input_columns=in_cols,
                                                                           per_batch_map=per_batch_map)
        try:
            for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
                pass
            return "success"
        except RuntimeError as e:
            return str(e)

    # test exception where column name is incorrect
    assert "'num1' of 'input_columns' doesn't exist" in test_exceptions_config(4, 2, ["num1"], simple_copy)
    assert "expects: 2 rows returned from 'per_batch_map', got: 4" in test_exceptions_config(4, 2, ["num"], concat_copy)
    assert "expects: 4 rows returned from 'per_batch_map', got: 2" in test_exceptions_config(4, 4, ["num"], shrink_copy)


if __name__ == '__main__':
    logger.info("Running test_var_batch_map.py test_batch_corner_cases() function")
    test_batch_corner_cases()

    logger.info("Running test_var_batch_map.py test_variable_size_batch() function")
    test_variable_size_batch()

    logger.info("Running test_var_batch_map.py test_get_batchsize_on_callable_batchsize() function")
    test_get_batchsize_on_callable_batchsize()

    logger.info("Running test_var_batch_map.py test_basic_batch_map() function")
    test_basic_batch_map()

    logger.info("Running test_var_batch_map.py test_batch_multi_col_map() function")
    test_batch_multi_col_map()

    logger.info("Running test_var_batch_map.py test_var_batch_multi_col_map() function")
    test_var_batch_multi_col_map()

    logger.info("Running test_var_batch_map.py test_var_batch_var_resize() function")
    test_var_batch_var_resize()

    logger.info("Running test_var_batch_map.py test_exception() function")
    test_exception()

    logger.info("Running test_var_batch_map.py test_multi_col_map() function")
    test_multi_col_map()

    logger.info("Running test_var_batch_map.py test_exceptions_2() function")
    test_exceptions_2()
