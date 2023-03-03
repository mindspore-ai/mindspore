# Copyright 2023 Huawei Technologies Co., Ltd
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
Test map operation in debug mode
"""

import numpy as np
import pytest
import mindspore.dataset as ds

# Need to run all these tests in separate processes since
# the global configuration setting of debug_mode may impact other tests running in parallel.
pytestmark = pytest.mark.forked

SEED_VAL = 0  # seed will be set internally in debug mode, save original seed value to restore.


def setup_function():
    global SEED_VAL
    SEED_VAL = ds.config.get_seed()


def teardown_function():
    ds.config.set_seed(SEED_VAL)


def generator_one_col(num):
    """ Use to create a dataset with one column """
    for i in range(num):
        yield i


def generator_two_cols(num):
    """ Use to create a dataset with two columns """
    for i in range(num):
        yield (i, i + 100)


def generator_np_one_col(num):
    """ Use to create a dataset with one column of NumPy data """
    for i in range(num):
        yield (np.array([i]),)


def generator_np_two_cols(num):
    """ Use to create a dataset with two columns of NumPy data """
    for i in range(num):
        yield (np.array([i]), np.array([i + 100]))


def generator_np_five_cols(num):
    """ Use to create a dataset with five columns of NumPy data """
    for i in range(num):
        yield (np.array([i]), np.array([i * 10]), np.array([i * 100]), np.array([i * 1000]), np.array([i * 10000]))


@pytest.mark.parametrize("my_debug_mode", (False, True))
def test_genmap1(my_debug_mode):
    """
    Feature: Pipeline debug mode
    Description: Test GeneratorDataset with 1 column, map 1 column -> 1 column (renamed)
    Expectation: The dataset is processed as expected
    """
    # Set configuration
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    my_num = 5
    data = ds.GeneratorDataset(generator_one_col(my_num), ["col1"],
                               python_multiprocessing=False, num_parallel_workers=1)
    data = data.map(operations=[lambda x: x],
                    input_columns=["col1"],
                    output_columns=["output_col1"],
                    python_multiprocessing=False, num_parallel_workers=1)

    row_count = 0
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item.keys()) == 1
        assert item["output_col1"] == row_count
        row_count += 1
    assert row_count == my_num

    # Restore configuration
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode", (False, True))
def test_gennpmap1(my_debug_mode):
    """
    Feature: Pipeline debug mode
    Description: Test GeneratorDataset with 1 column, map 1 column -> 1 column (renamed)
    Expectation: The dataset is processed as expected
    """
    # Set configuration
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    my_num = 5
    data = ds.GeneratorDataset(generator_np_one_col(my_num), ["col1"],
                               python_multiprocessing=False, num_parallel_workers=1)
    data = data.map(operations=[lambda x: x],
                    input_columns=["col1"],
                    output_columns=["col1A"],
                    python_multiprocessing=False, num_parallel_workers=1)

    row_count = 0
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item.keys()) == 1
        np.testing.assert_array_equal(item["col1A"], [row_count])
        row_count += 1
    assert row_count == my_num

    # Restore configuration
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode", (False, True))
def test_gennpmap1_1to3(my_debug_mode):
    """
    Feature: Pipeline debug mode
    Description: Test GeneratorDataset with 1 column, map 1 column -> 3 columns
    Expectation: The dataset is processed as expected
    """

    def copy_1column(x):
        return x, x, x

    # Set configuration
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    my_num = 5
    data = ds.GeneratorDataset(generator_np_one_col(my_num), ["col1"],
                               python_multiprocessing=False, num_parallel_workers=1)
    data = data.map(operations=[copy_1column],
                    input_columns=["col1"],
                    output_columns=["col1A", "col1B", "col1C"],
                    python_multiprocessing=False, num_parallel_workers=1)

    row_count = 0
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item.keys()) == 3
        np.testing.assert_array_equal(item["col1A"], [row_count])
        np.testing.assert_array_equal(item["col1B"], [row_count])
        np.testing.assert_array_equal(item["col1C"], [row_count])
        row_count += 1
    assert row_count == my_num

    # Restore configuration
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode", (False, True))
def test_gennpmap2(my_debug_mode):
    """
    Feature: Pipeline debug mode
    Description: Test GeneratorDataset with 2 columns, map 2 columns -> 2 columns (swapped)
    Expectation: The dataset is processed as expected
    """

    def swap_2columns(x, y):
        return y, x

    # Set configuration
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    my_num = 5
    data = ds.GeneratorDataset(generator_np_two_cols(my_num), ["col1", "col2"],
                               python_multiprocessing=False, num_parallel_workers=1)
    data = data.map(operations=[swap_2columns],
                    input_columns=["col1", "col2"],
                    output_columns=["col2A", "col1A"],
                    python_multiprocessing=False, num_parallel_workers=1)

    row_count = 0
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item.keys()) == 2
        np.testing.assert_array_equal(item["col1A"], [row_count])
        np.testing.assert_array_equal(item["col2A"], [row_count + 100])
        row_count += 1
    assert row_count == my_num

    # Restore configuration
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode", (False, True))
def test_gennpmap2_2to1_dropcol1(my_debug_mode):
    """
    Feature: Pipeline debug mode
    Description: Test GeneratorDataset with 2 columns, map 2 columns -> 1 column (drop 1st column)
    Expectation: The dataset is processed as expected
    """

    def drop_1st(x, y):
        return y

    # Set configuration
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    my_num = 5
    data = ds.GeneratorDataset(generator_np_two_cols(my_num), ["col1", "col2"],
                               python_multiprocessing=False, num_parallel_workers=1)
    data = data.map(operations=[drop_1st],
                    input_columns=["col1", "col2"],
                    output_columns=["col2A"],
                    python_multiprocessing=False, num_parallel_workers=1)

    row_count = 0
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item.keys()) == 1
        assert "col2A" in item.keys()
        np.testing.assert_array_equal(item["col2A"], [row_count + 100])
        row_count += 1
    assert row_count == my_num

    # Restore configuration
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode", (False, True))
def test_gennpmap2_2to1_dropcol2(my_debug_mode):
    """
    Feature: Pipeline debug mode
    Description: Test GeneratorDataset with 2 columns, map 2 columns -> 1 column (drop 2nd)
    Expectation: The dataset is processed as expected
    """

    def drop_2nd(x, y):
        return x

    # Set configuration
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    my_num = 5
    data = ds.GeneratorDataset(generator_np_two_cols(my_num), ["col1", "col2"],
                               python_multiprocessing=False, num_parallel_workers=1)
    data = data.map(operations=[drop_2nd],
                    input_columns=["col1", "col2"],
                    output_columns=["col1A"],
                    python_multiprocessing=False, num_parallel_workers=1)

    row_count = 0
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item.keys()) == 1
        assert "col1A" in item.keys()
        np.testing.assert_array_equal(item["col1A"], [row_count])
        row_count += 1
    assert row_count == my_num

    # Restore configuration
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode", (False, True))
def test_gennpmap2_2to1_sumcols(my_debug_mode):
    """
    Feature: Pipeline debug mode
    Description: Test GeneratorDataset with 2 columns, map 2 columns -> 1 column (sum both cols)
    Expectation: The dataset is processed as expected
    """

    def sum_cols(x, y):
        return x + y

    # Set configuration
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    my_num = 5
    data = ds.GeneratorDataset(generator_np_two_cols(my_num), ["col1", "col2"],
                               python_multiprocessing=False, num_parallel_workers=1)
    data = data.map(operations=[sum_cols],
                    input_columns=["col1", "col2"],
                    output_columns=["ocol1"],
                    python_multiprocessing=False, num_parallel_workers=1)

    row_count = 0
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item.keys()) == 1
        assert "ocol1" in item.keys()
        np.testing.assert_array_equal(item["ocol1"], [row_count * 2 + 100])
        row_count += 1
    assert row_count == my_num

    # Restore configuration
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode", (False, True))
def test_gennpmap2_2to1_multimaps(my_debug_mode):
    """
    Feature: Pipeline debug mode
    Description: Test GeneratorDataset with 2 columns, map 2 columns -> 1 column (multiple maps)
    Expectation: The dataset is processed as expected
    """
    # Set configuration
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    my_num = 5
    data = ds.GeneratorDataset(generator_np_two_cols(my_num), ["col1", "col2"],
                               python_multiprocessing=False, num_parallel_workers=1)
    # Create map ops with implicit output_column names
    data = data.map(operations=(lambda x: x), input_columns=["col1"])
    data = data.map(operations=(lambda x: x + 2000), input_columns=["col2"])
    # Note: #input_columns > #output_columns
    data = data.map(operations=[lambda x, y: x + y],
                    input_columns=["col1", "col2"],
                    output_columns=["ocol3"],
                    python_multiprocessing=False, num_parallel_workers=1)

    row_count = 0
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item.keys()) == 1
        assert "ocol3" in item.keys()
        np.testing.assert_array_equal(item["ocol3"], [row_count * 2 + 2100])
        row_count += 1
    assert row_count == my_num

    # Restore configuration
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode", (False, True))
def test_gennpmap2_2to2_uniqueoutputcols(my_debug_mode):
    """
    Feature: Pipeline debug mode
    Description: Test GeneratorDataset with 2 columns, map 2 columns -> 2 different columns
    Expectation: The dataset is processed as expected
    """
    # Set configuration
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    my_num = 5
    data = ds.GeneratorDataset(generator_np_two_cols(my_num), ["col1", "col2"],
                               python_multiprocessing=False, num_parallel_workers=1)
    # Note: #input_columns == #output_columns
    data = data.map(operations=[(lambda x, y: (x + y, y, x)),
                                (lambda z, y, x: (z, x * (y - x)))],
                    input_columns=["col1", "col2"],
                    output_columns=["ocol1", "ocol2"],
                    python_multiprocessing=False, num_parallel_workers=1)

    row_count = 0
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item.keys()) == 2
        assert "ocol1" in item.keys()
        assert "ocol2" in item.keys()
        np.testing.assert_array_equal(item["ocol1"], [row_count * 2 + 100])
        np.testing.assert_array_equal(item["ocol2"], [row_count * 100])
        row_count += 1
    assert row_count == my_num

    # Restore configuration
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode", (False, True))
def test_gennpmap2_2to3_uniqueoutputcols(my_debug_mode):
    """
    Feature: Pipeline debug mode
    Description: Test GeneratorDataset with 2 columns, map 2 columns -> 3 columns
    Expectation: The dataset is processed as expected
    """
    # Set configuration
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    my_num = 5
    data = ds.GeneratorDataset(generator_np_two_cols(my_num), ["col1", "col2"],
                               python_multiprocessing=False, num_parallel_workers=1)
    # Note: #input_columns < #output_columns
    # Use unique output_column names
    data = data.map(operations=[(lambda x, y: (x + y, y, x)),
                                (lambda z, y, x: (z, y - x, x * (y - x)))],
                    input_columns=["col1", "col2"],
                    output_columns=["ocol1", "ocol2", "ocol3"],
                    python_multiprocessing=False, num_parallel_workers=1)

    row_count = 0
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item.keys()) == 3
        assert "ocol1" in item.keys()
        assert "ocol2" in item.keys()
        assert "ocol3" in item.keys()
        np.testing.assert_array_equal(item["ocol1"], [row_count * 2 + 100])
        np.testing.assert_array_equal(item["ocol2"], [100])
        np.testing.assert_array_equal(item["ocol3"], [row_count * 100])
        row_count += 1
    assert row_count == my_num

    # Restore configuration
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode", (False, True))
def test_gennpmap2_2to3_includeinputcol(my_debug_mode):
    """
    Feature: Pipeline debug mode
    Description: Test GeneratorDataset with 2 columns, map 1 column -> 2 columns, dataset col2 included in result
    Expectation: The dataset is processed as expected
    """
    # Set configuration
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    my_num = 5
    data = ds.GeneratorDataset(generator_np_two_cols(my_num), ["col1", "col2"],
                               python_multiprocessing=False, num_parallel_workers=1)
    # Note: input_column col2 not explicitly in any map op, but is still apart of output resultant dataset
    data = data.map(operations=[lambda x: (x - 1, x + 1),
                                lambda x, y: (x + 1, 10 * (y - 1))],
                    input_columns=["col1"],
                    output_columns=["ocol1", "ocol2"],
                    python_multiprocessing=False, num_parallel_workers=1)

    row_count = 0
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item.keys()) == 3
        assert "ocol1" in item.keys()
        assert "ocol2" in item.keys()
        assert "col2" in item.keys()
        np.testing.assert_array_equal(item["ocol1"], [row_count])
        np.testing.assert_array_equal(item["ocol2"], [row_count * 10])
        np.testing.assert_array_equal(item["col2"], [row_count + 100])
        row_count += 1
    assert row_count == my_num

    # Restore configuration
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode", (False, True))
def test_gennpmap2_reuseinputcolname(my_debug_mode):
    """
    Feature: Pipeline debug mode
    Description: Test GeneratorDataset with 2 columns, map 1 column -> 2 columns. Beware col2 column name reused.
    Expectation: The dataset is processed as expected
    """
    # Set configuration
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    my_num = 5
    data = ds.GeneratorDataset(generator_np_two_cols(my_num), ["col1", "col2"],
                               python_multiprocessing=False, num_parallel_workers=1)
    # Note: input_column name col2 reused in map op (that does not use col2) as an output_column name.
    # This is a malformed pipeline.
    data = data.map(operations=[lambda x: (x - 1, x + 1),
                                lambda x, y: (x + 1, 10 * (y - 1))],
                    input_columns=["col1"],
                    output_columns=["col2", "ocol2"],
                    python_multiprocessing=False, num_parallel_workers=1)

    if my_debug_mode:
        with pytest.raises(RuntimeError) as error_info:
            for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                pass
        assert "[ERROR] Invalid column id encountered." in str(error_info.value)
    else:
        # For non-pull mode, map result of 1st output column is ignored and input value of col2 is used in result
        row_count = 0
        for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
            assert len(item.keys()) == 2
            assert "ocol2" in item.keys()
            assert "col2" in item.keys()
            np.testing.assert_array_equal(item["col2"], [row_count + 100])
            np.testing.assert_array_equal(item["ocol2"], [row_count * 10])
            row_count += 1
        assert row_count == my_num

    # Restore configuration
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode", (False, True))
def test_gennpmap1_1to3to2(my_debug_mode):
    """
    Feature: Pipeline debug mode
    Description: Test GeneratorDataset with 1 column, map 1 column -> 3 columns -> 2 columns (drop 2nd column)
    Expectation: The dataset is processed as expected
    """
    # Set configuration
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    my_num = 5
    data = ds.GeneratorDataset(generator_np_one_col(my_num), ["col1"],
                               python_multiprocessing=False, num_parallel_workers=1)
    data = data.map(operations=[lambda a: (a, a * 10, a * 100),
                                lambda a, b, c: (a, c)],
                    input_columns=["col1"],
                    output_columns=["col1A", "col1C"],
                    python_multiprocessing=False, num_parallel_workers=1)

    row_count = 0
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item.keys()) == 2
        np.testing.assert_array_equal(item["col1A"], [row_count])
        np.testing.assert_array_equal(item["col1C"], [row_count * 100])
        row_count += 1
    assert row_count == my_num

    # Restore configuration
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode", (False, True))
def test_gennpmap1_1to3_3to2_newoutputcolnames(my_debug_mode):
    """
    Feature: Pipeline debug mode
    Description: Test GeneratorDataset with 1 column, map#1 1 column -> 3 columns,
        map#2 3 columns -> 2 columns (drop 2nd column).  New output column names for each map.
    Expectation: The dataset is processed as expected
    """
    # Set configuration
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    my_num = 5
    data = ds.GeneratorDataset(generator_np_one_col(my_num), ["col1"],
                               python_multiprocessing=False, num_parallel_workers=1)
    data = data.map(operations=[lambda x: (x, x * 10, x * 100)],
                    input_columns=["col1"],
                    output_columns=["col1A", "col1B", "col1C"],
                    python_multiprocessing=False, num_parallel_workers=1)
    data = data.map(operations=[lambda a, b, c: (a, c)],
                    input_columns=["col1A", "col1B", "col1C"],
                    output_columns=["col2A", "col2C"],
                    python_multiprocessing=False, num_parallel_workers=1)

    row_count = 0
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item.keys()) == 2
        np.testing.assert_array_equal(item["col2A"], [row_count])
        np.testing.assert_array_equal(item["col2C"], [row_count * 100])
        row_count += 1
    assert row_count == my_num

    # Restore configuration
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode", (False, True))
def test_gennpmap1_1to3_3to2_varoutputcolnames(my_debug_mode):
    """
    Feature: Pipeline debug mode
    Description: Test GeneratorDataset with 1 column, map#1 1 column -> 3 columns with new column names.
        map#2 3 columns -> 2 columns (drop 2nd column) and keep output_column names..
    Expectation: The dataset is processed as expected
    """
    # Set configuration
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    my_num = 5
    data = ds.GeneratorDataset(generator_np_one_col(my_num), ["col1"],
                               python_multiprocessing=False, num_parallel_workers=1)
    data = data.map(operations=[lambda x: (x, x * 10, x * 100)],
                    input_columns=["col1"],
                    output_columns=["col1A", "col1B", "col1C"],
                    python_multiprocessing=False, num_parallel_workers=1)
    data = data.map(operations=[lambda a, b, c: (a, c)],
                    input_columns=["col1A", "col1B", "col1C"],
                    output_columns=["col1A", "col1C"],
                    python_multiprocessing=False, num_parallel_workers=1)

    row_count = 0
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item.keys()) == 2
        np.testing.assert_array_equal(item["col1A"], [row_count])
        np.testing.assert_array_equal(item["col1C"], [row_count * 100])
        row_count += 1
    assert row_count == my_num

    # Restore configuration
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode", (False, True))
def test_gennpmap1_1to5_5to3(my_debug_mode):
    """
    Feature: Pipeline debug mode
    Description: Test GeneratorDataset with 1 column, map#1 1 column -> 5 columns, map#2 5 columns -> 3 columns
        (drop 1st & 4th columns).
    Expectation: The dataset is processed as expected
    """
    # Set configuration
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    my_num = 5
    data = ds.GeneratorDataset(generator_np_one_col(my_num), ["col1"],
                               python_multiprocessing=False, num_parallel_workers=1)
    data = data.map(operations=[lambda x: (x, x * 10, x * 100, x * 1000, x * 10000)],
                    input_columns=["col1"],
                    output_columns=["col1A", "col1B", "col1C", "col1D", "col1E"],
                    python_multiprocessing=False, num_parallel_workers=1)
    # Drop 1st and 4th columns
    data = data.map(operations=[lambda a, b, c, d, e: (b, c, e)],
                    input_columns=["col1A", "col1B", "col1C", "col1D", "col1E"],
                    output_columns=["col1B", "col1C", "col1E"],
                    python_multiprocessing=False, num_parallel_workers=1)

    row_count = 0
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item.keys()) == 3
        np.testing.assert_array_equal(item["col1B"], [row_count * 10])
        np.testing.assert_array_equal(item["col1C"], [row_count * 100])
        np.testing.assert_array_equal(item["col1E"], [row_count * 10000])
        row_count += 1
    assert row_count == my_num

    # Restore configuration
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode, dict_iterator_flag", [(False, True), (True, True), (True, False)])
def test_gennpmap5_5to3(my_debug_mode, dict_iterator_flag):
    """
    Feature: Pipeline debug mode
    Description: Test GeneratorDataset with 5 columns, map 5 columns -> 3 columns (drop 1st & 4th columns)
    Expectation: The dataset is processed as expected
    """
    # Set configuration
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    my_num = 5
    data = ds.GeneratorDataset(generator_np_five_cols(my_num), ["col1", "col2", "col3", "col4", "col5"],
                               python_multiprocessing=False, num_parallel_workers=1)
    # Drop 1st and 4th columns
    data = data.map(operations=[lambda a, b, c, d, e: (b, c, e)],
                    input_columns=["col1", "col2", "col3", "col4", "col5"],
                    output_columns=["col2", "col3", "col5"],
                    python_multiprocessing=False, num_parallel_workers=1)

    if dict_iterator_flag:
        itr = data.create_dict_iterator(num_epochs=1, output_numpy=True)
    else:
        itr = data.create_tuple_iterator(num_epochs=1, output_numpy=True)

    row_count = 0
    for item in itr:
        assert len(item) == 3
        if dict_iterator_flag:
            # Check output from dict iterator
            np.testing.assert_array_equal(item["col2"], [row_count * 10])
            np.testing.assert_array_equal(item["col3"], [row_count * 100])
            np.testing.assert_array_equal(item["col5"], [row_count * 10000])
        else:
            # Check output from tuple iterator
            np.testing.assert_array_equal(item[0], [row_count * 10])
            np.testing.assert_array_equal(item[1], [row_count * 100])
            np.testing.assert_array_equal(item[2], [row_count * 10000])
        row_count += 1
    assert row_count == my_num

    # Restore configuration
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode, dict_iterator_flag", [(False, False), (True, True), (True, False)])
def test_gennpmap5_3to2(my_debug_mode, dict_iterator_flag):
    """
    Feature: Pipeline debug mode
    Description: Test GeneratorDataset with 5 columns, map 3 columns -> 2 columns (1st & 4th columns unused in map).
    Expectation: The dataset is processed as expected
    """
    # Set configuration
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    my_num = 5
    data = ds.GeneratorDataset(generator_np_five_cols(my_num), ["col1", "col2", "col3", "col4", "col5"],
                               python_multiprocessing=False, num_parallel_workers=1)
    # col1 and col4 are unused in map, but still apart of final dataset
    data = data.map(operations=[lambda a, b, c: (a, c)],
                    input_columns=["col2", "col3", "col5"],
                    output_columns=["col2", "col5"],
                    python_multiprocessing=False, num_parallel_workers=1)

    if dict_iterator_flag:
        itr = data.create_dict_iterator(num_epochs=1, output_numpy=True)
    else:
        itr = data.create_tuple_iterator(num_epochs=1, output_numpy=True)

    row_count = 0
    for item in itr:
        assert len(item) == 4
        if dict_iterator_flag:
            # Check output from dict iterator
            np.testing.assert_array_equal(item["col1"], [row_count])
            np.testing.assert_array_equal(item["col2"], [row_count * 10])
            np.testing.assert_array_equal(item["col4"], [row_count * 1000])
            np.testing.assert_array_equal(item["col5"], [row_count * 10000])
        else:
            # Check output from tuple iterator
            np.testing.assert_array_equal(item[0], [row_count * 10])
            np.testing.assert_array_equal(item[1], [row_count * 10000])
            np.testing.assert_array_equal(item[2], [row_count])
            np.testing.assert_array_equal(item[3], [row_count * 1000])
        row_count += 1
    assert row_count == my_num

    # Restore configuration
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


if __name__ == '__main__':
    setup_function()
    test_genmap1(True)
    test_gennpmap1(True)
    test_gennpmap1_1to3(True)
    test_gennpmap2(True)
    test_gennpmap2_2to1_dropcol1(True)
    test_gennpmap2_2to1_dropcol2(True)
    test_gennpmap2_2to1_sumcols(True)
    test_gennpmap2_2to1_multimaps(True)
    test_gennpmap2_2to2_uniqueoutputcols(True)
    test_gennpmap2_2to3_uniqueoutputcols(True)
    test_gennpmap2_2to3_includeinputcol(True)
    test_gennpmap2_reuseinputcolname(True)
    test_gennpmap1_1to3to2(True)
    test_gennpmap1_1to3_3to2_newoutputcolnames(True)
    test_gennpmap1_1to3_3to2_varoutputcolnames(True)
    test_gennpmap1_1to5_5to3(True)
    test_gennpmap5_5to3(True, False)
    test_gennpmap5_3to2(True, False)
    teardown_function()
