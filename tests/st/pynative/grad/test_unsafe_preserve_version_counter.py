# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Test _unsafe_preserve_version_counter"""
import numpy as np
import pytest
from mindspore import Tensor
from mindspore.common import _unsafe_preserve_version_counter
from tests.mark_utils import arg_mark


def test_preserve_version_counter_single_tensor():
    """
    Feature: _unsafe_preserve_version_counter
    Description: Test that version counter is preserved for a single tensor after inplace mutation.
    Expectation: Version counter should be restored to its original value after exiting the context.
    """
    x = Tensor(np.ones((2, 3), dtype=np.float32))
    assert x._version == 0  # pylint:disable=protected-access

    with _unsafe_preserve_version_counter(x):
        x[0] = 9
        assert x._version == 1  # pylint:disable=protected-access

    assert x._version == 0  # pylint:disable=protected-access


def test_preserve_version_counter_multiple_tensors():
    """
    Feature: _unsafe_preserve_version_counter
    Description: Test that version counters are preserved for multiple tensors.
    Expectation: All version counters should be restored after exiting the context.
    """
    x = Tensor(np.ones((2, 3), dtype=np.float32))
    y = Tensor(np.ones((3, 2), dtype=np.float32))
    assert x._version == 0  # pylint:disable=protected-access
    assert y._version == 0  # pylint:disable=protected-access

    with _unsafe_preserve_version_counter((x, y)):
        x[0] = 9
        y[0] = 5
        assert x._version == 1  # pylint:disable=protected-access
        assert y._version == 1  # pylint:disable=protected-access

    assert x._version == 0  # pylint:disable=protected-access
    assert y._version == 0  # pylint:disable=protected-access


def test_preserve_version_counter_multiple_mutations():
    """
    Feature: _unsafe_preserve_version_counter
    Description: Test preservation after multiple inplace mutations within the context.
    Expectation: Version counter should be restored regardless of mutation count.
    """
    x = Tensor(np.ones((2, 3), dtype=np.float32))
    assert x._version == 0  # pylint:disable=protected-access

    with _unsafe_preserve_version_counter(x):
        x[0] = 9
        x[1] = 8
        assert x._version == 2  # pylint:disable=protected-access

    assert x._version == 0  # pylint:disable=protected-access


def test_preserve_version_counter_nonzero_initial():
    """
    Feature: _unsafe_preserve_version_counter
    Description: Test preservation when tensor already has a non-zero version counter.
    Expectation: Version counter should be restored to the pre-context value, not zero.
    """
    x = Tensor(np.ones((2, 3), dtype=np.float32))
    x[0] = 9
    assert x._version == 1  # pylint:disable=protected-access

    with _unsafe_preserve_version_counter(x):
        x[1] = 8
        assert x._version == 2  # pylint:disable=protected-access

    assert x._version == 1  # pylint:disable=protected-access


def test_preserve_version_counter_no_mutation():
    """
    Feature: _unsafe_preserve_version_counter
    Description: Test context manager when no mutations happen inside.
    Expectation: Version counter should remain unchanged.
    """
    x = Tensor(np.ones((2, 3), dtype=np.float32))
    assert x._version == 0  # pylint:disable=protected-access

    with _unsafe_preserve_version_counter(x):
        pass

    assert x._version == 0  # pylint:disable=protected-access


def test_preserve_version_counter_invalid_input():
    """
    Feature: _unsafe_preserve_version_counter
    Description: Test that passing an invalid type raises AssertionError.
    Expectation: AssertionError should be raised for non-tensor, non-tuple input.
    """
    with pytest.raises(AssertionError):
        _unsafe_preserve_version_counter([Tensor(np.ones((2,), dtype=np.float32))])


def test_preserve_version_counter_nested():
    """
    Feature: _unsafe_preserve_version_counter
    Description: Test nested context managers on the same tensor.
    Expectation: Each context manager should independently restore the version it captured.
    """
    x = Tensor(np.ones((2, 3), dtype=np.float32))
    assert x._version == 0  # pylint:disable=protected-access

    with _unsafe_preserve_version_counter(x):
        x[0] = 9
        assert x._version == 1  # pylint:disable=protected-access

        with _unsafe_preserve_version_counter(x):
            x[1] = 8
            assert x._version == 2  # pylint:disable=protected-access

        # Inner context restores to version 1
        assert x._version == 1  # pylint:disable=protected-access

    # Outer context restores to version 0
    assert x._version == 0  # pylint:disable=protected-access


def test_preserve_version_counter_exception():
    """
    Feature: _unsafe_preserve_version_counter
    Description: Test that version counter is restored even when an exception occurs.
    Expectation: Version counter should be restored despite the exception.
    """
    x = Tensor(np.ones((2, 3), dtype=np.float32))
    assert x._version == 0  # pylint:disable=protected-access

    with pytest.raises(ValueError):
        with _unsafe_preserve_version_counter(x):
            x[0] = 9
            assert x._version == 1  # pylint:disable=protected-access
            raise ValueError("test error")

    assert x._version == 0  # pylint:disable=protected-access


def test_preserve_version_counter_data_unchanged():
    """
    Feature: _unsafe_preserve_version_counter
    Description: Test that the context manager only preserves version, not data.
    Expectation: Data mutations should persist, only version counter should be restored.
    """
    x = Tensor(np.ones((2, 3), dtype=np.float32))

    with _unsafe_preserve_version_counter(x):
        x[0] = 9

    assert x._version == 0  # pylint:disable=protected-access
    # Data mutation should persist
    expected = np.ones((2, 3), dtype=np.float32)
    expected[0] = 9
    assert np.allclose(x.asnumpy(), expected)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          mem_peak=1024)
def test_all():
    """
    Feature: _unsafe_preserve_version_counter
    Description: Test all scenarios of _unsafe_preserve_version_counter.
    Expectation: All sub-tests pass.
    """
    test_preserve_version_counter_single_tensor()
    test_preserve_version_counter_multiple_tensors()
    test_preserve_version_counter_multiple_mutations()
    test_preserve_version_counter_nonzero_initial()
    test_preserve_version_counter_no_mutation()
    test_preserve_version_counter_invalid_input()
    test_preserve_version_counter_nested()
    test_preserve_version_counter_exception()
    test_preserve_version_counter_data_unchanged()
