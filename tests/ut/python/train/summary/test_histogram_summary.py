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
"""Test histogram summary."""

import logging
import os
import tempfile
import numpy as np

from mindspore.common.tensor import Tensor
from mindspore.train.summary._summary_adapter import _calc_histogram_bins
from mindspore.train.summary.summary_record import SummaryRecord, _cache_summary_tensor_data
from tests.summary_utils import SummaryReader
from tests.security_utils import security_off_wrap

CUR_DIR = os.getcwd()
SUMMARY_DIR = os.path.join(CUR_DIR, "/test_temp_summary_event_file/")

LOG = logging.getLogger("test")
LOG.setLevel(level=logging.ERROR)


def _wrap_test_data(input_data: Tensor):
    """
    Wraps test data to summary format.

    Args:
        input_data (Tensor): Input data.

    Returns:
        dict, the wrapped data.
    """

    return [{
        "name": "test_data[:Histogram]",
        "data": input_data
    }]


@security_off_wrap
def test_histogram_summary():
    """Test histogram summary."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        with SummaryRecord(tmp_dir, file_suffix="_MS_HISTOGRAM") as test_writer:
            test_data = _wrap_test_data(Tensor([[1, 2, 3], [4, 5, 6]]))
            _cache_summary_tensor_data(test_data)
            test_writer.record(step=1)
        file_name = os.path.realpath(test_writer.log_dir)
        with SummaryReader(file_name) as reader:
            event = reader.read_event()
            assert event.summary.value[0].histogram.count == 6


@security_off_wrap
def test_histogram_multi_summary():
    """Test histogram multiple step."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        with SummaryRecord(tmp_dir, file_suffix="_MS_HISTOGRAM") as test_writer:

            rng = np.random.RandomState(10)
            size = 50
            num_step = 5

            for i in range(num_step):
                arr = rng.normal(size=size)

                test_data = _wrap_test_data(Tensor(arr))
                _cache_summary_tensor_data(test_data)
                test_writer.record(step=i)

        file_name = os.path.realpath(test_writer.log_dir)
        with SummaryReader(file_name) as reader:
            for _ in range(num_step):
                event = reader.read_event()
                assert event.summary.value[0].histogram.count == size


@security_off_wrap
def test_histogram_summary_same_value():
    """Test histogram summary, input is an ones tensor."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        with SummaryRecord(tmp_dir, file_suffix="_MS_HISTOGRAM") as test_writer:
            dim1 = 100
            dim2 = 100

            test_data = _wrap_test_data(Tensor(np.ones([dim1, dim2])))
            _cache_summary_tensor_data(test_data)
            test_writer.record(step=1)

        file_name = os.path.realpath(test_writer.log_dir)
        with SummaryReader(file_name) as reader:
            event = reader.read_event()
            LOG.debug(event)

            assert len(event.summary.value[0].histogram.buckets) == _calc_histogram_bins(dim1 * dim2)


@security_off_wrap
def test_histogram_summary_high_dims():
    """Test histogram summary, input is a 4-dimension tensor."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        with SummaryRecord(tmp_dir, file_suffix="_MS_HISTOGRAM") as test_writer:
            dim = 10

            rng = np.random.RandomState(0)
            tensor_data = rng.normal(size=[dim, dim, dim, dim])
            test_data = _wrap_test_data(Tensor(tensor_data))
            _cache_summary_tensor_data(test_data)
            test_writer.record(step=1)

        file_name = os.path.realpath(test_writer.log_dir)
        with SummaryReader(file_name) as reader:
            event = reader.read_event()
            LOG.debug(event)

            assert event.summary.value[0].histogram.count == tensor_data.size


@security_off_wrap
def test_histogram_summary_nan_inf():
    """Test histogram summary, input tensor has nan."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        with SummaryRecord(tmp_dir, file_suffix="_MS_HISTOGRAM") as test_writer:
            dim1 = 100
            dim2 = 100

            arr = np.ones([dim1, dim2])
            arr[0][0] = np.nan
            arr[0][1] = np.inf
            arr[0][2] = -np.inf
            test_data = _wrap_test_data(Tensor(arr))

            _cache_summary_tensor_data(test_data)
            test_writer.record(step=1)

        file_name = os.path.realpath(test_writer.log_dir)
        with SummaryReader(file_name) as reader:
            event = reader.read_event()
            LOG.debug(event)

            assert event.summary.value[0].histogram.nan_count == 1


@security_off_wrap
def test_histogram_summary_all_nan_inf():
    """Test histogram summary, input tensor has no valid number."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        with SummaryRecord(tmp_dir, file_suffix="_MS_HISTOGRAM") as test_writer:
            test_data = _wrap_test_data(Tensor(np.array([np.nan, np.nan, np.nan, np.inf, -np.inf])))
            _cache_summary_tensor_data(test_data)
            test_writer.record(step=1)

        file_name = os.path.realpath(test_writer.log_dir)
        with SummaryReader(file_name) as reader:
            event = reader.read_event()
            LOG.debug(event)

            histogram = event.summary.value[0].histogram
            assert histogram.nan_count == 3
            assert histogram.pos_inf_count == 1
            assert histogram.neg_inf_count == 1
