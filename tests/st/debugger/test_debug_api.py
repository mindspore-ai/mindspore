# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Test debug API."""
import pytest

from mindspore.offline_debug.dump_analyzer import DumpAnalyzer
from mindspore.offline_debug.watchpoints import TensorTooLargeWatchpoint


@pytest.mark.skip(reason="Feature under development.")
def test_export_graphs():
    """Test debug API."""
    my_run = DumpAnalyzer(
        summary_dir="/path/to/summary-dir1"
    )

    # Export the info about computational graph. Should support multi graphs.
    my_run.export_graphs()


@pytest.mark.skip(reason="Feature under development.")
def test_select_tensors():
    """Test debug API."""
    my_run = DumpAnalyzer(
        summary_dir="/path/to/summary-dir2"
    )

    # Find the interested tensors.
    matched_tensors = my_run.select_tensors(".*conv1.*", use_regex=True)
    assert matched_tensors == []


@pytest.mark.skip(reason="Feature under development.")
def test_check_watchpoints_all_iterations():
    """Test debug API."""
    my_run = DumpAnalyzer(
        summary_dir="/path/to/summary-dir3"
    )

    # Checking all the iterations.
    watchpoints = [
        TensorTooLargeWatchpoint(
            tensors=my_run.select_tensors(
                "(*.weight^)|(*.bias^)", use_regex=True),
            abs_mean_gt=0.1)
    ]

    watch_point_hits = my_run.check_watchpoints(watchpoints=watchpoints)
    assert watch_point_hits == []


@pytest.mark.skip(reason="Feature under development.")
def test_check_watchpoints_one_iteration():
    """Test debug API."""
    my_run = DumpAnalyzer(
        summary_dir="/path/to/summary-dir4"
    )
    # Checking specific iteration.
    watchpoints = [
        TensorTooLargeWatchpoint(
            tensors=my_run.select_tensors(
                "(*.weight^)|(*.bias^)", use_regex=True,
                iterations=[1]),
            abs_mean_gt=0.1)
    ]

    watch_point_hits = my_run.check_watchpoints(watchpoints=watchpoints)
    assert watch_point_hits == []
