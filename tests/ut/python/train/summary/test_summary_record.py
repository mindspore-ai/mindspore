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
# ============================================================================
"""test_summary_abnormal_input"""
import os
import shutil
import tempfile

import numpy as np
import pytest

from mindspore.common.tensor import Tensor
from mindspore.train.summary.summary_record import SummaryRecord


def get_test_data(step):
    """ get_test_data """
    test_data_list = []
    tag1 = "x1[:Scalar]"
    tag2 = "x2[:Scalar]"
    np1 = np.array(step + 1).astype(np.float32)
    np2 = np.array(step + 2).astype(np.float32)

    dict1 = {}
    dict1["name"] = tag1
    dict1["data"] = Tensor(np1)

    dict2 = {}
    dict2["name"] = tag2
    dict2["data"] = Tensor(np2)

    test_data_list.append(dict1)
    test_data_list.append(dict2)

    return test_data_list


class TestSummaryRecord:
    """Test SummaryRecord"""
    def setup_class(self):
        """Run before test this class."""
        self.base_summary_dir = tempfile.mkdtemp(suffix='summary')

    def teardown_class(self):
        """Run after test this class."""
        if os.path.exists(self.base_summary_dir):
            shutil.rmtree(self.base_summary_dir)

    @pytest.mark.parametrize("log_dir", ["", None, 32])
    def test_log_dir_with_type_error(self, log_dir):
        with pytest.raises(TypeError):
            with SummaryRecord(log_dir):
                pass

    @pytest.mark.parametrize("raise_exception", ["", None, 32])
    def test_raise_exception_with_type_error(self, raise_exception):
        summary_dir = tempfile.mkdtemp(dir=self.base_summary_dir)
        with pytest.raises(TypeError) as exc:
            with SummaryRecord(log_dir=summary_dir, raise_exception=raise_exception):
                pass

        assert "raise_exception" in str(exc.value)

    @pytest.mark.parametrize("step", ["str"])
    def test_step_of_record_with_type_error(self, step):
        summary_dir = tempfile.mkdtemp(dir=self.base_summary_dir)
        with pytest.raises(TypeError):
            with SummaryRecord(summary_dir) as sr:
                sr.record(step)
