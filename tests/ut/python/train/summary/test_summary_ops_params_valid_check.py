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
"""Test summary function of ops params valid check."""
import os
import tempfile
import shutil
from enum import Enum

import numpy as np
import pytest


import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.train.summary.summary_record import SummaryRecord
from tests.security_utils import security_off_wrap


class SummaryEnum(Enum):
    """Summary enum."""
    IMAGE = P.ImageSummary.__name__
    SCALAR = P.ScalarSummary.__name__
    TENSOR = P.TensorSummary.__name__
    HISTOGRAM = P.HistogramSummary.__name__


class SummaryNet(nn.Cell):
    """Summary net definition."""
    def __init__(self, summary_type, tag, data):
        super(SummaryNet, self).__init__()
        self.tag = tag
        self.data = data
        self.summary_fn = getattr(P, summary_type)()
        self.one = Tensor(np.array([1]).astype(np.float32))
        self.add = P.Add()

    def construct(self):
        self.summary_fn(self.tag, self.data)
        return self.add(self.one, self.one)


class TestSummaryOps:
    """Test summary operators."""
    summary_dir = ''

    @classmethod
    def run_case(cls, net):
        """ run_case """
        net.set_train()
        steps = 10
        with SummaryRecord(cls.summary_dir) as test_writer:
            for i in range(1, steps):
                net()
                test_writer.record(i)

    @classmethod
    def setup_class(cls):
        """Run before class."""
        if not os.path.exists(cls.summary_dir):
            cls.summary_dir = tempfile.mkdtemp(suffix='_summary')

    @classmethod
    def teardown_class(cls):
        """Run after class."""
        if os.path.exists(cls.summary_dir):
            shutil.rmtree(cls.summary_dir)

    @security_off_wrap
    @pytest.mark.parametrize(
        "summary_type, value",
        [
            (SummaryEnum.SCALAR.value, Tensor(1)),
            (SummaryEnum.SCALAR.value, Tensor(np.array([1]))),
            (SummaryEnum.IMAGE.value, Tensor(np.array([[[[1], [2], [3], [4]]]]))),
            (SummaryEnum.TENSOR.value, Tensor(np.array([[1], [2], [3], [4]]))),
            (SummaryEnum.HISTOGRAM.value, Tensor(np.array([[1], [2], [3], [4]]))),
        ])
    def test_summary_success(self, summary_type, value):
        """Test summary success with valid tag and valid data."""
        net = SummaryNet(summary_type, tag='tag', data=value)
        TestSummaryOps.run_case(net)

    @security_off_wrap
    @pytest.mark.parametrize(
        "summary_type",
        [
            SummaryEnum.SCALAR.value,
            SummaryEnum.IMAGE.value,
            SummaryEnum.HISTOGRAM.value,
            SummaryEnum.TENSOR.value
        ])
    def test_summary_tag_is_none(self, summary_type):
        """Test summary tag is None, all summary operator validation rules are consistent."""
        net = SummaryNet(summary_type, tag=None, data=Tensor(0))
        with pytest.raises(TypeError):
            TestSummaryOps.run_case(net)

    @security_off_wrap
    @pytest.mark.parametrize(
        "summary_type",
        [
            SummaryEnum.SCALAR.value,
            SummaryEnum.IMAGE.value,
            SummaryEnum.HISTOGRAM.value,
            SummaryEnum.TENSOR.value
        ])
    def test_summary_tag_is_empty_string(self, summary_type):
        """Test summary tag is a empty string, all summary operator validation rules are consistent."""
        net = SummaryNet(summary_type, tag='', data=Tensor(0))
        with pytest.raises(ValueError):
            TestSummaryOps.run_case(net)

    @security_off_wrap
    @pytest.mark.parametrize("tag", [123, True, Tensor(0)])
    def test_summary_tag_is_not_string(self, tag):
        """Test summary tag is not a string, all summary operator validation rules are consistent."""
        # All summary operator validation rules are consistent, so we only test scalar summary.
        net = SummaryNet(SummaryEnum.SCALAR.value, tag=tag, data=Tensor(0))
        with pytest.raises(TypeError):
            TestSummaryOps.run_case(net)

    @security_off_wrap
    @pytest.mark.parametrize("value", [123, True, 'data'])
    def test_summary_value_type_invalid(self, value):
        """Test the type of summary value is invalid, all summary operator validation rules are consistent."""
        # All summary operator validation rules are consistent, so we only test scalar summary.
        net = SummaryNet(SummaryEnum.SCALAR.value, tag='tag', data=value)
        with pytest.raises(TypeError):
            TestSummaryOps.run_case(net)

    @security_off_wrap
    @pytest.mark.parametrize(
        "summary_type, value",
        [
            (SummaryEnum.IMAGE.value, Tensor(np.array([1, 2]))),
            (SummaryEnum.TENSOR.value, Tensor(0)),
            (SummaryEnum.HISTOGRAM.value, Tensor(0))
        ])

    def test_value_shape_invalid(self, summary_type, value):
        """Test invalid shape of every summary operators."""
        net = SummaryNet(summary_type, tag='tag', data=value)
        with pytest.raises(ValueError):
            TestSummaryOps.run_case(net)
