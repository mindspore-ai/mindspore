# Copyright 2019 Huawei Technologies Co., Ltd
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
"""Summary gpu st."""
import os
import random
import tempfile
import shutil

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.train.summary.summary_record import SummaryRecord

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class SummaryNet(nn.Cell):
    """Summary net."""
    def __init__(self, tag_tuple=None, scalar=1):
        super(SummaryNet, self).__init__()
        self.summary_s = P.ScalarSummary()
        self.summary_i = P.ImageSummary()
        self.summary_t = P.TensorSummary()
        self.histogram_summary = P.HistogramSummary()
        self.add = P.TensorAdd()
        self.tag_tuple = tag_tuple
        self.scalar = scalar

    def construct(self, x, y, image):
        """Run summary net."""
        self.summary_i("image", image)
        self.summary_s("x1", x)
        z = self.add(x, y)
        self.summary_t("z1", z)
        self.histogram_summary("histogram", z)
        return z


def train_summary_record(test_writer, steps):
    """Train and record summary."""
    net = SummaryNet()
    out_me_dict = {}
    for i in range(0, steps):
        x = Tensor(np.array([1.1 + random.uniform(1, 10)]).astype(np.float32))
        y = Tensor(np.array([1.2 + random.uniform(1, 10)]).astype(np.float32))
        image = Tensor(np.array([[[[1.2]]]]).astype(np.float32))
        out_put = net(x, y, image)
        test_writer.record(i)
        out_me_dict[i] = out_put.asnumpy()
    return out_me_dict


class TestGpuSummary:
    """Test Gpu summary."""
    summary_dir = tempfile.mkdtemp(suffix='_gpu_summary')

    def setup_method(self):
        """Run before method."""
        if not os.path.exists(self.summary_dir):
            os.mkdir(self.summary_dir)

    def teardown_method(self):
        """Run after method."""
        if os.path.exists(self.summary_dir):
            shutil.rmtree(self.summary_dir)

    @pytest.mark.level0
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_summary_step10_summaryrecord1(self):
        """Test record 10 step summary."""
        with SummaryRecord(self.summary_dir) as test_writer:
            train_summary_record(test_writer, steps=10)
