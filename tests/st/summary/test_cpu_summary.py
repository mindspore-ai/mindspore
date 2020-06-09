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
"""Summary cpu st."""
import os
import tempfile

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from tests.summary_utils import SummaryReader
from mindspore.train.summary.summary_record import SummaryRecord

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class SummaryNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.scalar_summary = P.ScalarSummary()
        self.image_summary = P.ImageSummary()
        self.tensor_summary = P.TensorSummary()
        self.histogram_summary = P.HistogramSummary()

    def construct(self, image_tensor):
        self.image_summary("image", image_tensor)
        self.tensor_summary("tensor", image_tensor)
        self.histogram_summary("histogram", image_tensor)
        scalar = image_tensor[0][0][0][0]
        self.scalar_summary("scalar", scalar)
        return scalar


def train_summary_record(test_writer, steps):
    """Train and record summary."""
    net = SummaryNet()
    out_me_dict = {}
    for i in range(0, steps):
        image_tensor = Tensor(np.array([[[[i]]]]).astype(np.float32))
        out_put = net(image_tensor)
        test_writer.record(i)
        out_me_dict[i] = out_put.asnumpy()
    return out_me_dict


class TestCpuSummary:
    """Test cpu summary."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu_training
    @pytest.mark.env_onecard
    def test_summary_step2_summary_record1(self):
        """Test record 10 step summary."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            steps = 2
            with SummaryRecord(tmp_dir) as test_writer:
                train_summary_record(test_writer, steps=steps)

                file_name = os.path.realpath(test_writer.full_file_name)
            with SummaryReader(file_name) as summary_writer:
                for _ in range(steps):
                    event = summary_writer.read_event()
                    tags = set(value.tag for value in event.summary.value)
                    assert tags == {'tensor', 'histogram', 'scalar', 'image'}
