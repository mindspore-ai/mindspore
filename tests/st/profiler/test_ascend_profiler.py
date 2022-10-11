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
"""Test ascend profiling."""
import glob
import tempfile
import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.dataset as ds
from mindspore import Profiler
from mindspore import Model
from tests.security_utils import security_off_wrap


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.Add()

    def construct(self, x_, y_):
        return self.add(x_, y_)


x = np.random.randn(1, 3, 3, 4).astype(np.float32)
y = np.random.randn(1, 3, 3, 4).astype(np.float32)


class NetWork(nn.Cell):
    def __init__(self):
        super(NetWork, self).__init__()
        self.unique = P.Unique()
        self.shape = P.TensorShape()
        self.reshape = P.Reshape()
        self.add = P.Add()

    def construct(self, a, b):
        val = self.add(a, b)
        size = self.shape(val)
        res = self.reshape(val, size)
        return res


def dataset_generator():
    for i in range(1, 10):
        yield (np.ones((32, 2 * i), dtype=np.float32), np.ones((32, 2 * i), dtype=np.float32))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_ascend_profiling():
    """Test ascend profiling"""
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory() as tmpdir:
        profiler = Profiler(output_path=tmpdir)
        add = Net()
        add(Tensor(x), Tensor(y))
        profiler.analyse()
        assert len(glob.glob(f"{tmpdir}/profiler*/*PROF*/device_*/data/Framework*")) == 4


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_ascend_profiling_l2():
    """
    Feature: profiling l2 cache data.
    Description: profiling l2 cache data, analyze performance issues.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory() as tmpdir:
        profiler = Profiler(output_path=tmpdir, l2_cache=True)
        add = Net()
        add(Tensor(x), Tensor(y))
        profiler.analyse()
        assert len(glob.glob(f"{tmpdir}/profiler*/*PROF*/device_*/data/l2_cache.data*")) >= 2


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_ascend_pynative_profiling():
    """
    Feature: Test the ascend pynative model profiling
    Description: Generate the Net op timeline
    Expectation: Timeline generated successfully
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory() as tmpdir:
        profiler = Profiler(output_path=tmpdir)
        add = Net()
        add(Tensor(x), Tensor(y))
        profiler.analyse()
        assert len(glob.glob(f"{tmpdir}/profiler*/output_timeline_data_*.txt")) == 1


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_shape():
    """
    Feature: Test the ascend dynamic shape model profiling
    Description: Generate the Net dynamic shape data.
    Expectation: Dynamic shape data generated successfully
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory() as tmpdir:
        network = NetWork()
        profiler = Profiler(output_path=tmpdir)
        dataset = ds.GeneratorDataset(dataset_generator, ["data1", "data2"])
        dataset.set_dynamic_columns(columns={"data1": [32, None], "data2": [32, None]})
        model = Model(network)
        model.train(1, dataset, dataset_sink_mode=True)
        profiler.analyse()
        assert len(glob.glob(f"{tmpdir}/profiler*/dynamic_shape_*.json")) == 1
