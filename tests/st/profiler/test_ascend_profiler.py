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
import csv

import mindspore
import mindspore.context as context
import mindspore.dataset as ds
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Model
from mindspore import Profiler
from mindspore import Tensor
from mindspore.ops import operations as P
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
        self.shape = P.Shape()
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


class Net1(nn.Cell):
    def __init__(self):
        super(Net1, self).__init__()
        self.select = P.Select()
        self.reshape = P.Reshape()
        self.xlogy = P.Xlogy()
        self.tril = P.Tril(10)
        self.cast = P.Cast()
        self.expand_dims = P.ExpandDims()
        self.dense = nn.Dense(1, 3, activation='relu')
        self.flatten = nn.Flatten()

    def construct(self, a):
        shape = (2, 3)
        b = Tensor(np.array([4, 4, 5, 5, 6, 6]), mstype.float64)
        input_cond = Tensor([True, False, True, False, True, False])
        a = self.select(input_cond, a, b)
        a = self.reshape(a, shape)
        b = self.reshape(b, shape)
        a = self.tril(a)

        output = self.xlogy(a, b)
        output = self.expand_dims(output, -1)
        output = self.cast(output, mstype.float32)
        output = self.dense(output)
        output = self.flatten(output)
        return output


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_ascend_profiling():
    """Test ascend profiling"""
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory() as tmpdir:
        profiler = Profiler(output_path=tmpdir, l2_cache=True, data_simplification=False)
        add = Net()
        add(Tensor(x), Tensor(y))
        profiler.analyse()
        assert len(glob.glob(f"{tmpdir}/profiler*/*PROF*/mindstudio_profiler_output/op_summary*")) == 1
        assert len(glob.glob(f"{tmpdir}/profiler*/*PROF*/mindstudio_profiler_output/op_statistic*")) == 1
        assert len(glob.glob(f"{tmpdir}/profiler*/*PROF*/device_*/data/l2_cache.data*")) >= 2
    with tempfile.TemporaryDirectory() as tmpdir:
        profiler = Profiler(output_path=tmpdir, l2_cache=True, data_simplification=True)
        add = Net()
        add(Tensor(x), Tensor(y))
        profiler.analyse()
        assert glob.glob(f"{tmpdir}/profiler*/*PROF*/host/sqlite*") == []
        assert glob.glob(f"{tmpdir}/profiler*/*PROF*/mindstudio_profiler_output") == []
        assert glob.glob(f"{tmpdir}/profiler*/*PROF*/mindstudio_profiler_log") == []


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
        assert len(glob.glob(f"{tmpdir}/profiler*/ascend_timeline_display_*.json")) == 1


@pytest.mark.level1
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
        t0 = Tensor(dtype=mindspore.float32, shape=[32, None])
        t1 = Tensor(dtype=mindspore.float32, shape=[32, None])
        network.set_inputs(t0, t1)
        model = Model(network)
        model.train(1, dataset, dataset_sink_mode=True)
        profiler.analyse()
        assert len(glob.glob(f"{tmpdir}/profiler*/dynamic_shape_*.json")) == 1


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_collect_custom_aicpu():
    """
    Feature: Profiling can collect custom aicpu operators
    Description: Test profiling can collect custom aicpu operators on ascend
    Expectation: The file aicpu_intermediate_*.csv generated successfully and s1 == s2
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_level="O2")
    with tempfile.TemporaryDirectory() as tmpdir:
        profiler = Profiler(output_path=tmpdir)
        net = Net1()
        net(Tensor(np.random.random((6,)), mstype.float64))
        profiler.analyse()
        aicpu_intermediate_file_list = glob.glob(f"{tmpdir}/profiler/aicpu_intermediate_*.csv")
        assert len(aicpu_intermediate_file_list) == 1
        s1 = {'Cast', 'Select', 'Xlogy'}
        s2 = set()
        with open(aicpu_intermediate_file_list[0], 'r') as fr:
            reader = csv.DictReader(fr)
            for row in reader:
                s2.add(row.get('kernel_type'))
        assert s1 == s2
