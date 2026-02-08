# Copyright 2024 Huawei Technologies Co., Ltd
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

import subprocess
import numpy as np

import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import context, ops
from mindspore.train import Model
from tests.mark_utils import arg_mark


class MyDataset:
    def __init__(self):
        self.data = [np.array(1), np.array(2), np.array(3), np.array(4)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class Net(nn.Cell):
    def construct(self, x):
        return ops.square(x)


def test_train_with_simple_dataset():
    """
    Feature: A basic test case to run network with model.build.
    Description: Network just square the input data.
    Expectation: train successfully and output log.
    """
    context.set_context(mode=context.GRAPH_MODE)
    dataset = ds.GeneratorDataset(MyDataset(), ["A"], shuffle=False)
    dataset = dataset.batch(2)

    net = Net()
    model = Model(net)
    model.build(dataset, epoch=1)
    model.train(1, dataset, dataset_sink_mode=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_dataset_iterator_vlog_flow():
    """
    Feature: Test vlog print in model.build process.
    Description: Run a simple network and check if output vlog contains specified log.
    Expectation: 'expected_output' exists in output vlog.
    """
    cmd = f"VLOG_v=1 pytest -s test_dataset_vlog.py::test_train_with_simple_dataset"
    s = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    out = s.stdout.read().decode("UTF-8")
    s.stdout.close()
    lines = out.split('\n')

    expected_output = ["Begin to init dataset in model.build()",
                       "Begin to check device number in model.build()",
                       "Begin to check parameter broadcast in model.build()",
                       "Begin to exec preprocess in model.build()",
                       "Begin to create DatasetHelper",
                       "Begin to connect network with dataset",
                       "Begin to warmup dataset in model.build()",
                       "Dataset Pipeline TreeAdapter Compile started",
                       "Dataset Pipeline TreeAdapter Compile finished",
                       "Dataset Pipeline launched",
                       "Begin waiting for dataset warmup in model.build()",
                       "Loading dataset and begin to push first batch into device",
                       "Loading dataset and push first batch into device successful",
                       "The dataset warmup was successful in model.build()",
                       "Begin to compile train network in model.build()",
                       "The model.build() which contains dataset warmup and network compile is success"]

    for output in expected_output:
        matched = False
        for line in lines:
            if line.find(output) > 0:
                matched = True
                break
        assert matched, f'`VLOG_v=1` expect `{output}` fail'
