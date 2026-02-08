# Copyright 2025 Huawei Technologies Co., Ltd
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
"""build net"""

import argparse
import numpy as np

from mindspore import nn, Tensor
from mindspore import ops as P
from mindspore.train import DatasetHelper, connect_network_with_dataset
import mindspore.dataset as ds
import mindspore as ms
from mindspore.common.initializer import One
from mindspore import log
from mindspore.profiler import ProfilerLevel, ProfilerActivity, AicoreMetrics, ExportType
from mindspore.profiler.profiler import analyse


def _exec_preprocess(network, is_train, dataset, dataset_sink_mode, sink_size=1, epoch_num=1, dataset_helper=None):
    """_exec_preprocess"""
    if dataset_helper is None:
        dataset_helper = DatasetHelper(
            dataset, dataset_sink_mode, sink_size, epoch_num)

    if dataset_sink_mode:
        network = connect_network_with_dataset(network, dataset_helper)

    network.set_train(is_train)

    return dataset_helper, network


def _eval_dataset_sink_process(network, valid_dataset):
    """_eval_dataset_sink_process"""
    dataset_helper, eval_network = _exec_preprocess(network, is_train=False, dataset=valid_dataset,
                                                    dataset_sink_mode=True)
    for inputs1, inputs2 in zip(dataset_helper, valid_dataset.create_dict_iterator()):
        outputs = eval_network(*inputs1)
        for elem1, (_, elem2) in zip(outputs, inputs2.items()):
            assert elem1.shape == elem2.shape


def dataset_generator():
    """dataset_generator"""
    for i in range(1, 10):
        yield (
            np.ones((32, i), dtype=np.float32), np.zeros(
                (32, i, i, 3), dtype=np.int32),
            np.ones((32,), dtype=np.float32),
            np.ones((32, i, 8), dtype=np.float32), np.ones((32, 8, 8), dtype=np.float32))


class Net(nn.Cell):
    """Net"""

    def __init__(self):
        super().__init__()
        self.relu = P.ReLU()

    def construct(self, x1, x2, x3, x4, x5):
        """construct"""
        x1 = self.relu(x1)
        x1 = self.relu(x1)
        x2 = self.relu(x2)
        x3 = self.relu(x3)
        x3 = self.relu(x3)
        x4 = self.relu(x4)
        x5 = self.relu(x5)
        return x1, x2, x3, x4, x5


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore Example for getnext-op')
    parser.add_argument("--output_path", type=str, default="./profiler_output")
    args = parser.parse_args()
    # pylint: disable=W0212
    experimental_config = (
        ms.profiler._ExperimentalConfig(profiler_level=ProfilerLevel.Level2,
                                        aic_metrics=AicoreMetrics.PipeUtilization, l2_cache=True, mstx=False,
                                        data_simplification=False, export_type=[ExportType.Text],
                                        mstx_domain_include=[], mstx_domain_exclude=[]))

    profiler = ms.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.NPU], with_stack=True, profile_memory=True,
        data_process=True, parallel_strategy=True, hbm_ddr=True, pcie=True,
        on_trace_ready=ms.profiler.tensorboard_trace_handler(dir_name=args.output_path, worker_name="profs",
                                                             analyse_flag=False, async_mode=False),
        experimental_config=experimental_config)

    networks = Net()
    datasets = ds.GeneratorDataset(
        dataset_generator, ["data1", "data2", "data3", "data4", "data5"])
    t0 = Tensor(dtype=ms.float32, shape=[32, None])
    t1 = Tensor(dtype=ms.int32, shape=[32, None, None, 3])
    t2 = Tensor(dtype=ms.float32, shape=[32], init=One())
    t3 = Tensor(dtype=ms.float32, shape=[32, None, 8])
    t4 = Tensor(dtype=ms.float32, shape=[32, 8, 8], init=One())
    networks.set_inputs(t0, t1, t2, t3, t4)
    profiler.start()
    _eval_dataset_sink_process(networks, datasets)
    profiler.stop()
    analyse(profiler_path=args.output_path)
    log.info("training success")
