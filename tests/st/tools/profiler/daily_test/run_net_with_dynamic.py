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

import numpy as np
import argparse

from mindspore import nn, Tensor
import mindspore as ms
import mindspore.log as logger
from mindspore.profiler import ProfilerLevel, ProfilerActivity, AicoreMetrics, ExportType
from mindspore.profiler.profiler import analyse


class ReluNet(nn.Cell):
    """net"""
    def __init__(self):
        """init"""
        super().__init__()
        self.relu = nn.ReLU()

    def construct(self, x):
        """construct"""
        x = self.relu(x)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore Example for dynamic shape relu net')
    parser.add_argument("--output_path", type=str, default="./profiler_output")
    args = parser.parse_args()
    # pylint: disable=W0212
    experimental_config = ms.profiler._ExperimentalConfig(profiler_level=ProfilerLevel.Level2,
                                                          aic_metrics=AicoreMetrics.PipeUtilization,
                                                          l2_cache=True, mstx=False, data_simplification=False,
                                                          export_type=[ExportType.Text], mstx_domain_include=[],
                                                          mstx_domain_exclude=[])

    profiler = ms.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU], with_stack=True,
                                   profile_memory=True, data_process=True, parallel_strategy=True, hbm_ddr=True,
                                   pcie=True, on_trace_ready=ms.profiler.tensorboard_trace_handler(
            dir_name=args.output_path, worker_name="profs", analyse_flag=False, async_mode=False),
                                   experimental_config=experimental_config)

    net = ReluNet()
    input_dyn = Tensor(shape=[3, None], dtype=ms.float32)
    net.set_inputs(input_dyn)
    input1 = Tensor(np.random.random([3, 10]), dtype=ms.float32)
    profiler.start()
    output = net(input1)
    profiler.stop()
    analyse(profiler_path=args.output_path)
    logger.info(output)
