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

import mindspore
import numpy as np
from mindspore import log as logger
from mindspore import nn, Tensor
from mindspore import ops
from mindspore.profiler import ProfilerLevel, ProfilerActivity, AicoreMetrics, ExportType
from mindspore.profiler.profiler import analyse


class AdaptiveAvgPool3DNet(nn.Cell):
    """init net"""

    def __init__(self, output_size):
        super().__init__()
        self.output_size_ = output_size
        self.adaptive_avg_pool_3d = ops.AdaptiveAvgPool3D(self.output_size_)

    def construct(self, x_):
        """construct"""
        return self.adaptive_avg_pool_3d(x_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore Example for adaptive_avg_pool_3d')
    parser.add_argument("--output_path", type=str, default="./profiler_output")
    args = parser.parse_args()
    # pylint: disable=W0212
    experimental_config = mindspore.profiler._ExperimentalConfig(profiler_level=ProfilerLevel.Level2,
                                                                 aic_metrics=AicoreMetrics.PipeUtilization,
                                                                 l2_cache=True, mstx=False, data_simplification=False,
                                                                 export_type=[ExportType.Text], mstx_domain_include=[],
                                                                 mstx_domain_exclude=[])

    profiler = mindspore.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU], with_stack=True,
                                          profile_memory=True, data_process=True, parallel_strategy=True, hbm_ddr=True,
                                          pcie=True, on_trace_ready=mindspore.profiler.tensorboard_trace_handler(
            dir_name=args.output_path, worker_name="profs", analyse_flag=False, async_mode=False),
                                          experimental_config=experimental_config)

    input_x_val = np.zeros((1, 1, 2, 2, 2))
    input_x_val[:, :, 0, :, :] += 1
    input_x = Tensor(input_x_val, mindspore.float32)
    adaptive_avg_pool_3d = ops.AdaptiveAvgPool3D(output_size=(1, 1, 1))

    profiler.start()
    output = adaptive_avg_pool_3d(input_x)

    profiler.stop()
    analyse(profiler_path=args.output_path)

    logger.info(output)
