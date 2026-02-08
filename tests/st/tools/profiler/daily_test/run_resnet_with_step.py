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
"""resnet with step."""
import argparse

import mindspore
from mindspore.nn.optim.momentum import Momentum
from mindspore import nn
from tests.st.tools.profiler.daily_test.resnet50 import create_resnet50, create_dataset, Config
from tests.st.tools.profiler.daily_test.profiler_check import StopAtStepNew


def train_process(data_directory, prof_dir):
    """Train ResNet50 and profile selected steps via callback."""
    cfg = Config()
    profiler_cb = StopAtStepNew(start_profile=True, profile_memory=cfg.profile_memory,
                                activities=cfg.activities, with_stack=cfg.with_stack,
                                data_process=cfg.data_process, parallel_strategy=cfg.parallel_strategy,
                                hbm_ddr=cfg.hbm_ddr, pcie=cfg.pcie, dir_name=prof_dir, worker_name=cfg.worker_name,
                                analyse_flag=cfg.analyse_flag, async_mode=cfg.async_mode, wait=3,
                                warmup=cfg.warmup, active=5,
                                repeat=cfg.repeat, skip_first=cfg.skip_first, profiler_level=cfg.profiler_level,
                                mstx=cfg.mstx, data_simplification=cfg.data_simplification,
                                aic_metrics=cfg.aicore_metrics, l2_cache=cfg.l2_cache,
                                export_type=cfg.export_type, host_sys=cfg.host_sys, record_shapes=cfg.record_shapes,
                                mstx_domain_include=cfg.mstx_domain_include,
                                mstx_domain_exclude=cfg.mstx_domain_exclude)
    net = create_resnet50()
    dataset = create_dataset(data_directory)

    optim = Momentum(net.get_parameters(), 0.01, 0.9)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net.set_train()
    model = mindspore.train.Model(net, loss_fn=loss_fn, optimizer=optim)
    model.train(1, dataset, callbacks=profiler_cb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ResNet.")
    parser.add_argument("--prof_dir", type=str, help="Path to the prof directory")
    data_dir = "/home/workspace/mindspore_dataset/cifar-10-batches-bin"
    args = parser.parse_args()
    train_process(data_dir, args.prof_dir)
