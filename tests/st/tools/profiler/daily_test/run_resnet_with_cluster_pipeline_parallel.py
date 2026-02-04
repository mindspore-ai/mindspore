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
"""resnet with cluster."""
import json
import argparse

import mindspore
from mindspore.nn.optim.momentum import Momentum
from mindspore import nn, ops
from mindspore.profiler import mstx, tensorboard_trace_handler
from tests.st.tools.profiler.daily_test.resnet50 import create_resnet50, create_dataset, Config


def train_process(data_directory, prof_dir):
    """Train ResNet50."""
    cfg = Config()
    prof = mindspore.profiler.Profiler(profiler_level=cfg.profiler_level, activities=cfg.activities,
                                       aic_metrics=cfg.aicore_metrics,
                                       with_stack=cfg.with_stack, profile_memory=cfg.profile_memory,
                                       data_process=cfg.data_process,
                                       parallel_strategy=cfg.parallel_strategy, start_profile=cfg.start_profile,
                                       l2_cache=cfg.l2_cache, hbm_ddr=cfg.hbm_ddr, pcie=cfg.pcie,
                                       sync_enable=cfg.sync_enable, data_simplification=cfg.data_simplification,
                                       mstx=cfg.mstx,
                                       on_trace_ready=tensorboard_trace_handler(dir_name=prof_dir))
    if cfg.add_metadata:
        data = {"world_size": 2, "sequence_parallel": False, "hooks": "dajl"}
        prof.add_metadata("gsfa", "13")
        prof.add_metadata("q3123", "12%")
        prof.add_metadata_json("distribute_args", json.dumps(data))
    prof.start()
    net = create_resnet50()
    dataset = create_dataset(data_directory)

    optim = Momentum(net.get_parameters(), 0.01, 0.9)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    def forward_fn(data, label):
        """Forward pass with loss computation and MSTX range markers."""
        tl_id = mstx.range_start('Forward', None, domain="first_domain")
        out = net(data)
        loss_value = loss_fn(out, label)
        mstx.range_end(tl_id, domain="first_domain")
        return loss_value, out

    grad_fn = ops.value_and_grad(forward_fn, None, optim.parameters, has_aux=True)

    def train_step(data, label):
        """Compute gradients and update optimizer; marks MSTX ranges."""
        tl_id2 = mstx.range_start('Backward', None, domain="first2_domain")
        (loss, _), grads = grad_fn(data, label)
        mstx.range_end(tl_id2, domain="first2_domain")
        optim(grads)
        return loss

    size = dataset.get_dataset_size()
    net.set_train()

    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(data, label)
        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")
        prof.step()
        if batch == 10:
            break
    prof.stop()
    prof.analyse()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ResNet.")
    parser.add_argument("--prof_dir", type=str, help="Path to the prof directory")
    data_dir = "/home/workspace/mindspore_dataset/cifar-10-batches-bin"
    args = parser.parse_args()
    train_process(data_dir, args.prof_dir)
