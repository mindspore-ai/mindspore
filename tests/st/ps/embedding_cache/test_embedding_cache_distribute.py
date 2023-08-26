# Copyright 2022 Huawei Technologies Co., Ltd
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

import os
import argparse
import mindspore.context as context
from mindspore.communication.management import get_group_size, init
from mindspore.context import ParallelMode
from src.dataset import create_dataset
from src.model import ModelExecutor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_embedding_cache_distribute")
    parser.add_argument("--device_target", type=str, default="Ascend")
    parser.add_argument("--sparse", type=int, default=0, help="Enable sparse or not")
    args, _ = parser.parse_known_args()
    device_target = args.device_target
    sparse = bool(args.sparse)

    context.set_context(mode=context.GRAPH_MODE, device_target=device_target)
    context.set_ps_context(enable_ps=True)
    init()

    full_batch = False
    if os.getenv("MS_ROLE") == "MS_WORKER":
        context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL,
                                          full_batch=True, gradients_mean=True,
                                          search_mode="dynamic_programming")
        full_batch = True

    dataset = create_dataset(batch_size=8, resize_height=32, resize_width=32, scale=30.0, full_batch=full_batch,
                             rank_size=get_group_size())
    executor = ModelExecutor(dataset=dataset, sparse=sparse, save_ckpt=True, vocab_size=32, embedding_size=8,
                             vocab_cache_size=50, in_channels=24576, out_channels=12, input_shape=[32, 3, 32, 32])
    executor.run_embedding_cache()
