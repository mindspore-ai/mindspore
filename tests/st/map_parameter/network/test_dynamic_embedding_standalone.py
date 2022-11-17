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
from src.dataset import create_dataset
from src.model import ModelExecutor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_dynamic_embedding_standalone")
    parser.add_argument("--device_target", type=str, default="GPU")
    args, _ = parser.parse_known_args()
    device_target = args.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=device_target)

    # Only run this test case in cuda11 environment.
    if not 'SAULT_ENV_TYPE' in os.environ or not "CUDA10" in os.environ['SAULT_ENV_TYPE']:
        dataset = create_dataset(resize_height=32, resize_width=32, scale=30.0)
        executor = ModelExecutor(dataset=dataset, sparse=True, in_channels=30720,
                                 out_channels=12, input_shape=[32, 3, 32, 32])
        executor.run_dynamic_embedding()
