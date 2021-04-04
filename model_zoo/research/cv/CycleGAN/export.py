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

"""export file."""

import argparse
import numpy as np
from mindspore import context, Tensor
from mindspore.train.serialization import export
from src.models.cycle_gan import get_generator
from src.utils.args import get_args
from src.utils.tools import load_ckpt

model_args = get_args("export")
parser = argparse.ArgumentParser(description="openpose export")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--file_name", type=str, default="CycleGAN", help="output file name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR", help="file format")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=model_args.device_id)

if __name__ == '__main__':
    G_A = get_generator(model_args)
    G_B = get_generator(model_args)
    # Use BatchNorm2d with batchsize=1, affine=False, training=True instead of InstanceNorm2d
    # Use real mean and varance rather than moving_men and moving_varance in BatchNorm2d
    G_A.set_train(True)
    G_B.set_train(True)
    load_ckpt(model_args, G_A, G_B)

    input_shp = [args.batch_size, 3, model_args.image_size, model_args.image_size]
    input_array = Tensor(np.random.uniform(-1.0, 1.0, size=input_shp).astype(np.float32))
    G_A_file = f"{args.file_name}_BtoA"
    export(G_A, input_array, file_name=G_A_file, file_format=args.file_format)
    G_B_file = f"{args.file_name}_AtoB"
    export(G_B, input_array, file_name=G_B_file, file_format=args.file_format)
