# Copyright 2020 Huawei Technologies Co., Ltd
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
import numpy as np

from mindspore import context, Tensor
from mindspore.train.serialization import export
from src.models import get_generator
from src.utils import get_args, load_ckpt

args = get_args("export")

context.set_context(mode=context.GRAPH_MODE, device_target=args.platform)

if __name__ == '__main__':
    G_A = get_generator(args)
    G_B = get_generator(args)
    # Use BatchNorm2d with batchsize=1, affine=False, training=True instead of InstanceNorm2d
    # Use real mean and varance rather than moving_men and moving_varance in BatchNorm2d
    G_A.set_train(True)
    G_B.set_train(True)
    load_ckpt(args, G_A, G_B)

    input_shp = [1, 3, args.image_size, args.image_size]
    input_array = Tensor(np.random.uniform(-1.0, 1.0, size=input_shp).astype(np.float32))
    G_A_file = f"{args.file_name}_BtoA"
    export(G_A, input_array, file_name=G_A_file, file_format=args.file_format)
    G_B_file = f"{args.file_name}_AtoB"
    export(G_B, input_array, file_name=G_B_file, file_format=args.file_format)
