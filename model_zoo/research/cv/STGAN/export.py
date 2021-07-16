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
""" Model Export """
import numpy as np
from mindspore import context, Tensor
from mindspore.train.serialization import export
from src.models import STGANModel
from src.utils import get_args

if __name__ == '__main__':
    args = get_args("test")
    context.set_context(mode=context.GRAPH_MODE, device_id=args.device_id)
    model = STGANModel(args)
    model.netG.set_train(True)
    input_shp = [16, 3, 128, 128]
    input_shp_2 = [16, 4]
    input_array = Tensor(np.random.uniform(-1.0, 1.0, size=input_shp).astype(np.float32))
    input_array_2 = Tensor(np.random.uniform(-1.0, 1.0, size=input_shp_2).astype(np.float32))
    G_file = f"{args.file_name}_model"
    export(model.netG, input_array, input_array_2, file_name=G_file, file_format=args.file_format)
    