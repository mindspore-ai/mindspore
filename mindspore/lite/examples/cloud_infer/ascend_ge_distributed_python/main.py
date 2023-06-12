# Copyright 2023 Huawei Technologies Co., Ltd
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
"""main.py"""

import argparse
import numpy as np
import mindspore_lite as mslite

# read args
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--device_id', type=int)
parser.add_argument('--rank_id', type=int)
parser.add_argument('--config_file', type=str)
args = parser.parse_args()

# init context, and set Ascend target
context = mslite.Context()
context.target = ["Ascend"]

# set distributed info
context.ascend.device_id = args.device_id
context.ascend.rank_id = args.rank_id
context.ascend.provider = "ge"

# create Model and build Model
model = mslite.Model()
model.build_from_file(
    args.model_path, mslite.ModelType.MINDIR, context, args.config_file)

# set model input as ones
inputs = model.get_inputs()
for input_i in inputs:
    input_i.set_data_from_numpy(np.ones(input_i.shape, dtype=np.float32))

# execute inference
outputs = model.predict(inputs)

# get output and print
for output in outputs:
    name = output.name.rstrip()
    data_size = output.data_size
    element_num = output.element_num
    print("tensor's name is:%s data size is:%s tensor elements num is:%s" %
          (name, data_size, element_num))
    data = output.get_data_to_numpy()
    data = data.flatten()
    print("output data is:", end=" ")
    for i in range(10):
        print(data[i], end=" ")
    print("")
