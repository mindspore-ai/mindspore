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
args = parser.parse_args()

# init context, and set GPU target
context = mslite.Context()
context.target = ["gpu"]

# set distributed info
# as rank id is set by mpi automatically and visible device id is set in env, we use rank id as device id
context.gpu.device_id = context.gpu.rank_id
context.gpu.provider = "tensorrt"

# simply adding the rank id before extensions to distinguish each model from each rank
model_path = args.model_path.replace(
    ".mindir", "{}.mindir".format(context.gpu.rank_id))

# create Model and build Model
model = mslite.Model()
model.build_from_file(model_path, mslite.ModelType.MINDIR, context)

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
