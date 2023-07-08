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
parser.add_argument('--moadel_path1', type=str)
parser.add_argument('--moadel_path2', type=str)
parser.add_argument('--config_file', type=str)
args = parser.parse_args()

# init context, and set Ascend target
model_group_context = mslite.Context()
model_group_context.target = ["Ascend"]

# init model group
model_group = mslite.ModelGroup()
model_group.add_model([args.moadel_path1, args.moadel_path2])
model_group.cal_max_size_of_workspace(mslite.ModelType.MINDIR, model_group_context)

# 1: create Model and build Model
model_context_1 = mslite.Context()
model_context_1.target = ["Ascend"]
model1 = mslite.Model()
model1.build_from_file(args.model_path, mslite.ModelType.MINDIR, model_context_1)

# 2: create Model and build Model
model_context_2 = mslite.Context()
model_context_2.target = ["Ascend"]
model2 = mslite.Model()
model2.build_from_file(args.model_path, mslite.ModelType.MINDIR, model_context_2)

# for model1
# set model input as ones
inputs1 = model1.get_inputs()
for input_i in inputs1:
    input_i.set_data_from_numpy(np.ones(input_i.shape, dtype=np.float32))

# execute inference
outputs1 = model1.predict(inputs1)

# get output and print
for output in outputs1:
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
print("model 2 predict success.")

# for model 2
# set model input as ones
inputs2 = model2.get_inputs()
for input_i in inputs2:
    input_i.set_data_from_numpy(np.ones(input_i.shape, dtype=np.float32))

# execute inference
outputs2 = model2.predict(inputs2)

# get output and print
for output in outputs2:
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
print("model 2 predict success.")

print("=========== success ===========")
