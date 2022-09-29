/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common/common_test.h"
#include "nnacl/scatter_nd_parameter.h"
#include "nnacl/infer/scatter_nd_update_infer.h"

namespace mindspore {
class TestScatterNdAddInfer : public mindspore::CommonTest {
 public:
  TestScatterNdAddInfer() {}
};

void ScatterNdAddInferInitArgs(std::vector<TensorC *> *inputs, std::vector<TensorC *> *outputs) {
  auto *input_x = new TensorC;
  input_x->data_type_ = kNumberTypeFloat32;
  input_x->shape_size_ = 4;
  input_x->shape_[0] = 3;
  input_x->shape_[1] = 4;
  input_x->shape_[2] = 5;
  input_x->shape_[3] = 6;
  inputs->push_back(input_x);

  auto *indices = new TensorC;
  indices->data_type_ = kNumberTypeInt32;
  indices->shape_size_ = 3;
  indices->shape_[0] = 7;
  indices->shape_[1] = 8;
  indices->shape_[2] = 2;
  inputs->push_back(indices);

  auto *updates = new TensorC;
  updates->data_type_ = kNumberTypeFloat32;
  updates->shape_size_ = 4;
  updates->shape_[0] = 7;
  updates->shape_[1] = 8;
  updates->shape_[2] = 5;
  updates->shape_[3] = 6;
  inputs->push_back(updates);

  auto *output = new TensorC;
  outputs->push_back(output);
}

void ScatterNdAddInferReleaseResources(ScatterNDParameter *param, std::vector<TensorC *> inputs,
                                       std::vector<TensorC *> outputs) {
  delete param;
  for (auto t : inputs) delete t;
  for (auto t : outputs) delete t;
}

TEST_F(TestScatterNdAddInfer, FourDims) {
  std::vector<TensorC *> inputs;
  std::vector<TensorC *> outputs;
  ScatterNdAddInferInitArgs(&inputs, &outputs);
  auto *param = new ScatterNDParameter;
  int ret = ScatterNdUpdateInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                      reinterpret_cast<OpParameter *>(param));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 3);
  ASSERT_EQ(outputs[0]->shape_[1], 4);
  ASSERT_EQ(outputs[0]->shape_[2], 5);
  ASSERT_EQ(outputs[0]->shape_[3], 6);
  ScatterNdAddInferReleaseResources(param, inputs, outputs);
}
}  // namespace mindspore
