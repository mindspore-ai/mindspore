/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "nnacl/infer/adam_weight_decay_infer.h"

namespace mindspore {
class AdamWeightDecayInfer : public mindspore::CommonTest {
 public:
  AdamWeightDecayInfer() {}
};

void AdamWeightDecayInferInitArgs(std::vector<TensorC *> *inputs, std::vector<TensorC *> *outputs) {
  const size_t inputs_size = 9;
  for (size_t i = 0; i < inputs_size; i++) {
    auto *input_x = new TensorC;
    input_x->shape_size_ = 1;
    input_x->shape_[0] = 1;
    inputs->push_back(input_x);
  }
  auto *output = new TensorC;
  outputs->push_back(output);
}

void AdamWeightDecayInferReleaseResources(OpParameter *param, std::vector<TensorC *> inputs,
                                          std::vector<TensorC *> outputs) {
  delete param;
  for (auto t : inputs) delete t;
  for (auto t : outputs) delete t;
}

TEST_F(AdamWeightDecayInfer, OneDim) {
  std::vector<TensorC *> inputs;
  std::vector<TensorC *> outputs;
  AdamWeightDecayInferInitArgs(&inputs, &outputs);
  auto *param = new OpParameter;
  int ret = AdamWeightDecayInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                      reinterpret_cast<OpParameter *>(param));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 1);
  ASSERT_EQ(outputs[0]->shape_[0], 1);
  AdamWeightDecayInferReleaseResources(param, inputs, outputs);
}
}  // namespace mindspore
