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
#include "nnacl/infer/nllloss_infer.h"

namespace mindspore {
class TestNLLLossInfer : public mindspore::CommonTest {
 public:
  TestNLLLossInfer() {}
};

void NLLLossInferInitArgs(std::vector<TensorC *> *inputs, std::vector<TensorC *> *outputs) {
  auto *logits = new TensorC;
  logits->shape_size_ = 2;
  logits->shape_[0] = 3;
  logits->shape_[1] = 5;
  inputs->push_back(logits);

  auto *labels = new TensorC;
  labels->shape_size_ = 1;
  labels->shape_[0] = 3;
  inputs->push_back(labels);

  auto *weight = new TensorC;
  weight->shape_size_ = 1;
  weight->shape_[0] = 5;
  inputs->push_back(weight);

  auto *loss = new TensorC;
  outputs->push_back(loss);
  auto *total_weight = new TensorC;
  outputs->push_back(total_weight);
}

void NLLLossInferReleaseResources(NLLLossParameter *param, std::vector<TensorC *> inputs,
                                  std::vector<TensorC *> outputs) {
  delete param;
  for (auto t : inputs) delete t;
  for (auto t : outputs) delete t;
}

TEST_F(TestNLLLossInfer, ReductionNone) {
  std::vector<TensorC *> inputs;
  std::vector<TensorC *> outputs;
  NLLLossInferInitArgs(&inputs, &outputs);
  auto *param = new NLLLossParameter;
  param->reduction_type_ = Reduction_None;
  int ret = NLLLossInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                              reinterpret_cast<OpParameter *>(param));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 1);
  ASSERT_EQ(outputs[0]->shape_[0], 3);
  ASSERT_EQ(outputs[1]->shape_size_, 0);
  NLLLossInferReleaseResources(param, inputs, outputs);
}

TEST_F(TestNLLLossInfer, ReductionSum) {
  std::vector<TensorC *> inputs;
  std::vector<TensorC *> outputs;
  NLLLossInferInitArgs(&inputs, &outputs);
  auto *param = new NLLLossParameter;
  param->reduction_type_ = Reduction_Sum;
  int ret = NLLLossInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                              reinterpret_cast<OpParameter *>(param));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 0);
  ASSERT_EQ(outputs[1]->shape_size_, 0);
  NLLLossInferReleaseResources(param, inputs, outputs);
}

TEST_F(TestNLLLossInfer, ReductionMean) {
  std::vector<TensorC *> inputs;
  std::vector<TensorC *> outputs;
  NLLLossInferInitArgs(&inputs, &outputs);
  auto *param = new NLLLossParameter;
  param->reduction_type_ = Reduction_Mean;
  int ret = NLLLossInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                              reinterpret_cast<OpParameter *>(param));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 0);
  ASSERT_EQ(outputs[1]->shape_size_, 0);
  NLLLossInferReleaseResources(param, inputs, outputs);
}
}  // namespace mindspore
