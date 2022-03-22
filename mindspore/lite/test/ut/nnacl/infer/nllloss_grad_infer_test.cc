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
#include "nnacl/infer/nllloss_grad_infer.h"

namespace mindspore {
class TestNLLLossGradInfer : public mindspore::CommonTest {
 public:
  TestNLLLossGradInfer() {}
};

void NLLLossGradInferInitArgs(std::vector<TensorC *> *inputs, std::vector<TensorC *> *outputs,
                              ReductionType reduction_type) {
  auto *logits = new TensorC;
  logits->shape_size_ = 2;
  logits->shape_[0] = 3;
  logits->shape_[1] = 5;
  inputs->push_back(logits);

  auto *loss_grad = new TensorC;
  if (reduction_type == Reduction_None) {
    loss_grad->shape_size_ = 1;
    loss_grad->shape_[0] = 3;
  } else {
    loss_grad->shape_size_ = 0;
  }
  inputs->push_back(loss_grad);

  auto *labels = new TensorC;
  labels->shape_size_ = 1;
  labels->shape_[0] = 3;
  inputs->push_back(labels);

  auto *weight = new TensorC;
  weight->shape_size_ = 1;
  weight->shape_[0] = 5;
  inputs->push_back(weight);

  auto *total_weight = new TensorC;
  total_weight->shape_size_ = 0;
  inputs->push_back(total_weight);

  auto *logits_grad = new TensorC;
  outputs->push_back(logits_grad);
}

void CheckResults(int ret, const NLLLossParameter *param, const std::vector<TensorC *> &outputs) {
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 2);
  ASSERT_EQ(outputs[0]->shape_[0], 3);
  ASSERT_EQ(outputs[0]->shape_[1], 5);
}

void NLLLossGradInferReleaseResources(NLLLossParameter *param, std::vector<TensorC *> inputs,
                                      std::vector<TensorC *> outputs) {
  delete param;
  for (auto t : inputs) delete t;
  for (auto t : outputs) delete t;
}

TEST_F(TestNLLLossGradInfer, ReductionNone) {
  std::vector<TensorC *> inputs;
  std::vector<TensorC *> outputs;
  NLLLossGradInferInitArgs(&inputs, &outputs, Reduction_None);
  auto *param = new NLLLossParameter;
  param->reduction_type_ = Reduction_None;
  int ret = NLLLossGradInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                  reinterpret_cast<OpParameter *>(param));
  CheckResults(ret, param, outputs);
  NLLLossGradInferReleaseResources(param, inputs, outputs);
}

TEST_F(TestNLLLossGradInfer, ReductionSum) {
  std::vector<TensorC *> inputs;
  std::vector<TensorC *> outputs;
  NLLLossGradInferInitArgs(&inputs, &outputs, Reduction_Sum);
  auto *param = new NLLLossParameter;
  param->reduction_type_ = Reduction_Sum;
  int ret = NLLLossGradInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                  reinterpret_cast<OpParameter *>(param));
  CheckResults(ret, param, outputs);
  NLLLossGradInferReleaseResources(param, inputs, outputs);
}

TEST_F(TestNLLLossGradInfer, ReductionMean) {
  std::vector<TensorC *> inputs;
  std::vector<TensorC *> outputs;
  NLLLossGradInferInitArgs(&inputs, &outputs, Reduction_Mean);
  auto *param = new NLLLossParameter;
  param->reduction_type_ = Reduction_Mean;
  int ret = NLLLossGradInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                  reinterpret_cast<OpParameter *>(param));
  CheckResults(ret, param, outputs);
  NLLLossGradInferReleaseResources(param, inputs, outputs);
}
}  // namespace mindspore
