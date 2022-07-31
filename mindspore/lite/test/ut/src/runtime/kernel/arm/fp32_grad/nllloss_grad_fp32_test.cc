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

#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/src/litert/kernel/cpu/fp32_grad/nllloss_grad.h"
#include "src/litert/kernel_registry.h"
#include "src/litert/kernel_exec.h"
#include "src/litert/tensor_category.h"

namespace mindspore {
class TestNLLLossGradFp32 : public mindspore::CommonTest {
 public:
  TestNLLLossGradFp32() {}
};

void NLLLossGradInitArgs(std::vector<lite::Tensor *> *inputs, std::vector<lite::Tensor *> *outputs,
                         const float *loss_grad_ptr, const std::vector<int> &loss_grad_shape) {
  float logits_array[15] = {-1.3739, -2.2700, -3.2333, -2.4589, -0.6566, -1.2156, -2.6026, -1.2200,
                            -1.8731, -1.7119, -0.7130, -3.3672, -1.5368, -1.8289, -2.3058};
  int labels_array[3] = {1, 0, 4};
  float weight_array[5] = {0.2, 0.3, 0.1, 0.15, 0.25};
  float total_weight_array[1] = {0.75};
  std::vector<int> logits_shape = {3, 5};
  std::vector<int> labels_shape = {3};
  std::vector<int> weight_shape = {5};
  std::vector<int> total_weight_shape = {};

  auto *logits_t = new lite::Tensor(kNumberTypeFloat32, logits_shape, mindspore::NC, lite::Category::CONST_TENSOR);
  logits_t->MallocData();
  memcpy(logits_t->MutableData(), logits_array, sizeof(float) * logits_t->ElementsNum());
  inputs->push_back(logits_t);

  auto type = loss_grad_shape.empty() ? lite::Category::CONST_SCALAR : lite::Category::CONST_TENSOR;
  auto *loss_grad_t = new lite::Tensor(kNumberTypeFloat32, loss_grad_shape, mindspore::NC, type);
  loss_grad_t->MallocData();
  memcpy(loss_grad_t->MutableData(), loss_grad_ptr, sizeof(float) * loss_grad_t->ElementsNum());
  inputs->push_back(loss_grad_t);

  auto *labels_t = new lite::Tensor(kNumberTypeInt32, labels_shape, mindspore::NC, lite::Category::CONST_TENSOR);
  labels_t->MallocData();
  memcpy(labels_t->MutableData(), labels_array, sizeof(int) * labels_t->ElementsNum());
  inputs->push_back(labels_t);

  auto *weight_t = new lite::Tensor(kNumberTypeFloat32, weight_shape, mindspore::NC, lite::Category::CONST_TENSOR);
  weight_t->MallocData();
  memcpy(weight_t->MutableData(), weight_array, sizeof(float) * weight_t->ElementsNum());
  inputs->push_back(weight_t);

  auto *total_weight_t =
    new lite::Tensor(kNumberTypeFloat32, total_weight_shape, mindspore::NC, lite::Category::CONST_SCALAR);
  total_weight_t->MallocData();
  memcpy(total_weight_t->MutableData(), total_weight_array, sizeof(float) * weight_t->ElementsNum());
  inputs->push_back(total_weight_t);

  auto *logits_grad_t = new lite::Tensor(kNumberTypeFloat32, logits_shape, mindspore::NC, lite::Category::CONST_TENSOR);
  logits_grad_t->MallocData();
  outputs->push_back(logits_grad_t);
}

void NLLLossGradReleaseResources(lite::InnerContext *ctx, kernel::NLLLossGradCPUKernel *kernel, NLLLossParameter *param,
                                 std::vector<lite::Tensor *> inputs, std::vector<lite::Tensor *> outputs) {
  delete kernel;
  delete ctx;
  for (auto t : inputs) delete t;
  for (auto t : outputs) delete t;
}

TEST_F(TestNLLLossGradFp32, ReductionNone) {
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  float loss_grad[3] = {1.181, 0.74312, 1.07645};
  std::vector<int> loss_grad_shape = {3};
  NLLLossGradInitArgs(&inputs, &outputs, loss_grad, loss_grad_shape);

  auto *ctx = new lite::InnerContext;
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto *param = new NLLLossParameter;
  param->batch_ = 3;
  param->class_num_ = 5;
  param->reduction_type_ = Reduction_None;
  auto *kernel = new kernel::NLLLossGradCPUKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs, ctx);
  kernel->Prepare();
  kernel->Run();

  float expect_loss[15] = {0.0000, -0.35430002, 0.0000, 0.0000, 0.0000, -0.148624, 0.0000,    0.0000,
                           0.0000, 0.0000,      0.0000, 0.0000, 0.0000, 0.0000,    -0.2691125};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0]->MutableData()), expect_loss, 15, 0.0001));
  NLLLossGradReleaseResources(ctx, kernel, param, inputs, outputs);
}

TEST_F(TestNLLLossGradFp32, ReductionSum) {
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  float loss_grad[1] = {2.00057};
  std::vector<int> loss_grad_shape = {};
  NLLLossGradInitArgs(&inputs, &outputs, loss_grad, loss_grad_shape);

  auto *ctx = new lite::InnerContext;
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto *param = new NLLLossParameter;
  param->batch_ = 3;
  param->class_num_ = 5;
  param->reduction_type_ = Reduction_Sum;
  auto *kernel = new kernel::NLLLossGradCPUKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs, ctx);
  kernel->Prepare();
  kernel->Run();

  float expect_loss[15] = {0.0000, -0.600171, 0.0000, 0.0000, 0.0000, -0.40011403, 0.0000,    0.0000,
                           0.0000, 0.0000,    0.0000, 0.0000, 0.0000, 0.0000,      -0.5001425};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0]->MutableData()), expect_loss, 15, 0.0001));
  NLLLossGradReleaseResources(ctx, kernel, param, inputs, outputs);
}

TEST_F(TestNLLLossGradFp32, ReductionMean) {
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  float loss_grad[1] = {2.50076};
  std::vector<int> loss_grad_shape = {};
  NLLLossGradInitArgs(&inputs, &outputs, loss_grad, loss_grad_shape);

  auto *ctx = new lite::InnerContext;
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto *param = new NLLLossParameter;
  param->batch_ = 3;
  param->class_num_ = 5;
  param->reduction_type_ = Reduction_Mean;
  auto *kernel = new kernel::NLLLossGradCPUKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs, ctx);
  kernel->Prepare();
  kernel->Run();

  float expect_loss[15] = {0.0000, -1.0003041, 0.0000, 0.0000, 0.0000, -0.6668694, 0.0000,    0.0000,
                           0.0000, 0.0000,     0.0000, 0.0000, 0.0000, 0.0000,     -0.8335867};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0]->MutableData()), expect_loss, 15, 0.0001));
  NLLLossGradReleaseResources(ctx, kernel, param, inputs, outputs);
}
}  // namespace mindspore
