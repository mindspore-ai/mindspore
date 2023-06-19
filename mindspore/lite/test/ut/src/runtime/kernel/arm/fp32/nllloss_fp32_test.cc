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
#include "src/executor/kernel_exec.h"
#include "src/litert/tensor_category.h"
#include "nnacl/nllloss_parameter.h"
#include "nnacl/nnacl_manager.h"

namespace mindspore {
class TestNLLLossFp32 : public mindspore::CommonTest {
 public:
  TestNLLLossFp32() {}
};

void NLLLossInitArgs(std::vector<lite::Tensor *> *inputs, std::vector<lite::Tensor *> *outputs,
                     const std::vector<int> &loss_shape) {
  float logits_array[15] = {-1.3739, -2.2700, -3.2333, -2.4589, -0.6566, -1.2156, -2.6026, -1.2200,
                            -1.8731, -1.7119, -0.7130, -3.3672, -1.5368, -1.8289, -2.3058};
  int labels_array[3] = {1, 0, 4};
  float weight_array[5] = {0.2, 0.3, 0.1, 0.15, 0.25};
  std::vector<int> logits_shape = {3, 5};
  std::vector<int> labels_shape = {3};
  std::vector<int> weight_shape = {5};
  std::vector<int> total_weight_shape = {};

  auto *logits_t = new lite::Tensor(kNumberTypeFloat32, logits_shape, mindspore::NC, lite::Category::CONST_TENSOR);
  logits_t->MallocData();
  memcpy(logits_t->MutableData(), logits_array, sizeof(float) * logits_t->ElementsNum());
  inputs->push_back(logits_t);

  auto *labels_t = new lite::Tensor(kNumberTypeInt32, labels_shape, mindspore::NC, lite::Category::CONST_TENSOR);
  labels_t->MallocData();
  memcpy(labels_t->MutableData(), labels_array, sizeof(int) * labels_t->ElementsNum());
  inputs->push_back(labels_t);

  auto *weight_t = new lite::Tensor(kNumberTypeFloat32, weight_shape, mindspore::NC, lite::Category::CONST_TENSOR);
  weight_t->MallocData();
  memcpy(weight_t->MutableData(), weight_array, sizeof(float) * weight_t->ElementsNum());
  inputs->push_back(weight_t);

  auto type = loss_shape.empty() ? lite::Category::CONST_SCALAR : lite::Category::CONST_TENSOR;
  auto *loss_t = new lite::Tensor(kNumberTypeFloat32, loss_shape, mindspore::NC, type);
  loss_t->MallocData();
  outputs->push_back(loss_t);

  auto *total_weight_t =
    new lite::Tensor(kNumberTypeFloat32, total_weight_shape, mindspore::NC, lite::Category::CONST_SCALAR);
  total_weight_t->MallocData();
  outputs->push_back(total_weight_t);
}

void NLLLossReleaseResources(lite::InnerContext *ctx, kernel::LiteKernel *kernel, std::vector<lite::Tensor *> inputs,
                             std::vector<lite::Tensor *> outputs) {
  delete kernel;
  delete ctx;
  for (auto t : inputs) delete t;
  for (auto t : outputs) delete t;
}

TEST_F(TestNLLLossFp32, ReductionNone) {
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  std::vector<int> loss_shape = {3};
  NLLLossInitArgs(&inputs, &outputs, loss_shape);

  auto *ctx = new lite::InnerContext;
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto *param = new NLLLossParameter;
  param->op_parameter_.thread_num_ = ctx->thread_num_;
  param->op_parameter_.type_ = schema::PrimitiveType_NLLLoss;
  param->reduction_type_ = Reduction_None;

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, NHWC, param->op_parameter_.type_};
  auto kernel = nnacl::NNACLKernelRegistry(&param->op_parameter_, inputs, outputs, ctx, desc);
  ASSERT_NE(kernel, nullptr);

  kernel->Prepare();
  kernel->Run();

  float expect_loss[3] = {0.681, 0.24312, 0.57645};
  float expect_total_weight[1] = {0.75};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0]->MutableData()), expect_loss, 3, 0.0001));
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[1]->MutableData()), expect_total_weight, 1, 0.0001));
  NLLLossReleaseResources(ctx, kernel, inputs, outputs);
}

TEST_F(TestNLLLossFp32, ReductionSum) {
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  std::vector<int> loss_shape = {};
  NLLLossInitArgs(&inputs, &outputs, loss_shape);

  auto *ctx = new lite::InnerContext;
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto *param = new NLLLossParameter;
  param->op_parameter_.thread_num_ = ctx->thread_num_;
  param->op_parameter_.type_ = schema::PrimitiveType_NLLLoss;
  param->reduction_type_ = Reduction_Sum;

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, NHWC, param->op_parameter_.type_};
  auto kernel = nnacl::NNACLKernelRegistry(&param->op_parameter_, inputs, outputs, ctx, desc);
  ASSERT_NE(kernel, nullptr);

  kernel->Prepare();
  kernel->Run();

  float expect_loss[1] = {1.50057};
  float expect_total_weight[1] = {0.75};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0]->MutableData()), expect_loss, 1, 0.0001));
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[1]->MutableData()), expect_total_weight, 1, 0.0001));
  NLLLossReleaseResources(ctx, kernel, inputs, outputs);
}

TEST_F(TestNLLLossFp32, ReductionMean) {
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  std::vector<int> loss_shape = {};
  NLLLossInitArgs(&inputs, &outputs, loss_shape);

  auto *ctx = new lite::InnerContext;
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto *param = new NLLLossParameter;
  param->op_parameter_.thread_num_ = ctx->thread_num_;
  param->op_parameter_.type_ = schema::PrimitiveType_NLLLoss;
  param->reduction_type_ = Reduction_Mean;

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, NHWC, param->op_parameter_.type_};
  auto kernel = nnacl::NNACLKernelRegistry(&param->op_parameter_, inputs, outputs, ctx, desc);
  ASSERT_NE(kernel, nullptr);

  kernel->Prepare();
  kernel->Run();

  float expect_loss[1] = {2.00076};
  float expect_total_weight[1] = {0.75};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0]->MutableData()), expect_loss, 1, 0.0001));
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[1]->MutableData()), expect_total_weight, 1, 0.0001));
  NLLLossReleaseResources(ctx, kernel, inputs, outputs);
}
}  // namespace mindspore
