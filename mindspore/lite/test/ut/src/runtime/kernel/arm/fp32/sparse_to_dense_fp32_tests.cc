/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <iostream>
#include "schema/inner/model_generated.h"
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "nnacl/fp32/sparse_to_dense_fp32.h"
#include "mindspore/lite/src/litert/kernel_registry.h"
#include "mindspore/lite/src/litert/kernel_exec.h"
#include "mindspore/lite/src/tensor.h"

namespace mindspore {

class TestSparseToDenseFp32 : public mindspore::CommonTest {
 public:
  TestSparseToDenseFp32() {}
};

TEST_F(TestSparseToDenseFp32, SparseToDense_test1) {
  std::vector<lite::Tensor *> inputs;
  inputs.push_back(CreateTensor<int>(kNumberTypeInt32, {6, 2}, {0, 0, 1, 2, 2, 3, 3, 6, 4, 7, 5, 9}));
  inputs.push_back(CreateTensor<int>(kNumberTypeInt32, {2}, {6, 10}));
  inputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {1}, {1}));
  inputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {1}, {0}));

  std::vector<lite::Tensor *> outputs;
  outputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {6, 10}, {}));

  auto ctx = std::make_shared<lite::InnerContext>();
  ctx->thread_num_ = 3;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto op_param = static_cast<SparseToDenseParameter *>(malloc(sizeof(SparseToDenseParameter)));
  memset(op_param, 0, sizeof(SparseToDenseParameter));
  op_param->op_parameter_.thread_num_ = ctx->thread_num_;
  op_param->op_parameter_.type_ = schema::PrimitiveType_SpaceToDepth;
  op_param->validate_indices_ = false;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, NHWC, schema::PrimitiveType_SparseToDense};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(op_param), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<float> except_result = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
  ASSERT_EQ(0, CompareOutputData(static_cast<float *>(outputs[0]->data()), except_result.data(),
                                 outputs[0]->ElementsNum(), 0.000001));
  delete kernel;
  DestroyTensors(inputs);
  DestroyTensors(outputs);
}

TEST_F(TestSparseToDenseFp32, SparseToDense_test2) {
  std::vector<lite::Tensor *> inputs;
  inputs.push_back(CreateTensor<int>(kNumberTypeInt32, {6, 2}, {0, 0, 1, 2, 2, 3, 3, 6, 4, 7, 5, 9}));
  inputs.push_back(CreateTensor<int>(kNumberTypeInt32, {2}, {6, 10}));
  inputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {6}, {1, 2, 3, 4, 5, 6}));
  inputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {1}, {0}));

  std::vector<lite::Tensor *> outputs;
  outputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {6, 10}, {}));

  auto ctx = std::make_shared<lite::InnerContext>();
  ctx->thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto op_param = static_cast<SparseToDenseParameter *>(malloc(sizeof(SparseToDenseParameter)));
  memset(op_param, 0, sizeof(SparseToDenseParameter));
  op_param->op_parameter_.thread_num_ = ctx->thread_num_;
  op_param->op_parameter_.type_ = schema::PrimitiveType_SpaceToDepth;
  op_param->validate_indices_ = false;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, NHWC, schema::PrimitiveType_SparseToDense};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(op_param), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<float> except_result = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6};
  ASSERT_EQ(0, CompareOutputData(static_cast<float *>(outputs[0]->data()), except_result.data(),
                                 outputs[0]->ElementsNum(), 0.000001));
  delete kernel;
  DestroyTensors(inputs);
  DestroyTensors(outputs);
}

TEST_F(TestSparseToDenseFp32, SparseToDense_test3) {
  std::vector<lite::Tensor *> inputs;
  inputs.push_back(CreateTensor<int>(kNumberTypeInt32, {3}, {1, 3, 4}));
  inputs.push_back(CreateTensor<int>(kNumberTypeInt32, {2}, {1, 10}));
  inputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {1}, {1}));
  inputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {1}, {0}));

  std::vector<lite::Tensor *> outputs;
  outputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {1, 10}, {}));

  auto ctx = std::make_shared<lite::InnerContext>();
  ctx->thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto op_param = static_cast<SparseToDenseParameter *>(malloc(sizeof(SparseToDenseParameter)));
  memset(op_param, 0, sizeof(SparseToDenseParameter));
  op_param->op_parameter_.thread_num_ = ctx->thread_num_;
  op_param->op_parameter_.type_ = schema::PrimitiveType_SpaceToDepth;
  op_param->validate_indices_ = true;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, NHWC, schema::PrimitiveType_SparseToDense};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(op_param), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<float> except_result = {0, 1, 0, 1, 1, 0, 0, 0, 0, 0};
  PrintData("output data", static_cast<float *>(outputs[0]->data()), outputs[0]->ElementsNum());
  ASSERT_EQ(0, CompareOutputData(static_cast<float *>(outputs[0]->data()), except_result.data(),
                                 outputs[0]->ElementsNum(), 0.000001));
  delete kernel;
  DestroyTensors(inputs);
  DestroyTensors(outputs);
}

TEST_F(TestSparseToDenseFp32, SparseToDense_test4) {
  std::vector<lite::Tensor *> inputs;
  inputs.push_back(CreateTensor<int>(kNumberTypeInt32, {1}, {5}));
  inputs.push_back(CreateTensor<int>(kNumberTypeInt32, {1}, {10}));
  inputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {1}, {1}));
  inputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {1}, {0}));

  std::vector<lite::Tensor *> outputs;
  outputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {1, 10}, {}));

  auto ctx = std::make_shared<lite::InnerContext>();
  ctx->thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto op_param = static_cast<SparseToDenseParameter *>(malloc(sizeof(SparseToDenseParameter)));
  memset(op_param, 0, sizeof(SparseToDenseParameter));
  op_param->op_parameter_.thread_num_ = ctx->thread_num_;
  op_param->op_parameter_.type_ = schema::PrimitiveType_SpaceToDepth;
  op_param->validate_indices_ = true;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, NHWC, schema::PrimitiveType_SparseToDense};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(op_param), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<float> except_result = {0, 0, 0, 0, 0, 1, 0, 0, 0, 0};
  ASSERT_EQ(0, CompareOutputData(static_cast<float *>(outputs[0]->data()), except_result.data(),
                                 outputs[0]->ElementsNum(), 0.000001));
  delete kernel;
  DestroyTensors(inputs);
  DestroyTensors(outputs);
}

TEST_F(TestSparseToDenseFp32, SparseToDense_test5) {
  std::vector<lite::Tensor *> inputs;
  inputs.push_back(CreateTensor<int>(kNumberTypeInt32, {6, 2}, {0, 0, 1, 2, 2, 3, 3, 6, 4, 7, 5, 9}));
  inputs.push_back(CreateTensor<int>(kNumberTypeInt32, {2}, {6, 10}));
  inputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {6}, {1, 2, 3, 4, 5, 6}));
  inputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {1}, {0}));

  std::vector<lite::Tensor *> outputs;
  outputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {6, 10}, {}));

  auto ctx = std::make_shared<lite::InnerContext>();
  ctx->thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto op_param = static_cast<SparseToDenseParameter *>(malloc(sizeof(SparseToDenseParameter)));
  memset(op_param, 0, sizeof(SparseToDenseParameter));
  op_param->op_parameter_.thread_num_ = ctx->thread_num_;
  op_param->op_parameter_.type_ = schema::PrimitiveType_SpaceToDepth;
  op_param->validate_indices_ = true;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, NHWC, schema::PrimitiveType_SparseToDense};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(op_param), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<float> except_result = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6};
  PrintData("output data", static_cast<float *>(outputs[0]->data()), outputs[0]->ElementsNum());
  ASSERT_EQ(0, CompareOutputData(static_cast<float *>(outputs[0]->data()), except_result.data(),
                                 outputs[0]->ElementsNum(), 0.000001));
  delete kernel;
  DestroyTensors(inputs);
  DestroyTensors(outputs);
}
}  // namespace mindspore
