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
#include "mindspore/lite/nnacl/lsh_projection_parameter.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/src/lite_kernel.h"
#include "mindspore/lite/src/tensor.h"

namespace mindspore {

namespace {
constexpr int kSparseType = 1;
constexpr int kDenseType = 2;
}  // namespace

class TestLshProjectionFp32 : public mindspore::CommonTest {
 public:
  TestLshProjectionFp32() {}
};

TEST_F(TestLshProjectionFp32, Dense1DInputs) {
  lite::Tensor in_tensor0(kNumberTypeFloat, {3, 2});
  lite::Tensor in_tensor1(kNumberTypeInt32, {5});
  lite::Tensor in_tensor2(kNumberTypeFloat, {5});
  lite::Tensor out_tensor(kNumberTypeInt32, {6});

  float input_data0[] = {0.123, 0.456, -0.321, 1.234, 5.678, -4.321};
  int32_t input_data1[] = {12345, 54321, 67890, 9876, -12345678};
  float input_data2[] = {1.0, 1.0, 1.0, 1.0, 1.0};
  int32_t output_data[6] = {0};
  in_tensor0.set_data(input_data0);
  in_tensor1.set_data(input_data1);
  in_tensor2.set_data(input_data2);
  out_tensor.set_data(output_data);

  std::vector<lite::Tensor *> inputs = {&in_tensor0, &in_tensor1, &in_tensor2};
  std::vector<lite::Tensor *> outputs = {&out_tensor};

  LshProjectionParameter parameter = {};
  parameter.lsh_type_ = kDenseType;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_LshProjection};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ctx->thread_num_ = 3;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(&parameter), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<int32_t> except_result = {0, 0, 0, 1, 0, 0};
  PrintData("output data", output_data, 6);
  ASSERT_EQ(0, CompareOutputData(output_data, except_result.data(), 6, 0.000001));

  in_tensor0.set_data(nullptr);
  in_tensor1.set_data(nullptr);
  out_tensor.set_data(nullptr);
}

TEST_F(TestLshProjectionFp32, Sparse1DInputs) {
  lite::Tensor in_tensor0(kNumberTypeFloat, {3, 2});
  lite::Tensor in_tensor1(kNumberTypeInt32, {5});
  lite::Tensor out_tensor(kNumberTypeInt32, {3});

  float input_data0[] = {0.123, 0.456, -0.321, 1.234, 5.678, -4.321};
  int32_t input_data1[] = {12345, 54321, 67890, 9876, -12345678};
  int32_t output_data[3] = {0};
  in_tensor0.set_data(input_data0);
  in_tensor1.set_data(input_data1);
  out_tensor.set_data(output_data);

  std::vector<lite::Tensor *> inputs = {&in_tensor0, &in_tensor1};
  std::vector<lite::Tensor *> outputs = {&out_tensor};

  LshProjectionParameter parameter = {};
  parameter.lsh_type_ = kSparseType;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_LshProjection};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(&parameter), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<int32_t> except_result = {0, 5, 8};
  PrintData("output data", output_data, 3);
  ASSERT_EQ(0, CompareOutputData(output_data, except_result.data(), 3, 0.000001));

  in_tensor0.set_data(nullptr);
  in_tensor1.set_data(nullptr);
  out_tensor.set_data(nullptr);
}

TEST_F(TestLshProjectionFp32, Sparse3DInputs) {
  lite::Tensor in_tensor0(kNumberTypeFloat, {3, 2});
  lite::Tensor in_tensor1(kNumberTypeInt32, {5, 2, 2});
  lite::Tensor in_tensor2(kNumberTypeFloat, {5});
  lite::Tensor out_tensor(kNumberTypeInt32, {3});

  float input_data0[] = {0.123, 0.456, -0.321, 1.234, 5.678, -4.321};
  int32_t input_data1[] = {1234, 2345, 3456, 1234, 4567, 5678, 6789, 4567, 7891, 8912,
                           9123, 7890, -987, -876, -765, -987, -543, -432, -321, -543};
  float input_data2[] = {0.12, 0.34, 0.56, 0.67, 0.78};
  int32_t output_data[3] = {0};
  in_tensor0.set_data(input_data0);
  in_tensor1.set_data(input_data1);
  in_tensor2.set_data(input_data2);
  out_tensor.set_data(output_data);

  std::vector<lite::Tensor *> inputs = {&in_tensor0, &in_tensor1, &in_tensor2};
  std::vector<lite::Tensor *> outputs = {&out_tensor};

  LshProjectionParameter parameter = {};
  parameter.lsh_type_ = kSparseType;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_LshProjection};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ctx->thread_num_ = 3;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(&parameter), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<int32_t> except_result = {2, 5, 9};
  PrintData("output data", output_data, 3);
  ASSERT_EQ(0, CompareOutputData(output_data, except_result.data(), 3, 0.000001));

  in_tensor0.set_data(nullptr);
  in_tensor1.set_data(nullptr);
  out_tensor.set_data(nullptr);
}
}  // namespace mindspore
