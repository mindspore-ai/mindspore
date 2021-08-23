/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <memory>
#include "common/common_test.h"
#include "nnacl/fp32/ragged_range_fp32.h"
#include "src/tensor.h"
#include "src/lite_kernel.h"
#include "src/kernel_registry.h"

namespace mindspore {
class TestRaggedRangeFp32 : public mindspore::CommonTest {
 public:
  TestRaggedRangeFp32() {}
};

TEST_F(TestRaggedRangeFp32, 001) {
  lite::Tensor in_tensor0(kNumberTypeFloat32, {1});
  lite::Tensor in_tensor1(kNumberTypeFloat32, {1});
  lite::Tensor in_tensor2(kNumberTypeFloat32, {1});
  lite::Tensor out_tensor0(kNumberTypeFloat32, {2});
  lite::Tensor out_tensor1(kNumberTypeInt32, {5});

  float input_data0[] = {0};
  float input_data1[] = {5};
  float input_data2[] = {1};
  int output_data0[2];
  float output_data1[5];
  in_tensor0.set_data(input_data0);
  in_tensor1.set_data(input_data1);
  in_tensor2.set_data(input_data2);
  out_tensor0.set_data(output_data0);
  out_tensor1.set_data(output_data1);
  std::vector<lite::Tensor *> inputs = {&in_tensor0, &in_tensor1, &in_tensor2};
  std::vector<lite::Tensor *> outputs = {&out_tensor0, &out_tensor1};

  RaggedRangeParameter param = {{}, 1, true, true, true};
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_RaggedRange};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(&param), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);

  auto ret = kernel->Init();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  int expect0[] = {0, 5};
  float expect1[] = {0, 1, 2, 3, 4};
  EXPECT_EQ(output_data0[0], expect0[0]);
  EXPECT_EQ(output_data0[1], expect0[1]);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(output_data1[i], expect1[i]);
  }

  in_tensor0.set_data(nullptr);
  in_tensor1.set_data(nullptr);
  in_tensor2.set_data(nullptr);
  out_tensor0.set_data(nullptr);
  out_tensor1.set_data(nullptr);
  delete kernel;
}

TEST_F(TestRaggedRangeFp32, 002) {
  lite::Tensor in_tensor0(kNumberTypeFloat32, {4});
  lite::Tensor in_tensor1(kNumberTypeFloat32, {4});
  lite::Tensor in_tensor2(kNumberTypeFloat32, {4});
  lite::Tensor out_tensor0(kNumberTypeFloat32, {2});
  lite::Tensor out_tensor1(kNumberTypeInt32, {5});

  float input_data0[] = {0, 1, 7, 3};
  float input_data1[] = {3, 8, 4, 4};
  float input_data2[] = {1, 2, -1, 1};
  int output_data0[5];
  float output_data1[11];
  in_tensor0.set_data(input_data0);
  in_tensor1.set_data(input_data1);
  in_tensor2.set_data(input_data2);
  out_tensor0.set_data(output_data0);
  out_tensor1.set_data(output_data1);
  std::vector<lite::Tensor *> inputs = {&in_tensor0, &in_tensor1, &in_tensor2};
  std::vector<lite::Tensor *> outputs = {&out_tensor0, &out_tensor1};

  RaggedRangeParameter param = {{}, 4, false, false, false};
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_RaggedRange};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(&param), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);

  auto ret = kernel->Init();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  int expect0[] = {0, 3, 7, 10, 11};
  float expect1[] = {0, 1, 2, 1, 3, 5, 7, 7, 6, 5, 3};
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(output_data0[i], expect0[i]);
  }
  for (int i = 0; i < 11; ++i) {
    EXPECT_EQ(output_data1[i], expect1[i]);
  }

  in_tensor0.set_data(nullptr);
  in_tensor1.set_data(nullptr);
  in_tensor2.set_data(nullptr);
  out_tensor0.set_data(nullptr);
  out_tensor1.set_data(nullptr);
  delete kernel;
}
}  // namespace mindspore
