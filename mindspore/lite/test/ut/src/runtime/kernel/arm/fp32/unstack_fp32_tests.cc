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
#include <memory>
#include "common/common_test.h"
#include "mindspore/lite/nnacl/base/unstack_base.h"
#include "mindspore/lite/src/kernel_registry.h"

namespace mindspore {
class TestUnstackFp32 : public mindspore::CommonTest {
 public:
  TestUnstackFp32() {}
};

TEST_F(TestUnstackFp32, Unstack) {
  lite::Tensor in_tensor(kNumberTypeFloat32, {3, 4, 2});
  lite::Tensor out_tensor0(kNumberTypeFloat32, {3, 2});
  lite::Tensor out_tensor1(kNumberTypeFloat32, {3, 2});
  lite::Tensor out_tensor2(kNumberTypeFloat32, {3, 2});
  lite::Tensor out_tensor3(kNumberTypeFloat32, {3, 2});
  float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
  float output_data0[6] = {0};
  float output_data1[6] = {0};
  float output_data2[6] = {0};
  float output_data3[6] = {0};
  in_tensor.set_data(input_data);
  out_tensor0.set_data(output_data0);
  out_tensor1.set_data(output_data1);
  out_tensor2.set_data(output_data2);
  out_tensor3.set_data(output_data3);
  std::vector<lite::Tensor *> inputs = {&in_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor0, &out_tensor1, &out_tensor2, &out_tensor3};

  UnstackParameter parameter = {{}, 4, -2, 3, 4, 2};
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Unstack};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  EXPECT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(&parameter), ctx.get(), desc);
  EXPECT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);

  float expect0[] = {1, 2, 9, 10, 17, 18};
  float expect1[] = {3, 4, 11, 12, 19, 20};
  float expect2[] = {5, 6, 13, 14, 21, 22};
  float expect3[] = {7, 8, 15, 16, 23, 24};
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(output_data0[i], expect0[i]);
    EXPECT_EQ(output_data1[i], expect1[i]);
    EXPECT_EQ(output_data2[i], expect2[i]);
    EXPECT_EQ(output_data3[i], expect3[i]);
  }

  in_tensor.set_data(nullptr);
  out_tensor0.set_data(nullptr);
  out_tensor1.set_data(nullptr);
  out_tensor2.set_data(nullptr);
  out_tensor3.set_data(nullptr);
}

TEST_F(TestUnstackFp32, Unstack2) {
  lite::Tensor in_tensor(kNumberTypeFloat32, {3, 4, 2});
  lite::Tensor out_tensor0(kNumberTypeFloat32, {4, 2});
  lite::Tensor out_tensor1(kNumberTypeFloat32, {4, 2});
  lite::Tensor out_tensor2(kNumberTypeFloat32, {4, 2});
  float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
  float output_data0[8] = {0};
  float output_data1[8] = {0};
  float output_data2[8] = {0};
  in_tensor.set_data(input_data);
  out_tensor0.set_data(output_data0);
  out_tensor1.set_data(output_data1);
  out_tensor2.set_data(output_data2);
  std::vector<lite::Tensor *> inputs = {&in_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor0, &out_tensor1, &out_tensor2};

  UnstackParameter parameter = {{}, 3, 0, 1, 3, 8};
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Unstack};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  EXPECT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(&parameter), ctx.get(), desc);
  EXPECT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);

  float expect0[] = {1, 2, 3, 4, 5, 6, 7, 8};
  float expect1[] = {9, 10, 11, 12, 13, 14, 15, 16};
  float expect2[] = {17, 18, 19, 20, 21, 22, 23, 24};
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(output_data0[i], expect0[i]);
    EXPECT_EQ(output_data1[i], expect1[i]);
    EXPECT_EQ(output_data2[i], expect2[i]);
  }

  in_tensor.set_data(nullptr);
  out_tensor0.set_data(nullptr);
  out_tensor1.set_data(nullptr);
  out_tensor2.set_data(nullptr);
}
}  // namespace mindspore
