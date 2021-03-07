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
#include "mindspore/lite/nnacl/fp32/unique_fp32.h"
#include "mindspore/lite/src/kernel_registry.h"

namespace mindspore {
class TestUniqueFp32 : public mindspore::CommonTest {
 public:
  TestUniqueFp32() {}
};

TEST_F(TestUniqueFp32, Unique) {
  lite::Tensor in_tensor(kNumberTypeFloat32, {9});
  lite::Tensor out_tensor0(kNumberTypeFloat32, {9});
  lite::Tensor out_tensor1(kNumberTypeInt32, {9});
  float input_data[] = {1, 1, 2, 4, 4, 4, 7, 8, 8};
  float output_data0[9] = {0};
  int output_data1[9] = {0};
  in_tensor.set_data(input_data);
  out_tensor0.set_data(output_data0);
  out_tensor1.set_data(output_data1);
  std::vector<lite::Tensor *> inputs = {&in_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor0, &out_tensor1};

  OpParameter parameter = {0};
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Unique};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  EXPECT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, &parameter, ctx.get(), desc);
  EXPECT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);

  float expect0[] = {1, 2, 4, 7, 8};
  int expect1[] = {0, 0, 1, 2, 2, 2, 3, 4, 4};
  EXPECT_EQ(out_tensor0.ElementsNum(), 5);

  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(output_data0[i], expect0[i]);
  }
  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(output_data1[i], expect1[i]);
  }

  in_tensor.set_data(nullptr);
  out_tensor0.set_data(nullptr);
  out_tensor1.set_data(nullptr);
}
}  // namespace mindspore
