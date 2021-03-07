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
#include "mindspore/lite/nnacl/fp32/topk_fp32.h"
#include "mindspore/lite/src/kernel_registry.h"

namespace mindspore {
class TestTopKFp32 : public mindspore::CommonTest {
 public:
  TestTopKFp32() {}
};

TEST_F(TestTopKFp32, TopK) {
  lite::Tensor in_tensor(kNumberTypeFloat32, {2, 2, 3});
  lite::Tensor out_tensor0(kNumberTypeFloat32, {2, 2, 2});
  lite::Tensor out_tensor1(kNumberTypeInt32, {2, 2, 2});

  float input_data[] = {1, 2, 3, 6, 5, 4, 9, 8, 7, 10, 12, 11};
  float output_data0[8] = {0};
  int32_t output_data1[8] = {0};
  in_tensor.set_data(input_data);
  out_tensor0.set_data(output_data0);
  out_tensor1.set_data(output_data1);
  std::vector<lite::Tensor *> inputs = {&in_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor0, &out_tensor1};

  TopkParameter parameter = {{}, 2, true, 3, 4};
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_TopKFusion};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(&parameter), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);

  float expect0[] = {3, 2, 6, 5, 9, 8, 12, 11};
  int32_t expect1[] = {2, 1, 0, 1, 0, 1, 1, 2};
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(output_data0[i], expect0[i]);
    EXPECT_EQ(output_data1[i], expect1[i]);
  }

  in_tensor.set_data(nullptr);
  out_tensor0.set_data(nullptr);
  out_tensor1.set_data(nullptr);
}
}  // namespace mindspore
