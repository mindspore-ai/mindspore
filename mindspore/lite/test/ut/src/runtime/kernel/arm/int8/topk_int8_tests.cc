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
#include "mindspore/lite/src/runtime/kernel/arm/nnacl/fp32/topk.h"
#include "mindspore/lite/src/kernel_registry.h"

namespace mindspore {
class TestTopKInt8 : public mindspore::CommonTest {
 public:
  TestTopKInt8() {}
};

TEST_F(TestTopKInt8, TopK) {
  lite::tensor::Tensor in_tensor(kNumberTypeInt8, {2, 2, 3});
  lite::tensor::Tensor out_tensor0(kNumberTypeInt8, {2, 2, 2});
  lite::tensor::Tensor out_tensor1(kNumberTypeInt32, {2, 2, 2});
  int8_t input_data[] = {1, 2, 3,   6, 5, 4,   9, 8, 7,   10, 12, 11};
  int8_t output_data0[8] = {0};
  int32_t output_data1[8] = {0};
  in_tensor.SetData(input_data);
  out_tensor0.SetData(output_data0);
  out_tensor1.SetData(output_data1);
  std::vector<lite::tensor::Tensor *> inputs = {&in_tensor};
  std::vector<lite::tensor::Tensor *> outputs = {&out_tensor0, &out_tensor1};

  TopkParameter parameter = {{}, 3, 4, 2, true};
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_TopK};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);

  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(&parameter), nullptr, desc, nullptr);
  ASSERT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);

  int8_t expect0[] = {3, 2,   6, 5,   9, 8,   12, 11};
  int32_t expect1[] = {2, 1,   0, 1,   0, 1,   1, 2};
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(output_data0[i], expect0[i]);
    EXPECT_EQ(output_data1[i], expect1[i]);
  }

  in_tensor.SetData(nullptr);
  out_tensor0.SetData(nullptr);
  out_tensor1.SetData(nullptr);
}
}  // namespace mindspore
