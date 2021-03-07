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
#include "common/common_test.h"
#include "nnacl/fp32/space_to_batch_fp32.h"
#include "mindspore/lite/src/kernel_registry.h"

namespace mindspore {
class SpaceToBatchTestInt8 : public mindspore::CommonTest {
 public:
  SpaceToBatchTestInt8() {}
};

TEST_F(SpaceToBatchTestInt8, test1) {
  lite::Tensor in_tensor(kNumberTypeInt8, {1, 2, 2, 1});
  lite::Tensor out_tensor(kNumberTypeInt8, {4, 2, 2, 1});
  int8_t input_data[] = {1, 2, 3, 4};
  int8_t output_data[16] = {0};
  in_tensor.set_data(input_data);
  out_tensor.set_data(output_data);
  std::vector<lite::Tensor *> inputs = {&in_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor};

  SpaceToBatchParameter parameter = {{}, {2, 2}, {1, 1, 1, 1}, false, 2};
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_SpaceToBatchND};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(&parameter), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);

  int8_t expect[] = {0, 0, 0, 4, 0, 0, 3, 0, 0, 2, 0, 0, 1, 0, 0, 0};
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(output_data[i], expect[i]);
  }
  in_tensor.set_data(nullptr);
  out_tensor.set_data(nullptr);
}
}  // namespace mindspore
