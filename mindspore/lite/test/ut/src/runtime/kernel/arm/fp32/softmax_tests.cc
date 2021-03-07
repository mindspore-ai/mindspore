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
#include "nnacl/softmax_parameter.h"
#include "mindspore/lite/src/kernel_registry.h"

namespace mindspore {
class TestSoftmaxFp32 : public mindspore::CommonTest {
 public:
  TestSoftmaxFp32() {}
};

TEST_F(TestSoftmaxFp32, 001) {
  lite::Tensor in_tensor(kNumberTypeFloat32, {2, 1, 1, 5});
  lite::Tensor out_tensor(kNumberTypeFloat32, {2, 1, 1, 5});
  float input_data[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  float output_data[10] = {0};
  in_tensor.set_data(input_data);
  out_tensor.set_data(output_data);
  std::vector<lite::Tensor *> inputs = {&in_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor};

  SoftmaxParameter parameter = {{}, -1, {2, 1, 1, 5}, 10, 4};
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Softmax};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(&parameter), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);

  float expect[] = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f};
  for (size_t i = 0; i < sizeof(expect) / sizeof(expect[0]); ++i) {
    EXPECT_EQ(output_data[i], expect[i]);
  }
  in_tensor.set_data(nullptr);
  out_tensor.set_data(nullptr);
}
}  // namespace mindspore
