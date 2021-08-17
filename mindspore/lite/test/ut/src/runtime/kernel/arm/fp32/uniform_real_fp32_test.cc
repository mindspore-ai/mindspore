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
#include "common/common_test.h"
#include "nnacl/random_parameter.h"
#include "mindspore/lite/src/kernel_registry.h"

namespace mindspore {
class TestUniformRealFp32 : public mindspore::CommonTest {
 public:
  TestUniformRealFp32() {}
};

TEST_F(TestUniformRealFp32, UniformReal) {
  lite::Tensor out_tensor0(kNumberTypeFloat32, {10});
  float output_data0[10] = {0};
  out_tensor0.set_data(output_data0);
  std::vector<lite::Tensor *> inputs = {};
  std::vector<lite::Tensor *> outputs = {&out_tensor0};

  RandomParam parameter;
  parameter.op_parameter_.type_ = schema::PrimitiveType_UniformReal;
  parameter.seed_ = 42;
  parameter.seed2_ = 959;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt32, schema::PrimitiveType_UniformReal};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  EXPECT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(&parameter), ctx.get(), desc);
  EXPECT_NE(kernel, nullptr);

  auto ret = kernel->Init();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);
  EXPECT_NEAR(0.138693, output_data0[0], 0.000001);
  EXPECT_NEAR(0.511552, output_data0[1], 0.000001);
  EXPECT_NEAR(0.27194, output_data0[2], 0.000001);
  EXPECT_NEAR(0.336527, output_data0[3], 0.000001);
  EXPECT_NEAR(0.896684, output_data0[4], 0.000001);
  EXPECT_NEAR(0.476402, output_data0[5], 0.000001);
  EXPECT_NEAR(0.155924, output_data0[6], 0.000001);
  EXPECT_NEAR(0.817732, output_data0[7], 0.000001);
  EXPECT_NEAR(0.619868, output_data0[8], 0.000001);
  EXPECT_NEAR(0.274392, output_data0[9], 0.000001);

  for (int i = 0; i < 10; ++i) {
    std::cout << output_data0[i] << " ";
  }
  out_tensor0.set_data(nullptr);
  delete kernel;
}
}  // namespace mindspore
