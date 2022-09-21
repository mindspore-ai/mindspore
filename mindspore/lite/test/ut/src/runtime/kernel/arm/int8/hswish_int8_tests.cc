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
#include "schema/inner/model_generated.h"
#include "common/common_test.h"
#include "mindspore/lite/src/litert/kernel/cpu/fp32/activation_fp32.h"
#include "nnacl/fp32/activation_fp32.h"
#include "mindspore/lite/src/litert/kernel/cpu/int8/hswish_int8.h"
#include "mindspore/lite/src/litert/kernel_registry.h"

namespace mindspore {
class TestHSwishInt8 : public mindspore::CommonTest {
 public:
  TestHSwishInt8() {}
};

TEST_F(TestHSwishInt8, HSwish) {
  lite::Tensor in_tensor(kNumberTypeInt8, {4, 4});
  lite::Tensor out_tensor(kNumberTypeInt8, {4, 4});

  int8_t input_data[] = {-116, -105, -93, -35, 23, 35, 46, 104};  // -3.5f, -3.0f, -2.5f, 0.f, 2.5f, 3.0f, 3.5f, 6.0f
  int8_t output_data[8] = {0};
  in_tensor.set_data(input_data);
  out_tensor.set_data(output_data);

  const lite::LiteQuantParam quant_in = {0.0431373f, -35};   // -4.0 -- 7.0
  const lite::LiteQuantParam quant_out = {0.0392157f, -52};  // -3.0 -- 7.0
  in_tensor.AddQuantParam(quant_in);
  out_tensor.AddQuantParam(quant_out);

  std::vector<lite::Tensor *> inputs = {&in_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor};

  ActivationParameter parameter = {0};
  parameter.op_parameter_.type_ = schema::PrimitiveType_Activation;
  parameter.type_ = schema::ActivationType_HSWISH;

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, NHWC, schema::PrimitiveType_Activation};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(&parameter), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);

  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  int8_t expect[8] = {-52, -52, -57, -52, 7, 25, 37, 101};  // 0, 0, -0.208333, 0, 2.29167, 3, 3.5, 6
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(output_data[i], expect[i]);
  }

  in_tensor.set_data(nullptr);
  out_tensor.set_data(nullptr);
  delete kernel;
}
}  // namespace mindspore
