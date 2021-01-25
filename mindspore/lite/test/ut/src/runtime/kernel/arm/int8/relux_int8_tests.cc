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
#include "mindspore/lite/src/runtime/kernel/arm/int8/relux_int8.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/include/context.h"

namespace mindspore {
class TestReluXInt8 : public mindspore::CommonTest {
 public:
  TestReluXInt8() {}
};

TEST_F(TestReluXInt8, Relu) {
  lite::Tensor in_tensor(kNumberTypeInt8, {2, 2});
  lite::Tensor out_tensor(kNumberTypeInt8, {2, 2});

  int8_t input_data[] = {-102, 25, -51, 89};  // -0.8 0.2 -0.4 0.7
  int8_t output_data[4] = {0};
  in_tensor.set_data(input_data);
  out_tensor.set_data(output_data);

  const lite::QuantArg quant_in = {0.00784314f, 0};  // -1.0--1.0 ->
  const lite::QuantArg quant_out = {0.00784314f, 0};
  in_tensor.AddQuantParam(quant_in);
  out_tensor.AddQuantParam(quant_out);

  std::vector<lite::Tensor *> inputs = {&in_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor};

  ActivationParameter parameter = {0};
  parameter.op_parameter_.type_ = schema::PrimitiveType_Activation;
  parameter.type_ = schema::ActivationType_RELU;

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_Activation};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(&parameter), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);

  int8_t expect0[4] = {0, 26, 0, 90};  //
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(output_data[i], expect0[i]);
  }

  in_tensor.set_data(nullptr);
  out_tensor.set_data(nullptr);
}

TEST_F(TestReluXInt8, Relu6) {
  lite::Tensor in_tensor(kNumberTypeInt8, {2, 4});
  lite::Tensor out_tensor(kNumberTypeInt8, {2, 4});

  // -2.5f, -1.5f, 1.25f, 3.0f, 4.5f, 6.0f, 6.5f, 9.0f
  int8_t input_data[] = {-118, -98, -44, -10, 19, 49, 59, 108};
  int8_t output_data[8] = {0};
  in_tensor.set_data(input_data);
  out_tensor.set_data(output_data);

  const lite::QuantArg quant_in = {0.0509804f, -69};    // -3.0 -- 10.0
  const lite::QuantArg quant_out = {0.0392157f, -128};  // 0.0 -- 10.0
  in_tensor.AddQuantParam(quant_in);
  out_tensor.AddQuantParam(quant_out);

  std::vector<lite::Tensor *> inputs = {&in_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor};

  ActivationParameter parameter = {0};
  parameter.op_parameter_.type_ = schema::PrimitiveType_Activation;
  parameter.type_ = schema::ActivationType_RELU6;

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_Activation};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(&parameter), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);

  // 0.0f, 0.0f, 1.25f, 3.0f, 4.5f, 6.0f, 6.0f, 6.0f
  int8_t expect[8] = {-128, -128, -96, -52, -14, 25, 25, 25};
  for (unsigned int i = 0; i < sizeof(expect); ++i) {
    EXPECT_EQ(output_data[i], expect[i]);
  }

  in_tensor.set_data(nullptr);
  out_tensor.set_data(nullptr);
}
}  // namespace mindspore
