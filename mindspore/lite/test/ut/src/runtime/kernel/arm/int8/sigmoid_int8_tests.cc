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

#include <memory>
#include "schema/inner/model_generated.h"
#include "common/common_test.h"
#include "mindspore/lite/nnacl/fp32/activation_fp32.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/include/context.h"

namespace mindspore {
class TestSigmoidInt8 : public mindspore::CommonTest {
 public:
  TestSigmoidInt8() {}
};

TEST_F(TestSigmoidInt8, Sigmoid) {
  lite::Tensor in_tensor(kNumberTypeInt8, {4, 4});
  lite::Tensor out_tensor(kNumberTypeInt8, {4, 4});

  int8_t input_data[] = {0, 0, 0, 0, 1, 1, 1, 1};  // -3.5f, -3.0f, -2.5f, 0.f, 2.5f, 3.0f, 3.5f, 6.0f
  int8_t output_data[8] = {0};
  in_tensor.set_data(input_data);
  out_tensor.set_data(output_data);

  const lite::QuantArg quant_in = {1.0, 0};   // -4.0 -- 7.0
  const lite::QuantArg quant_out = {1.0, 0};  // -3.0 -- 7.0
  in_tensor.AddQuantParam(quant_in);
  out_tensor.AddQuantParam(quant_out);

  std::vector<lite::Tensor *> inputs = {&in_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor};

  ActivationParameter parameter = {0};
  parameter.op_parameter_.type_ = schema::PrimitiveType_Activation;
  parameter.type_ = schema::ActivationType_SIGMOID;

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_Activation};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(&parameter), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);

  int8_t expect[8] = {1, 1, 1, 1, 1, 1, 1, 1};  // 0, 0, -0.208333, 0, 2.29167, 3, 3.5, 6
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(output_data[i], expect[i]);
  }

  in_tensor.set_data(nullptr);
  out_tensor.set_data(nullptr);
}
}  // namespace mindspore
