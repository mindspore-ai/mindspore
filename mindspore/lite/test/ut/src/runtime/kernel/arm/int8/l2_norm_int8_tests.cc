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
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/include/context.h"
#include "nnacl/l2_norm_parameter.h"

namespace mindspore {
class TestL2NormInt8 : public mindspore::CommonTest {
 public:
  TestL2NormInt8() {}
  L2NormParameter param_;
};

TEST_F(TestL2NormInt8, norm) {
  lite::Tensor in_tensor(kNumberTypeInt8, {2, 1, 1, 5});
  lite::Tensor out_tensor(kNumberTypeInt8, {2, 1, 1, 5});
  // -6.0 -4.5 -3.0 -1.5 0 1.0 2.5 3.5 4.0 6.0
  int8_t input_data[] = {-128, -96, -64, -32, 0, 21, 53, 74, 85, 127};
  int8_t output_data[10] = {0};
  in_tensor.set_data(input_data);
  out_tensor.set_data(output_data);

  const lite::QuantArg quant_in = {0.0470588244497776f, 0};
  const lite::QuantArg quant_out = {0.0078125f, 0};
  in_tensor.AddQuantParam(quant_in);
  out_tensor.AddQuantParam(quant_out);

  std::vector<lite::Tensor *> inputs = {&in_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor};

  param_.axis_num_ = 1;
  param_.axis_[0] = -1;
  param_.epsilon_ = 1e-6;
  param_.act_type_ = ActType_No;
  param_.shape_ = nullptr;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_L2NormalizeFusion};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(&param_), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);
  int8_t expect[10] = {-93, -70, -47, -23, 0, 15, 38, 53, 61, 91};
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(output_data[i], expect[i]);
  }
  free(param_.axis_);
  in_tensor.set_data(nullptr);
  out_tensor.set_data(nullptr);
}

TEST_F(TestL2NormInt8, norm2) {
  lite::Tensor in_tensor(kNumberTypeInt8, {1, 1, 1, 51});
  lite::Tensor out_tensor(kNumberTypeInt8, {1, 1, 1, 51});
  int8_t input_data[] = {65, 83, 90, 0, 58, 0,  60, 0, 52, 58, 10, 0,  0,  54, 53, 0,  0,
                         0,  99, 45, 0, 59, 66, 0,  0, 44, 48, 68, 88, 0,  16, 55, 60, 0,
                         0,  52, 0,  0, 66, 33, 0,  0, 81, 0,  0,  74, 57, 0,  0,  0,  26};
  int8_t output_data[51] = {0};
  in_tensor.set_data(input_data);
  out_tensor.set_data(output_data);

  const lite::QuantArg quant_in = {0.0470588244f, 0};
  const lite::QuantArg quant_out = {0.0078125f, 0};
  in_tensor.AddQuantParam(quant_in);
  out_tensor.AddQuantParam(quant_out);

  std::vector<lite::Tensor *> inputs = {&in_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor};

  param_.axis_num_ = 1;
  param_.axis_[0] = -1;
  param_.epsilon_ = 1e-6;
  param_.act_type_ = ActType_No;
  param_.shape_ = nullptr;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_L2NormalizeFusion};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(&param_), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);
  int8_t expect[] = {26, 33, 36, 0, 23, 0,  24, 0, 21, 23, 4, 0, 0,  21, 21, 0, 0,  0, 39, 18, 0,  23, 26, 0, 0, 17,
                     19, 27, 35, 0, 6,  22, 24, 0, 0,  21, 0, 0, 26, 13, 0,  0, 32, 0, 0,  29, 22, 0,  0,  0, 10};
  for (size_t i = 0; i < sizeof(expect); ++i) {
    EXPECT_EQ(output_data[i], expect[i]);
  }
  free(param_.axis_);
  in_tensor.set_data(nullptr);
  out_tensor.set_data(nullptr);
}
}  // namespace mindspore
