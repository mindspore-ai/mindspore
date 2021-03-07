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
#include "mindspore/lite/src/runtime/kernel/arm/int8/slice_int8.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/include/context.h"

namespace mindspore {
class TestSliceInt8 : public mindspore::CommonTest {
 public:
  TestSliceInt8() {}
};

TEST_F(TestSliceInt8, SliceInt8) {
  lite::Tensor in_tensor(kNumberTypeInt8, {1, 3, 2, 3});
  lite::Tensor out_tensor(kNumberTypeInt8, {1, 1, 2, 3});

  int8_t input_data[] = {105, 35, -27, 0, -63, 99, 16, 45, 67, -49, -115, 106, -98, 119, 103, 81, -114, 68};
  int8_t output_data[12];
  in_tensor.set_data(input_data);
  out_tensor.set_data(output_data);

  const lite::QuantArg quant_in0 = {0.00784314f, 0};  // -1.0--1.0 -> 0--255
  const lite::QuantArg quant_out = {0.00784314f, 0};
  in_tensor.AddQuantParam(quant_in0);
  out_tensor.AddQuantParam(quant_out);

  std::vector<lite::Tensor *> inputs = {&in_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor};

  SliceParameter parameter;
  parameter.begin_[0] = 1;
  parameter.begin_[1] = 0;
  parameter.begin_[2] = 0;
  parameter.size_[0] = -1;
  parameter.size_[1] = -1;
  parameter.size_[2] = -1;
  parameter.param_length_ = 3;

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_SliceFusion};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(&parameter), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);

  int8_t expect0[12] = {16, 45, 67, -49, -115, 106, -98, 119, 103, 81, -114, 68};
  for (int i = 0; i < 12; ++i) {
    EXPECT_EQ(output_data[i], expect0[i]);
  }

  in_tensor.set_data(nullptr);
  out_tensor.set_data(nullptr);
}
}  // namespace mindspore
