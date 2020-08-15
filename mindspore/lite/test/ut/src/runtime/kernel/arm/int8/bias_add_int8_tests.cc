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
#include "mindspore/lite/src/runtime/kernel/arm/int8/bias_add_int8.h"
#include "mindspore/lite/src/kernel_registry.h"

using mindspore::lite::DeviceType;

namespace mindspore {
class TestBiasAddInt8 : public mindspore::CommonTest {
 public:
  TestBiasAddInt8() {}
};

TEST_F(TestBiasAddInt8, BiasAdd) {
  lite::tensor::Tensor in_tensor0(kNumberTypeInt8, {1, 2, 3, 2});
  lite::tensor::Tensor in_tensor1(kNumberTypeInt8, {2});
  lite::tensor::Tensor out_tensor(kNumberTypeInt8, {1, 2, 3, 2});
  int8_t input_data0[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int8_t input_data1[] = {1, 1};
  int8_t output_data[12] = {0};
  in_tensor0.SetData(input_data0);
  in_tensor1.SetData(input_data1);
  out_tensor.SetData(output_data);
  std::vector<lite::tensor::Tensor *> inputs = {&in_tensor0, &in_tensor1};
  std::vector<lite::tensor::Tensor *> outputs = {&out_tensor};

  ArithmeticParameter parameter = {};
  int dims[] = {1, 2, 3, 4};
  parameter.ndim_ = 4;
  for (int i = 0; i < 4; i++) {
    parameter.in_shape0_[i] = dims[i];
    parameter.in_shape1_[i] = 1;
    parameter.out_shape_[i] = dims[i];
  }
  parameter.in_shape1_[3] = dims[3];

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_BiasAdd};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  EXPECT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::Context>();
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(&parameter), ctx.get(), desc, nullptr);
  EXPECT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);

  float expect[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  for (int i = 0; i < 12; ++i) {
    EXPECT_EQ(output_data[i], expect[i]);
  }

  in_tensor0.SetData(nullptr);
  in_tensor1.SetData(nullptr);
  out_tensor.SetData(nullptr);
}
}  // namespace mindspore
