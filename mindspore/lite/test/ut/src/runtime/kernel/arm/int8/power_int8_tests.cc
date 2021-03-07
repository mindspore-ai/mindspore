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
#include "mindspore/lite/src/runtime/kernel/arm/int8/power_int8.h"
#include "mindspore/lite/nnacl/power_parameter.h"
#include "mindspore/lite/src/kernel_registry.h"

namespace mindspore {

class TestPowerInt8 : public mindspore::CommonTest {
 public:
  TestPowerInt8() {}
};

TEST_F(TestPowerInt8, PowerInt8) {
  std::vector<lite::Tensor *> inputs_tensor;
  std::vector<lite::Tensor *> outputs_tensor;

  PowerParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_PowFusion;
  op_param.power_ = 2;
  op_param.scale_ = 1;
  op_param.shift_ = 0;

  lite::QuantArg input_quant_arg;
  input_quant_arg.scale = 0.0156863;
  input_quant_arg.zeroPoint = -128;
  lite::QuantArg output_quant_arg;
  output_quant_arg.scale = 0.0627451;
  output_quant_arg.zeroPoint = -128;

  std::vector<int8_t> input = {-64, -1, 63, 127};
  std::vector<int> in_shape = {1, 1, 1, 4};

  lite::Tensor input0_tensor;
  TypeId tid_int8 = kNumberTypeInt8;
  inputs_tensor.push_back(&input0_tensor);
  input0_tensor.set_data(input.data());
  input0_tensor.set_shape(in_shape);
  input0_tensor.AddQuantParam(input_quant_arg);
  input0_tensor.set_data_type(tid_int8);

  std::vector<int8_t> output(4);
  std::vector<int> output_shape = {1, 1, 1, 4};

  lite::Tensor output0_tensor;
  outputs_tensor.push_back(&output0_tensor);
  output0_tensor.set_data(output.data());
  output0_tensor.AddQuantParam(output_quant_arg);
  output0_tensor.set_data_type(tid_int8);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_PowFusion};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);

  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor.shape();
  kernel->Run();

  std::vector<int8_t> except_result = {-112, -65, 15, 127};
  ASSERT_EQ(0, CompareOutputData(output.data(), except_result.data(), input.size(), 0.000001));

  input0_tensor.set_data(nullptr);
  output0_tensor.set_data(nullptr);
}

TEST_F(TestPowerInt8, normal) {
  std::vector<lite::Tensor *> inputs_tensor;
  std::vector<lite::Tensor *> outputs_tensor;

  PowerParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_PowFusion;
  op_param.scale_ = 1;
  op_param.shift_ = 0;

  lite::QuantArg input_quant_arg;
  input_quant_arg.scale = 0.0156863;
  input_quant_arg.zeroPoint = -128;

  lite::QuantArg exp_quant_arg;
  exp_quant_arg.scale = 0.0156863;
  exp_quant_arg.zeroPoint = -128;

  lite::QuantArg output_quant_arg;
  output_quant_arg.scale = 0.0352941;
  output_quant_arg.zeroPoint = -128;

  std::vector<int8_t> input = {-64, -1, 63, 127};
  std::vector<int> in_shape = {1, 1, 1, 4};

  std::vector<int8_t> input1 = {127, 63, -1, -64};
  std::vector<int> in_shape1 = {1, 1, 1, 4};

  lite::Tensor input0_tensor, input1_tensor;
  TypeId tid_int8 = kNumberTypeInt8;
  inputs_tensor.push_back(&input0_tensor);
  inputs_tensor.push_back(&input1_tensor);
  input0_tensor.set_data(input.data());
  input0_tensor.set_shape(in_shape);
  input0_tensor.AddQuantParam(input_quant_arg);
  input0_tensor.set_data_type(tid_int8);

  input1_tensor.set_data(input1.data());
  input1_tensor.set_shape(in_shape1);
  input1_tensor.AddQuantParam(exp_quant_arg);
  input1_tensor.set_data_type(tid_int8);

  std::vector<int8_t> output(4);
  std::vector<int> output_shape = {1, 1, 1, 4};

  lite::Tensor output0_tensor;
  outputs_tensor.push_back(&output0_tensor);
  output0_tensor.set_data(output.data());
  output0_tensor.AddQuantParam(output_quant_arg);
  output0_tensor.set_data_type(tid_int8);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_PowFusion};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);

  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor.shape();
  kernel->Run();

  std::vector<int8_t> except_result = {-99, 95, 124, -14};
  ASSERT_EQ(0, CompareOutputData(output.data(), except_result.data(), input.size(), 0.000001));

  input0_tensor.set_data(nullptr);
  output0_tensor.set_data(nullptr);
}
}  // namespace mindspore
