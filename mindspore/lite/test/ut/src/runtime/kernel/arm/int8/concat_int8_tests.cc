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
#include "schema/inner/model_generated.h"
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/nnacl/concat_parameter.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/src/lite_kernel.h"
#include "mindspore/lite/src/tensor.h"

namespace mindspore {

class TestConcatInt8 : public mindspore::CommonTest {
 public:
  TestConcatInt8() {}
};

TEST_F(TestConcatInt8, Concat1_axis0) {
  std::vector<int8_t> input1 = {1, 2, 3, 4, 5, 6};
  std::vector<int> shape1 = {3, 2};
  std::vector<int8_t> input2 = {7, 8, 9, 10, 11, 12};
  std::vector<int> shape2 = {3, 2};
  std::vector<int8_t *> input(2, nullptr);
  input[0] = input1.data();
  input[1] = input2.data();

  int8_t output[12];
  std::vector<int> output_shape = {6, 2};

  lite::QuantArg input_quant_arg;
  input_quant_arg.scale = 1.0;
  input_quant_arg.zeroPoint = 0;
  lite::QuantArg output_quant_arg;
  output_quant_arg.scale = 1.0;
  output_quant_arg.zeroPoint = 0;

  lite::Tensor *input_tensor1 = new lite::Tensor;
  TypeId tid_int8 = kNumberTypeInt8;
  input_tensor1->set_data(input1.data());
  input_tensor1->set_shape(shape1);
  input_tensor1->AddQuantParam(input_quant_arg);
  input_tensor1->set_data_type(tid_int8);

  lite::Tensor *input_tensor2 = new lite::Tensor;
  input_tensor2->set_data(input2.data());
  input_tensor2->set_shape(shape2);
  input_tensor2->AddQuantParam(input_quant_arg);
  input_tensor2->set_data_type(tid_int8);

  std::vector<lite::Tensor *> inputs_tensor(2);
  inputs_tensor[0] = input_tensor1;
  inputs_tensor[1] = input_tensor2;

  std::vector<lite::Tensor *> outputs_tensor(1);
  lite::Tensor *output0_tensor = new lite::Tensor;
  output0_tensor->set_data(output);
  output0_tensor->set_shape(output_shape);
  output0_tensor->AddQuantParam(output_quant_arg);
  output0_tensor->set_data_type(tid_int8);
  outputs_tensor[0] = output0_tensor;

  ConcatParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_Concat;
  op_param.axis_ = 0;
  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_Concat};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor->shape();
  ASSERT_EQ(output_tensor_shape, output_shape);
  kernel->Run();

  std::vector<int8_t> except_result = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  PrintData("output data", output, input1.size() + input2.size());
  ASSERT_EQ(0, CompareOutputData(output, except_result.data(), input1.size() + input2.size(), 0.000001));
  input_tensor1->set_data(nullptr);
  input_tensor2->set_data(nullptr);
  output0_tensor->set_data(nullptr);
  delete input_tensor1;
  delete input_tensor2;
  delete output0_tensor;
  delete ctx;
}

TEST_F(TestConcatInt8, Concat1_axis1_thread2) {
  std::vector<int8_t> input1 = {10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25};
  std::vector<int> shape1 = {2, 3, 2};
  std::vector<int8_t> input2 = {30, 31, 32, 33};
  std::vector<int> shape2 = {2, 1, 2};
  std::vector<int8_t *> input(2, nullptr);
  input[0] = input1.data();
  input[1] = input2.data();

  int8_t output[16];
  std::vector<int> output_shape = {2, 4, 2};

  lite::QuantArg input_quant_arg;
  input_quant_arg.scale = 1.0;
  input_quant_arg.zeroPoint = 0;
  lite::QuantArg output_quant_arg;
  output_quant_arg.scale = 1.0;
  output_quant_arg.zeroPoint = 0;

  lite::Tensor *input_tensor1 = new lite::Tensor;
  TypeId tid_int8 = kNumberTypeInt8;
  input_tensor1->set_data(input1.data());
  input_tensor1->set_shape(shape1);
  input_tensor1->AddQuantParam(input_quant_arg);
  input_tensor1->set_data_type(tid_int8);

  lite::Tensor *input_tensor2 = new lite::Tensor;
  input_tensor2->set_data(input2.data());
  input_tensor2->set_shape(shape2);
  input_tensor2->AddQuantParam(input_quant_arg);
  input_tensor2->set_data_type(tid_int8);

  std::vector<lite::Tensor *> inputs_tensor(2);
  inputs_tensor[0] = input_tensor1;
  inputs_tensor[1] = input_tensor2;

  std::vector<lite::Tensor *> outputs_tensor(1);
  lite::Tensor *output0_tensor = new lite::Tensor;
  output0_tensor->set_data(output);
  output0_tensor->set_shape(output_shape);
  output0_tensor->AddQuantParam(output_quant_arg);
  output0_tensor->set_data_type(tid_int8);
  outputs_tensor[0] = output0_tensor;

  ConcatParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_Concat;
  op_param.axis_ = 1;
  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_Concat};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor->shape();
  ASSERT_EQ(output_tensor_shape, output_shape);
  kernel->Run();

  std::vector<int8_t> except_result = {10, 11, 12, 13, 14, 15, 30, 31, 20, 21, 22, 23, 24, 25, 32, 33};
  PrintData("output data", output, input1.size() + input2.size());
  ASSERT_EQ(0, CompareOutputData(output, except_result.data(), input1.size() + input2.size(), 0.000001));

  input_tensor1->set_data(nullptr);
  input_tensor2->set_data(nullptr);
  output0_tensor->set_data(nullptr);
  delete input_tensor1;
  delete input_tensor2;
  delete output0_tensor;
  delete ctx;
}

TEST_F(TestConcatInt8, Concat1_axis1_thread2_quant1) {
  std::vector<int8_t> input1 = {10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25};
  std::vector<int> shape1 = {2, 3, 2};
  std::vector<int8_t> input2 = {30, 31, 32, 33};
  std::vector<int> shape2 = {2, 1, 2};
  std::vector<int8_t *> input(2, nullptr);
  input[0] = input1.data();
  input[1] = input2.data();

  int8_t output[16];
  std::vector<int> output_shape = {2, 4, 2};

  lite::QuantArg input_quant_arg;
  input_quant_arg.scale = 1.0;
  input_quant_arg.zeroPoint = 0;
  lite::QuantArg output_quant_arg;
  output_quant_arg.scale = 2.0;
  output_quant_arg.zeroPoint = 0;

  lite::Tensor *input_tensor1 = new lite::Tensor;
  TypeId tid_int8 = kNumberTypeInt8;
  input_tensor1->set_data(input1.data());
  input_tensor1->set_shape(shape1);
  input_tensor1->AddQuantParam(input_quant_arg);
  input_tensor1->set_data_type(tid_int8);

  lite::Tensor *input_tensor2 = new lite::Tensor;
  input_tensor2->set_data(input2.data());
  input_tensor2->set_shape(shape2);
  input_tensor2->AddQuantParam(input_quant_arg);
  input_tensor2->set_data_type(tid_int8);

  std::vector<lite::Tensor *> inputs_tensor(2);
  inputs_tensor[0] = input_tensor1;
  inputs_tensor[1] = input_tensor2;

  std::vector<lite::Tensor *> outputs_tensor(1);
  lite::Tensor *output0_tensor = new lite::Tensor;
  output0_tensor->set_data(output);
  output0_tensor->set_shape(output_shape);
  output0_tensor->AddQuantParam(output_quant_arg);
  output0_tensor->set_data_type(tid_int8);
  outputs_tensor[0] = output0_tensor;

  ConcatParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_Concat;
  op_param.axis_ = 1;
  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_Concat};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor->shape();
  ASSERT_EQ(output_tensor_shape, output_shape);
  kernel->Run();

  std::vector<int8_t> except_result = {5, 6, 6, 7, 7, 8, 15, 16, 10, 11, 11, 12, 12, 13, 16, 17};
  PrintData("output data", output, input1.size() + input2.size());
  ASSERT_EQ(0, CompareOutputData(output, except_result.data(), input1.size() + input2.size(), 0.000001));

  input_tensor1->set_data(nullptr);
  input_tensor2->set_data(nullptr);
  output0_tensor->set_data(nullptr);
  delete input_tensor1;
  delete input_tensor2;
  delete output0_tensor;
  delete ctx;
}

}  // namespace mindspore
