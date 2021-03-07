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
#include "mindspore/lite/nnacl/split_parameter.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/src/lite_kernel.h"
#include "mindspore/lite/src/tensor.h"

namespace mindspore {

class TestSplitInt8 : public mindspore::CommonTest {
 public:
  TestSplitInt8() {}
};

TEST_F(TestSplitInt8, Split_quant0_thread2) {
  std::vector<int8_t> input1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int> shape1 = {2, 3, 2};
  std::vector<int8_t *> input(1, nullptr);
  input[0] = input1.data();

  const int output1_size = 4;
  int8_t output1[4];
  const int output2_size = 8;
  int8_t output2[8];
  std::vector<int> output1_shape = {2, 1, 2};
  std::vector<int> output2_shape = {2, 2, 2};

  lite::QuantArg input_quant_arg;
  input_quant_arg.scale = 1.0;
  input_quant_arg.zeroPoint = 0;
  lite::QuantArg output_quant_arg;
  output_quant_arg.scale = 1.0;
  output_quant_arg.zeroPoint = 0;

  TypeId tid_int8 = kNumberTypeInt8;
  lite::Tensor *input_tensor1 = new lite::Tensor;
  input_tensor1->set_data(input1.data());
  input_tensor1->set_shape(shape1);
  input_tensor1->AddQuantParam(input_quant_arg);
  input_tensor1->set_data_type(tid_int8);
  std::vector<lite::Tensor *> inputs_tensor(1);
  inputs_tensor[0] = input_tensor1;

  lite::Tensor *output1_tensor = new lite::Tensor;
  output1_tensor->set_data(output1);
  output1_tensor->set_shape(output1_shape);
  output1_tensor->AddQuantParam(output_quant_arg);
  output1_tensor->set_data_type(tid_int8);
  lite::Tensor *output2_tensor = new lite::Tensor;
  output2_tensor->set_data(output2);
  output2_tensor->set_shape(output2_shape);
  output2_tensor->AddQuantParam(output_quant_arg);
  output2_tensor->set_data_type(tid_int8);
  std::vector<lite::Tensor *> outputs_tensor(2);
  outputs_tensor[0] = output1_tensor;
  outputs_tensor[1] = output2_tensor;

  SplitParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_Split;
  op_param.num_split_ = 2;
  op_param.split_dim_ = 1;
  op_param.split_sizes_[0] = 1;
  op_param.split_sizes_[1] = 2;
  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_Split};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output1_tensor_shape = output1_tensor->shape();
  auto output2_tensor_shape = output2_tensor->shape();
  ASSERT_EQ(output1_tensor_shape, output1_shape);
  ASSERT_EQ(output2_tensor_shape, output2_shape);
  kernel->Run();

  std::vector<int8_t> except_result1 = {1, 2, 7, 8};
  std::vector<int8_t> except_result2 = {3, 4, 5, 6, 9, 10, 11, 12};
  PrintData("output data", output1, output1_size);
  PrintData("output data shape", output1_tensor_shape.data(), output1_tensor_shape.size());
  PrintData("output data", output2, output2_size);
  PrintData("output data shape", output2_tensor_shape.data(), output2_tensor_shape.size());
  ASSERT_EQ(0, CompareOutputData(output1, except_result1.data(), output1_size, 0.000001));
  ASSERT_EQ(0, CompareOutputData(output2, except_result2.data(), output2_size, 0.000001));

  input_tensor1->set_data(nullptr);
  output1_tensor->set_data(nullptr);
  output2_tensor->set_data(nullptr);
  delete input_tensor1;
  delete output1_tensor;
  delete output2_tensor;
  delete ctx;
}

TEST_F(TestSplitInt8, Split_quant0_thread2_num) {
  std::vector<int8_t> input1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int> shape1 = {2, 3, 2};
  std::vector<int8_t *> input(1, nullptr);
  input[0] = input1.data();

  const int output1_size = 4;
  int8_t output1[4];
  const int output2_size = 4;
  int8_t output2[4];
  const int output3_size = 4;
  int8_t output3[4];
  std::vector<int> output1_shape = {2, 1, 2};
  std::vector<int> output2_shape = {2, 1, 2};
  std::vector<int> output3_shape = {2, 1, 2};

  lite::QuantArg input_quant_arg;
  input_quant_arg.scale = 1.0;
  input_quant_arg.zeroPoint = 0;
  lite::QuantArg output_quant_arg;
  output_quant_arg.scale = 1.0;
  output_quant_arg.zeroPoint = 0;

  TypeId tid_int8 = kNumberTypeInt8;
  lite::Tensor *input_tensor1 = new lite::Tensor;
  input_tensor1->set_data(input1.data());
  input_tensor1->set_shape(shape1);
  input_tensor1->AddQuantParam(input_quant_arg);
  input_tensor1->set_data_type(tid_int8);
  std::vector<lite::Tensor *> inputs_tensor(1);
  inputs_tensor[0] = input_tensor1;

  lite::Tensor *output1_tensor = new lite::Tensor;
  output1_tensor->set_data(output1);
  output1_tensor->set_shape(output1_shape);
  output1_tensor->AddQuantParam(output_quant_arg);
  output1_tensor->set_data_type(tid_int8);
  lite::Tensor *output2_tensor = new lite::Tensor;
  output2_tensor->set_data(output2);
  output2_tensor->set_shape(output2_shape);
  output2_tensor->AddQuantParam(output_quant_arg);
  output2_tensor->set_data_type(tid_int8);
  lite::Tensor *output3_tensor = new lite::Tensor;
  output3_tensor->set_data(output3);
  output3_tensor->set_shape(output3_shape);
  output3_tensor->AddQuantParam(output_quant_arg);
  output3_tensor->set_data_type(tid_int8);
  std::vector<lite::Tensor *> outputs_tensor(3);
  outputs_tensor[0] = output1_tensor;
  outputs_tensor[1] = output2_tensor;
  outputs_tensor[2] = output3_tensor;

  SplitParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_Split;
  op_param.num_split_ = 3;
  op_param.split_dim_ = 1;
  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_Split};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output1_tensor_shape = output1_tensor->shape();
  auto output2_tensor_shape = output2_tensor->shape();
  auto output3_tensor_shape = output3_tensor->shape();
  ASSERT_EQ(output1_tensor_shape, output1_shape);
  ASSERT_EQ(output2_tensor_shape, output2_shape);
  ASSERT_EQ(output3_tensor_shape, output3_shape);
  kernel->Run();

  std::vector<int8_t> except_result1 = {1, 2, 7, 8};
  std::vector<int8_t> except_result2 = {3, 4, 9, 10};
  std::vector<int8_t> except_result3 = {5, 6, 11, 12};
  PrintData("output data", output1, output1_size);
  PrintData("output data shape", output1_tensor_shape.data(), output1_tensor_shape.size());
  PrintData("output data", output2, output2_size);
  PrintData("output data shape", output2_tensor_shape.data(), output2_tensor_shape.size());
  PrintData("output data", output3, output3_size);
  PrintData("output data shape", output3_tensor_shape.data(), output3_tensor_shape.size());
  ASSERT_EQ(0, CompareOutputData(output1, except_result1.data(), output1_size, 0.000001));
  ASSERT_EQ(0, CompareOutputData(output2, except_result2.data(), output2_size, 0.000001));
  ASSERT_EQ(0, CompareOutputData(output3, except_result3.data(), output3_size, 0.000001));

  input_tensor1->set_data(nullptr);
  output1_tensor->set_data(nullptr);
  output2_tensor->set_data(nullptr);
  output3_tensor->set_data(nullptr);
  delete input_tensor1;
  delete output1_tensor;
  delete output2_tensor;
  delete output3_tensor;
  delete ctx;
}

TEST_F(TestSplitInt8, Split_quant1_thread2_num) {
  std::vector<int8_t> input1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int> shape1 = {2, 3, 2};
  std::vector<int8_t *> input(1, nullptr);
  input[0] = input1.data();

  const int output1_size = 4;
  int8_t output1[4];
  const int output2_size = 4;
  int8_t output2[4];
  const int output3_size = 4;
  int8_t output3[4];
  std::vector<int> output1_shape = {2, 1, 2};
  std::vector<int> output2_shape = {2, 1, 2};
  std::vector<int> output3_shape = {2, 1, 2};

  lite::QuantArg input_quant_arg;
  input_quant_arg.scale = 1.0;
  input_quant_arg.zeroPoint = 0;
  lite::QuantArg output_quant_arg;
  output_quant_arg.scale = 2.0;
  output_quant_arg.zeroPoint = 0;

  TypeId tid_int8 = kNumberTypeInt8;
  lite::Tensor *input_tensor1 = new lite::Tensor;
  input_tensor1->set_data(input1.data());
  input_tensor1->set_shape(shape1);
  input_tensor1->AddQuantParam(input_quant_arg);
  input_tensor1->set_data_type(tid_int8);
  std::vector<lite::Tensor *> inputs_tensor(1);
  inputs_tensor[0] = input_tensor1;

  lite::Tensor *output1_tensor = new lite::Tensor;
  output1_tensor->set_data(output1);
  output1_tensor->set_shape(output1_shape);
  output1_tensor->AddQuantParam(output_quant_arg);
  output1_tensor->set_data_type(tid_int8);
  lite::Tensor *output2_tensor = new lite::Tensor;
  output2_tensor->set_data(output2);
  output2_tensor->set_shape(output2_shape);
  output2_tensor->AddQuantParam(output_quant_arg);
  output2_tensor->set_data_type(tid_int8);
  lite::Tensor *output3_tensor = new lite::Tensor;
  output3_tensor->set_data(output3);
  output3_tensor->set_shape(output3_shape);
  output3_tensor->AddQuantParam(output_quant_arg);
  output3_tensor->set_data_type(tid_int8);
  std::vector<lite::Tensor *> outputs_tensor(3);
  outputs_tensor[0] = output1_tensor;
  outputs_tensor[1] = output2_tensor;
  outputs_tensor[2] = output3_tensor;

  SplitParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_Split;
  op_param.num_split_ = 3;
  op_param.split_dim_ = 1;
  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_Split};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output1_tensor_shape = output1_tensor->shape();
  auto output2_tensor_shape = output2_tensor->shape();
  auto output3_tensor_shape = output3_tensor->shape();
  ASSERT_EQ(output1_tensor_shape, output1_shape);
  ASSERT_EQ(output2_tensor_shape, output2_shape);
  ASSERT_EQ(output3_tensor_shape, output3_shape);
  kernel->Run();

  std::vector<int8_t> except_result1 = {1, 1, 4, 4};
  std::vector<int8_t> except_result2 = {2, 2, 5, 5};
  std::vector<int8_t> except_result3 = {3, 3, 6, 6};
  PrintData("output data", output1, output1_size);
  PrintData("output data shape", output1_tensor_shape.data(), output1_tensor_shape.size());
  PrintData("output data", output2, output2_size);
  PrintData("output data shape", output2_tensor_shape.data(), output2_tensor_shape.size());
  PrintData("output data", output3, output3_size);
  PrintData("output data shape", output3_tensor_shape.data(), output3_tensor_shape.size());
  ASSERT_EQ(0, CompareOutputData(output1, except_result1.data(), output1_size, 0.000001));
  ASSERT_EQ(0, CompareOutputData(output2, except_result2.data(), output2_size, 0.000001));
  ASSERT_EQ(0, CompareOutputData(output3, except_result3.data(), output3_size, 0.000001));

  input_tensor1->set_data(nullptr);
  output1_tensor->set_data(nullptr);
  output2_tensor->set_data(nullptr);
  output3_tensor->set_data(nullptr);
  delete input_tensor1;
  delete output1_tensor;
  delete output2_tensor;
  delete output3_tensor;
  delete ctx;
}

}  // namespace mindspore
