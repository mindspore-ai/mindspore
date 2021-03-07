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
#include "mindspore/lite/nnacl/crop_parameter.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/src/lite_kernel.h"
#include "mindspore/lite/src/tensor.h"

namespace mindspore {

class TestCropInt8 : public mindspore::CommonTest {
 public:
  TestCropInt8() {}
};

TEST_F(TestCropInt8, crop_1d_axis0_offset0_quant0_thread2) {
  std::vector<int8_t> input1 = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> shape1 = {8};
  std::vector<int8_t *> input(1, nullptr);
  input[0] = input1.data();

  const int output_size = 7;
  int8_t output[7];
  std::vector<int> output_shape = {7};
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

  std::vector<lite::Tensor *> inputs_tensor(1);
  inputs_tensor[0] = input_tensor1;

  std::vector<lite::Tensor *> outputs_tensor(1);
  lite::Tensor *output0_tensor = new lite::Tensor;
  output0_tensor->set_data(output);
  output0_tensor->set_shape(output_shape);
  output0_tensor->AddQuantParam(output_quant_arg);
  output0_tensor->set_data_type(tid_int8);
  outputs_tensor[0] = output0_tensor;

  CropParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_Crop;
  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  op_param.axis_ = 0;
  op_param.offset_[0] = 1;
  op_param.offset_size_ = 1;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_Crop};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor->shape();
  ASSERT_EQ(output_tensor_shape, output_shape);
  kernel->Run();

  std::vector<int8_t> except_result = {2, 3, 4, 5, 6, 7, 8};
  PrintData("output data", output, output_size);
  PrintData("output data shape", output_tensor_shape.data(), output_tensor_shape.size());
  ASSERT_EQ(0, CompareOutputData(output, except_result.data(), output_size, 0.000001));

  input_tensor1->set_data(nullptr);
  output0_tensor->set_data(nullptr);
  delete input_tensor1;
  delete output0_tensor;
  delete ctx;
}

TEST_F(TestCropInt8, crop_2d_axis1_offset0_quant0_thread2) {
  std::vector<int8_t> input1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<int> shape1 = {2, 8};
  std::vector<int8_t *> input(1, nullptr);
  input[0] = input1.data();

  const int output_size = 14;
  int8_t output[14];
  std::vector<int> output_shape = {2, 7};
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

  std::vector<lite::Tensor *> inputs_tensor(1);
  inputs_tensor[0] = input_tensor1;

  std::vector<lite::Tensor *> outputs_tensor(1);
  lite::Tensor *output0_tensor = new lite::Tensor;
  output0_tensor->set_data(output);
  output0_tensor->set_shape(output_shape);
  output0_tensor->AddQuantParam(output_quant_arg);
  output0_tensor->set_data_type(tid_int8);
  outputs_tensor[0] = output0_tensor;

  CropParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_Crop;
  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  op_param.axis_ = 1;
  op_param.offset_[0] = 1;
  op_param.offset_size_ = 1;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_Crop};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor->shape();
  ASSERT_EQ(output_tensor_shape, output_shape);
  kernel->Run();

  std::vector<int8_t> except_result = {2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16};
  PrintData("output data", output, output_size);
  PrintData("output data shape", output_tensor_shape.data(), output_tensor_shape.size());
  ASSERT_EQ(0, CompareOutputData(output, except_result.data(), output_size, 0.000001));

  input_tensor1->set_data(nullptr);
  output0_tensor->set_data(nullptr);
  delete input_tensor1;
  delete output0_tensor;
  delete ctx;
}

TEST_F(TestCropInt8, crop_3d_axis1_offset0_quant0_thread0) {
  std::vector<int8_t> input1 = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> shape1 = {2, 2, 2};
  std::vector<int8_t *> input(1, nullptr);
  input[0] = input1.data();

  const int output_size = 2;
  int8_t output[2];
  std::vector<int> output_shape = {2, 1, 1};
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

  std::vector<lite::Tensor *> inputs_tensor(1);
  inputs_tensor[0] = input_tensor1;

  std::vector<lite::Tensor *> outputs_tensor(1);
  lite::Tensor *output0_tensor = new lite::Tensor;
  output0_tensor->set_data(output);
  output0_tensor->set_shape(output_shape);
  output0_tensor->AddQuantParam(output_quant_arg);
  output0_tensor->set_data_type(tid_int8);
  outputs_tensor[0] = output0_tensor;

  CropParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_Crop;
  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  op_param.axis_ = 1;
  op_param.offset_[0] = 1;
  op_param.offset_size_ = 1;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_Crop};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor->shape();
  ASSERT_EQ(output_tensor_shape, output_shape);
  kernel->Run();

  std::vector<int8_t> except_result = {4, 8};
  PrintData("output data", output, output_size);
  PrintData("output data shape", output_tensor_shape.data(), output_tensor_shape.size());
  ASSERT_EQ(0, CompareOutputData(output, except_result.data(), output_size, 0.000001));

  input_tensor1->set_data(nullptr);
  output0_tensor->set_data(nullptr);
  delete input_tensor1;
  delete output0_tensor;
  delete ctx;
}

TEST_F(TestCropInt8, crop_3d_axis1_offset0_quant0_thread2) {
  std::vector<int8_t> input1 = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
  std::vector<int> shape1 = {2, 8, 2};
  std::vector<int8_t *> input(1, nullptr);
  input[0] = input1.data();

  const int output_size = 14;
  int8_t output[14];
  std::vector<int> output_shape = {2, 7, 1};
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

  std::vector<lite::Tensor *> inputs_tensor(1);
  inputs_tensor[0] = input_tensor1;

  std::vector<lite::Tensor *> outputs_tensor(1);
  lite::Tensor *output0_tensor = new lite::Tensor;
  output0_tensor->set_data(output);
  output0_tensor->set_shape(output_shape);
  output0_tensor->AddQuantParam(output_quant_arg);
  output0_tensor->set_data_type(tid_int8);
  outputs_tensor[0] = output0_tensor;

  CropParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_Crop;
  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  op_param.axis_ = 1;
  op_param.offset_[0] = 1;
  op_param.offset_size_ = 1;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_Crop};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor->shape();
  ASSERT_EQ(output_tensor_shape, output_shape);
  kernel->Run();

  std::vector<int8_t> except_result = {4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 26, 28, 30, 32};
  PrintData("output data", output, output_size);
  PrintData("output data shape", output_tensor_shape.data(), output_tensor_shape.size());
  ASSERT_EQ(0, CompareOutputData(output, except_result.data(), output_size, 0.000001));

  input_tensor1->set_data(nullptr);
  output0_tensor->set_data(nullptr);
  delete input_tensor1;
  delete output0_tensor;
  delete ctx;
}

TEST_F(TestCropInt8, crop_4d_axis0_offset0_quant0_thread0) {
  std::vector<int8_t> input1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<int> shape1 = {2, 2, 2, 2};
  std::vector<int8_t *> input(1, nullptr);
  input[0] = input1.data();

  const int output_size = 1;
  int8_t output[1];
  std::vector<int> output_shape = {1, 1, 1, 1};
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

  std::vector<lite::Tensor *> inputs_tensor(1);
  inputs_tensor[0] = input_tensor1;

  std::vector<lite::Tensor *> outputs_tensor(1);
  lite::Tensor *output0_tensor = new lite::Tensor;
  output0_tensor->set_data(output);
  output0_tensor->set_shape(output_shape);
  output0_tensor->AddQuantParam(output_quant_arg);
  output0_tensor->set_data_type(tid_int8);
  outputs_tensor[0] = output0_tensor;

  CropParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_Crop;
  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  op_param.axis_ = 0;
  op_param.offset_[0] = 1;
  op_param.offset_size_ = 1;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_Crop};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor->shape();
  ASSERT_EQ(output_tensor_shape, output_shape);
  kernel->Run();

  std::vector<int8_t> except_result = {16};
  PrintData("output data", output, output_size);
  PrintData("output data shape", output_tensor_shape.data(), output_tensor_shape.size());
  ASSERT_EQ(0, CompareOutputData(output, except_result.data(), output_size, 0.000001));

  input_tensor1->set_data(nullptr);
  output0_tensor->set_data(nullptr);
  delete input_tensor1;
  delete output0_tensor;
  delete ctx;
}

TEST_F(TestCropInt8, crop_4d_axis1_offset0_quant0_thread0) {
  std::vector<int8_t> input1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<int> shape1 = {2, 2, 2, 2};
  std::vector<int8_t *> input(1, nullptr);
  input[0] = input1.data();

  const int output_size = 2;
  int8_t output[2];
  std::vector<int> output_shape = {2, 1, 1, 1};
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

  std::vector<lite::Tensor *> inputs_tensor(1);
  inputs_tensor[0] = input_tensor1;

  std::vector<lite::Tensor *> outputs_tensor(1);
  lite::Tensor *output0_tensor = new lite::Tensor;
  output0_tensor->set_data(output);
  output0_tensor->set_shape(output_shape);
  output0_tensor->AddQuantParam(output_quant_arg);
  output0_tensor->set_data_type(tid_int8);
  outputs_tensor[0] = output0_tensor;

  CropParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_Crop;
  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  op_param.axis_ = 1;
  op_param.offset_[0] = 1;
  op_param.offset_size_ = 1;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_Crop};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor->shape();
  ASSERT_EQ(output_tensor_shape, output_shape);
  kernel->Run();

  std::vector<int8_t> except_result = {8, 16};
  PrintData("output data", output, output_size);
  PrintData("output data shape", output_tensor_shape.data(), output_tensor_shape.size());
  ASSERT_EQ(0, CompareOutputData(output, except_result.data(), output_size, 0.000001));

  input_tensor1->set_data(nullptr);
  output0_tensor->set_data(nullptr);
  delete input_tensor1;
  delete output0_tensor;
  delete ctx;
}

TEST_F(TestCropInt8, crop_4d_axis1_offset1_quant0_thread0) {
  std::vector<int8_t> input1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<int> shape1 = {2, 2, 2, 2};
  std::vector<int8_t *> input(1, nullptr);
  input[0] = input1.data();

  const int output_size = 4;
  int8_t output[4];
  std::vector<int> output_shape = {1, 1, 2, 2};
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

  std::vector<lite::Tensor *> inputs_tensor(1);
  inputs_tensor[0] = input_tensor1;

  std::vector<lite::Tensor *> outputs_tensor(1);
  lite::Tensor *output0_tensor = new lite::Tensor;
  output0_tensor->set_data(output);
  output0_tensor->set_shape(output_shape);
  output0_tensor->AddQuantParam(output_quant_arg);
  output0_tensor->set_data_type(tid_int8);
  outputs_tensor[0] = output0_tensor;

  CropParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_Crop;
  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  op_param.axis_ = 0;
  op_param.offset_[0] = 1;
  op_param.offset_[1] = 1;
  op_param.offset_[2] = 0;
  op_param.offset_[3] = 0;
  op_param.offset_size_ = 4;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_Crop};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor->shape();
  ASSERT_EQ(output_tensor_shape, output_shape);
  kernel->Run();

  std::vector<int8_t> except_result = {13, 14, 15, 16};
  PrintData("output data", output, output_size);
  PrintData("output data shape", output_tensor_shape.data(), output_tensor_shape.size());
  ASSERT_EQ(0, CompareOutputData(output, except_result.data(), output_size, 0.000001));

  input_tensor1->set_data(nullptr);
  output0_tensor->set_data(nullptr);
  delete input_tensor1;
  delete output0_tensor;
  delete ctx;
}

TEST_F(TestCropInt8, crop_4d_axis1_offset1_quant1_thread0) {
  std::vector<int8_t> input1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<int> shape1 = {2, 2, 2, 2};
  std::vector<int8_t *> input(1, nullptr);
  input[0] = input1.data();

  const int output_size = 4;
  int8_t output[4];
  std::vector<int> output_shape = {1, 1, 2, 2};
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

  std::vector<lite::Tensor *> inputs_tensor(1);
  inputs_tensor[0] = input_tensor1;

  std::vector<lite::Tensor *> outputs_tensor(1);
  lite::Tensor *output0_tensor = new lite::Tensor;
  output0_tensor->set_data(output);
  output0_tensor->set_shape(output_shape);
  output0_tensor->AddQuantParam(output_quant_arg);
  output0_tensor->set_data_type(tid_int8);
  outputs_tensor[0] = output0_tensor;

  CropParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_Crop;
  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  op_param.axis_ = 0;
  op_param.offset_[0] = 1;
  op_param.offset_[1] = 1;
  op_param.offset_[2] = 0;
  op_param.offset_[3] = 0;
  op_param.offset_size_ = 4;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_Crop};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor->shape();
  ASSERT_EQ(output_tensor_shape, output_shape);
  kernel->Run();

  std::vector<int8_t> except_result = {7, 7, 8, 8};
  PrintData("output data", output, output_size);
  PrintData("output data shape", output_tensor_shape.data(), output_tensor_shape.size());
  ASSERT_EQ(0, CompareOutputData(output, except_result.data(), output_size, 0.000001));

  input_tensor1->set_data(nullptr);
  output0_tensor->set_data(nullptr);
  delete input_tensor1;
  delete output0_tensor;
  delete ctx;
}

TEST_F(TestCropInt8, crop_4d_axis0_offset0_quant0_thread2) {
  std::vector<int8_t> input1 = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                                45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64};
  std::vector<int> shape1 = {2, 8, 2, 2};
  std::vector<int8_t *> input(1, nullptr);
  input[0] = input1.data();

  const int output_size = 7;
  int8_t output[7];
  std::vector<int> output_shape = {1, 7, 1, 1};
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

  std::vector<lite::Tensor *> inputs_tensor(1);
  inputs_tensor[0] = input_tensor1;

  std::vector<lite::Tensor *> outputs_tensor(1);
  lite::Tensor *output0_tensor = new lite::Tensor;
  output0_tensor->set_data(output);
  output0_tensor->set_shape(output_shape);
  output0_tensor->AddQuantParam(output_quant_arg);
  output0_tensor->set_data_type(tid_int8);
  outputs_tensor[0] = output0_tensor;

  CropParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_Crop;
  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  op_param.axis_ = 0;
  op_param.offset_[0] = 1;
  op_param.offset_size_ = 1;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_Crop};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor->shape();
  ASSERT_EQ(output_tensor_shape, output_shape);
  kernel->Run();

  std::vector<int8_t> except_result = {40, 44, 48, 52, 56, 60, 64};
  PrintData("output data", output, output_size);
  PrintData("output data shape", output_tensor_shape.data(), output_tensor_shape.size());
  ASSERT_EQ(0, CompareOutputData(output, except_result.data(), output_size, 0.000001));

  input_tensor1->set_data(nullptr);
  output0_tensor->set_data(nullptr);
  delete input_tensor1;
  delete output0_tensor;
  delete ctx;
}

TEST_F(TestCropInt8, crop_4d_axis0_offset0_quant0_thread3) {
  std::vector<int8_t> input1 = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                                45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64};
  std::vector<int> shape1 = {2, 8, 2, 2};
  std::vector<int8_t *> input(1, nullptr);
  input[0] = input1.data();

  const int output_size = 7;
  int8_t output[7];
  std::vector<int> output_shape = {1, 7, 1, 1};
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

  std::vector<lite::Tensor *> inputs_tensor(1);
  inputs_tensor[0] = input_tensor1;

  std::vector<lite::Tensor *> outputs_tensor(1);
  lite::Tensor *output0_tensor = new lite::Tensor;
  output0_tensor->set_data(output);
  output0_tensor->set_shape(output_shape);
  output0_tensor->AddQuantParam(output_quant_arg);
  output0_tensor->set_data_type(tid_int8);
  outputs_tensor[0] = output0_tensor;

  CropParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_Crop;
  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 3;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  op_param.axis_ = 0;
  op_param.offset_[0] = 1;
  op_param.offset_size_ = 1;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_Crop};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor->shape();
  ASSERT_EQ(output_tensor_shape, output_shape);
  kernel->Run();

  std::vector<int8_t> except_result = {40, 44, 48, 52, 56, 60, 64};
  PrintData("output data", output, output_size);
  PrintData("output data shape", output_tensor_shape.data(), output_tensor_shape.size());
  ASSERT_EQ(0, CompareOutputData(output, except_result.data(), output_size, 0.000001));

  input_tensor1->set_data(nullptr);
  output0_tensor->set_data(nullptr);
  delete input_tensor1;
  delete output0_tensor;
  delete ctx;
}
}  // namespace mindspore
