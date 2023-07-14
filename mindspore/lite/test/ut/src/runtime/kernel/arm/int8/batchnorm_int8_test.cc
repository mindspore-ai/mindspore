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
#include "nnacl/batchnorm_parameter.h"
#include "nnacl/int8/batchnorm_int8.h"
#include "mindspore/lite/src/litert/kernel_registry.h"
#include "mindspore/lite/src/executor/kernel_exec.h"

namespace mindspore {
class TestBatchnormInt8 : public mindspore::CommonTest {
 public:
  TestBatchnormInt8() {}
};

TEST_F(TestBatchnormInt8, FusedTest) {
  lite::LiteQuantParam input_quant_arg;
  input_quant_arg.scale = 0.1;
  input_quant_arg.zeroPoint = 1;
  lite::LiteQuantParam input_quant_arg_1;
  input_quant_arg_1.scale = 0.5;
  input_quant_arg_1.zeroPoint = 2;
  lite::LiteQuantParam input_quant_arg_2;
  input_quant_arg_2.scale = 0.02;
  input_quant_arg_2.zeroPoint = 3;
  lite::LiteQuantParam input_quant_arg_3;
  input_quant_arg_3.scale = 0.5;
  input_quant_arg_3.zeroPoint = 15;
  lite::LiteQuantParam input_quant_arg_4;
  input_quant_arg_4.scale = 0.25;
  input_quant_arg_4.zeroPoint = 1;
  lite::LiteQuantParam output_quant_arg;
  output_quant_arg.scale = 0.8;
  output_quant_arg.zeroPoint = 0;

  int8_t in_data0[] = {11, 41, 21, 51, 31, 61, -11, -41, -21, -51, -31, -61};
  int8_t in_data1[] = {4, 4};
  int8_t in_data2[] = {8, 33};
  int8_t in_data3[] = {35, 55};
  int8_t in_data4[] = {2, 3};

  lite::Tensor input0(kNumberTypeInt8, {1, 1, 6, 2});
  lite::Tensor input1(kNumberTypeInt8, {2});
  lite::Tensor input2(kNumberTypeInt8, {2});
  lite::Tensor input3(kNumberTypeInt8, {2});
  lite::Tensor input4(kNumberTypeInt8, {2});
  memcpy(input0.MutableData(), in_data0, input0.Size());
  memcpy(input1.MutableData(), in_data1, input1.Size());
  memcpy(input2.MutableData(), in_data2, input2.Size());
  memcpy(input3.MutableData(), in_data3, input3.Size());
  memcpy(input4.MutableData(), in_data4, input4.Size());
  input0.AddQuantParam(input_quant_arg);
  input1.AddQuantParam(input_quant_arg_1);
  input2.AddQuantParam(input_quant_arg_2);
  input3.AddQuantParam(input_quant_arg_3);
  input4.AddQuantParam(input_quant_arg_4);
  std::vector<lite::Tensor *> inputs_tensor = {&input0, &input1, &input2, &input3, &input4};

  lite::Tensor output0_tensor(kNumberTypeInt8, {1, 1, 6, 2});
  output0_tensor.MallocData();
  output0_tensor.AddQuantParam(output_quant_arg);
  std::vector<lite::Tensor *> outputs_tensor = {&output0_tensor};

  int thread_number = 3;
  BatchNormParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_FusedBatchNorm;
  op_param.epsilon_ = 0.001f;
  op_param.op_parameter_.thread_num_ = thread_number;

  lite::InnerContext ctx;
  ctx.thread_num_ = thread_number;
  ASSERT_EQ(lite::RET_OK, ctx.Init());

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, NHWC, schema::PrimitiveType_FusedBatchNorm};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), &ctx, desc);
  ASSERT_NE(kernel, nullptr);

  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  int8_t correct[] = {-22, -28, -20, -26, -17, -24, -28, -42, -30, -44, -33, -46};
  int8_t *output_data = reinterpret_cast<int8_t *>(output0_tensor.data());
  ASSERT_EQ(0, CompareOutputData(output_data, correct, output0_tensor.ElementsNum(), 0.001));

  kernel->set_parameter(nullptr);
  delete kernel;
}

TEST_F(TestBatchnormInt8, BNTest) {
  lite::LiteQuantParam input_quant_arg;
  input_quant_arg.scale = 0.1;
  input_quant_arg.zeroPoint = 1;
  lite::LiteQuantParam input_quant_arg_1;
  input_quant_arg_1.scale = 0.05;
  input_quant_arg_1.zeroPoint = 2;
  lite::LiteQuantParam input_quant_arg_2;
  input_quant_arg_2.scale = 0.1;
  input_quant_arg_2.zeroPoint = -1;
  lite::LiteQuantParam output_quant_arg;
  output_quant_arg.scale = 0.5;
  output_quant_arg.zeroPoint = 0;

  int8_t in_data0[] = {11, 41, 21, 51, 31, 61, -11, -41, -21, -51, -31, -61};
  int8_t in_data1[] = {4, 14};
  int8_t in_data2[] = {29, 39};

  lite::Tensor input0(kNumberTypeInt8, {1, 1, 6, 2});
  lite::Tensor input1(kNumberTypeInt8, {2});
  lite::Tensor input2(kNumberTypeInt8, {2});
  memcpy(input0.MutableData(), in_data0, input0.Size());
  memcpy(input1.MutableData(), in_data1, input1.Size());
  memcpy(input2.MutableData(), in_data2, input2.Size());
  input0.AddQuantParam(input_quant_arg);
  input1.AddQuantParam(input_quant_arg_1);
  input2.AddQuantParam(input_quant_arg_2);
  std::vector<lite::Tensor *> inputs_tensor = {&input0, &input1, &input2};

  lite::Tensor output0_tensor(kNumberTypeInt8, {1, 1, 6, 2});
  output0_tensor.MallocData();
  output0_tensor.AddQuantParam(output_quant_arg);
  std::vector<lite::Tensor *> outputs_tensor = {&output0_tensor};

  int thread_number = 3;
  BatchNormParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_BatchNorm;
  op_param.epsilon_ = 0.001f;
  op_param.op_parameter_.thread_num_ = thread_number;

  lite::InnerContext ctx;
  ctx.thread_num_ = thread_number;
  ASSERT_EQ(lite::RET_OK, ctx.Init());

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, NHWC, schema::PrimitiveType_BatchNorm};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), &ctx, desc);
  ASSERT_NE(kernel, nullptr);

  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  int8_t corr_out[] = {1, 3, 2, 4, 3, 5, -2, -5, -3, -6, -4, -7};
  int8_t *output_data = reinterpret_cast<int8_t *>(output0_tensor.data());
  ASSERT_EQ(0, CompareOutputData(output_data, corr_out, output0_tensor.ElementsNum(), 0.001));

  kernel->set_parameter(nullptr);
  delete kernel;
}
}  // namespace mindspore
