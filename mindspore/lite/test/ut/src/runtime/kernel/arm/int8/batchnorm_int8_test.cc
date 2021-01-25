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
#include "mindspore/lite/nnacl/batchnorm_parameter.h"
#include "mindspore/lite/nnacl/int8/batchnorm_int8.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/src/lite_kernel.h"

namespace mindspore {
class TestBatchnormInt8 : public mindspore::CommonTest {
 public:
  TestBatchnormInt8() {}
};

TEST_F(TestBatchnormInt8, FusedTest) {
  std::vector<int8_t> in_data = {11, 41, 21, 51, 31, 61, -11, -41, -21, -51, -31, -61};
  std::vector<int8_t> in_data1 = {4, 4};
  std::vector<int8_t> in_data2 = {8, 33};
  std::vector<int8_t> in_data3 = {35, 55};
  std::vector<int8_t> in_data4 = {2, 3};
  std::vector<lite::Tensor *> inputs_tensor;
  std::vector<lite::Tensor *> outputs_tensor;

  BatchNormParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_FusedBatchNorm;
  op_param.epsilon_ = 0.001f;
  op_param.fused_ = true;

  std::vector<int> shape = {1, 1, 6, 2};

  lite::QuantArg input_quant_arg;
  input_quant_arg.scale = 0.1;
  input_quant_arg.zeroPoint = 1;
  lite::QuantArg input_quant_arg_1;
  input_quant_arg_1.scale = 0.5;
  input_quant_arg_1.zeroPoint = 2;
  lite::QuantArg input_quant_arg_2;
  input_quant_arg_2.scale = 0.02;
  input_quant_arg_2.zeroPoint = 3;
  lite::QuantArg input_quant_arg_3;
  input_quant_arg_3.scale = 0.5;
  input_quant_arg_3.zeroPoint = 15;
  lite::QuantArg input_quant_arg_4;
  input_quant_arg_4.scale = 0.25;
  input_quant_arg_4.zeroPoint = 1;
  lite::QuantArg output_quant_arg;
  output_quant_arg.scale = 0.8;
  output_quant_arg.zeroPoint = 0;

  lite::Tensor input0_tensor;
  lite::Tensor input1_tensor;
  lite::Tensor input2_tensor;
  lite::Tensor input3_tensor;
  lite::Tensor input4_tensor;
  inputs_tensor.push_back(&input0_tensor);
  inputs_tensor.push_back(&input1_tensor);
  inputs_tensor.push_back(&input2_tensor);
  inputs_tensor.push_back(&input3_tensor);
  inputs_tensor.push_back(&input4_tensor);
  input0_tensor.set_data(in_data.data());
  input1_tensor.set_data(in_data1.data());
  input2_tensor.set_data(in_data2.data());
  input3_tensor.set_data(in_data3.data());
  input4_tensor.set_data(in_data4.data());
  input0_tensor.set_shape(shape);
  input1_tensor.set_shape({2});
  input2_tensor.set_shape({2});
  input3_tensor.set_shape({2});
  input4_tensor.set_shape({2});
  input0_tensor.AddQuantParam(input_quant_arg);
  input1_tensor.AddQuantParam(input_quant_arg_1);
  input2_tensor.AddQuantParam(input_quant_arg_2);
  input3_tensor.AddQuantParam(input_quant_arg_3);
  input4_tensor.AddQuantParam(input_quant_arg_4);

  std::vector<int8_t> output(12);
  // std::vector<int8_t> corr_out = {-18, -22, -16, -21, -14, -19, -22, -34, -24, -35, -26, -36 };
  std::vector<int8_t> corr_out = {-22, -28, -20, -26, -17, -24, -28, -42, -30, -44, -33, -46};
  lite::Tensor output0_tensor;
  outputs_tensor.push_back(&output0_tensor);
  output0_tensor.set_data(output.data());
  output0_tensor.set_shape(shape);
  output0_tensor.AddQuantParam(output_quant_arg);

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_FusedBatchNorm};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  lite::InnerContext ctx;
  ctx.thread_num_ = 3;
  ASSERT_EQ(lite::RET_OK, ctx.Init());
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), &ctx, desc);
  ASSERT_NE(kernel, nullptr);

  auto output_tensor_shape = output0_tensor.shape();
  kernel->Run();

  printf("==================output data=================\n");
  for (int i = 0; i < output0_tensor.ElementsNum(); i++) {
    printf("%d, ", output[i]);
  }
  std::cout << std::endl;
  ASSERT_EQ(0, CompareOutputData(output.data(), corr_out.data(), output0_tensor.ElementsNum(), 0.001));

  input0_tensor.set_data(nullptr);
  input1_tensor.set_data(nullptr);
  input2_tensor.set_data(nullptr);
  input3_tensor.set_data(nullptr);
  input4_tensor.set_data(nullptr);
  output0_tensor.set_data(nullptr);
  MS_LOG(INFO) << "TestBathNormFp32 accuracy passed";
}

TEST_F(TestBatchnormInt8, BNTest) {
  std::vector<int8_t> in_data = {11, 41, 21, 51, 31, 61, -11, -41, -21, -51, -31, -61};
  std::vector<int8_t> in_data1 = {4, 14};
  std::vector<int8_t> in_data2 = {29, 39};
  std::vector<lite::Tensor *> inputs_tensor;
  std::vector<lite::Tensor *> outputs_tensor;

  BatchNormParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_BatchNorm;
  op_param.epsilon_ = 0.001f;
  op_param.fused_ = false;

  std::vector<int> shape = {1, 1, 6, 2};

  lite::QuantArg input_quant_arg;
  input_quant_arg.scale = 0.1;
  input_quant_arg.zeroPoint = 1;
  lite::QuantArg input_quant_arg_1;
  input_quant_arg_1.scale = 0.05;
  input_quant_arg_1.zeroPoint = 2;
  lite::QuantArg input_quant_arg_2;
  input_quant_arg_2.scale = 0.1;
  input_quant_arg_2.zeroPoint = -1;
  lite::QuantArg output_quant_arg;
  output_quant_arg.scale = 0.5;
  output_quant_arg.zeroPoint = 0;

  lite::Tensor input0_tensor;
  lite::Tensor input1_tensor;
  lite::Tensor input2_tensor;
  inputs_tensor.push_back(&input0_tensor);
  inputs_tensor.push_back(&input1_tensor);
  inputs_tensor.push_back(&input2_tensor);
  input0_tensor.set_data(in_data.data());
  input1_tensor.set_data(in_data1.data());
  input2_tensor.set_data(in_data2.data());
  input0_tensor.set_shape(shape);
  input1_tensor.set_shape({2});
  input2_tensor.set_shape({2});
  input0_tensor.AddQuantParam(input_quant_arg);
  input1_tensor.AddQuantParam(input_quant_arg_1);
  input2_tensor.AddQuantParam(input_quant_arg_2);

  std::vector<int8_t> output(12);
  std::vector<int8_t> corr_out = {1, 3, 2, 4, 3, 5, -2, -5, -3, -6, -4, -7};

  lite::Tensor output0_tensor;
  outputs_tensor.push_back(&output0_tensor);
  output0_tensor.set_data(output.data());
  output0_tensor.set_shape(shape);
  output0_tensor.AddQuantParam(output_quant_arg);

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_BatchNorm};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  lite::InnerContext ctx;
  ctx.thread_num_ = 3;
  ASSERT_EQ(lite::RET_OK, ctx.Init());
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), &ctx, desc);
  ASSERT_NE(kernel, nullptr);

  auto output_tensor_shape = output0_tensor.shape();
  kernel->Run();

  printf("==================output data=================\n");
  for (int i = 0; i < output0_tensor.ElementsNum(); i++) {
    printf("%d, ", output[i]);
  }
  std::cout << std::endl;
  ASSERT_EQ(0, CompareOutputData(output.data(), corr_out.data(), output0_tensor.ElementsNum(), 0.001));

  input0_tensor.set_data(nullptr);
  input1_tensor.set_data(nullptr);
  input2_tensor.set_data(nullptr);
  output0_tensor.set_data(nullptr);
  MS_LOG(INFO) << "TestBathNormFp32 accuracy passed";
}

}  // namespace mindspore
