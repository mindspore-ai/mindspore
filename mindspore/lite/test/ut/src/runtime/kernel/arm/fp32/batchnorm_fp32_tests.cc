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
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/nnacl/fp32/batchnorm_fp32.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/src/lite_kernel.h"

namespace mindspore {
class TestBatchnormFp32 : public mindspore::CommonTest {
 public:
  TestBatchnormFp32() {}
};

TEST_F(TestBatchnormFp32, BNTest) {
  std::vector<float> in_data = {-11.18675,  11.433986,  11.386012, 11.245945,   -2.7614849, 14.692399,
                                -1.1983503, -6.6790967, 6.383416,  -13.3213005, -8.693595,  9.476344};
  std::vector<float> in_data1 = {12.352293, 5.122387, 14.249514};
  std::vector<float> in_data2 = {14.632595, 0.70900035, 11.179003};

  BatchNormParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_BatchNorm;
  op_param.epsilon_ = 0.001f;

  lite::Tensor input0_tensor(kNumberTypeFloat32, {1, 2, 2, 3});
  lite::Tensor input1_tensor(kNumberTypeFloat32, {3});
  lite::Tensor input2_tensor(kNumberTypeFloat32, {3});
  input0_tensor.set_data(in_data.data());
  input1_tensor.set_data(in_data1.data());
  input2_tensor.set_data(in_data2.data());
  std::vector<lite::Tensor *> inputs_tensor = {&input0_tensor, &input1_tensor, &input2_tensor};

  std::vector<float> output(12);
  std::vector<float> corr_out = {-6.1533737, 7.4904885,  -0.8563998, -0.289212,  -9.356432,  0.13245535,
                                 -3.5422924, -14.005781, -2.3525476, -6.7113695, -16.396551, -1.4275324};

  lite::Tensor output0_tensor(kNumberTypeFloat32, {1, 2, 2, 3});
  output0_tensor.set_data(output.data());
  std::vector<lite::Tensor *> outputs_tensor = {&output0_tensor};

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_BatchNorm};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  lite::InnerContext ctx;
  ctx.thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx.Init());
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), &ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor.shape();
  kernel->Run();

  printf("==================output data=================\n");
  for (int i = 0; i < output0_tensor.ElementsNum(); i++) {
    std::cout << output[i] << " ,";
  }
  std::cout << std::endl;
  ASSERT_EQ(0, CompareOutputData(output.data(), corr_out.data(), output0_tensor.ElementsNum(), 0.001));

  input0_tensor.set_data(nullptr);
  input1_tensor.set_data(nullptr);
  input2_tensor.set_data(nullptr);
  output0_tensor.set_data(nullptr);
}

TEST_F(TestBatchnormFp32, FusedBNTest) {
  std::vector<float> in_data = {-7.400094, 11.37495, 2.0271842,  5.5954003,  13.255154, 4.6289115,
                                9.591311,  8.699771, -12.226144, -6.1819935, 6.957936,  -8.70818};
  std::vector<float> scale = {13.323708, 14.0656395, 12.634319};
  std::vector<float> offset = {27.888096, 24.533648, 15.335093};
  std::vector<float> mean = {11.5127125, 0.47681615, 5.851508};
  std::vector<float> var = {1.270583, 13.005714, 6.089223};

  BatchNormParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_BatchNorm;
  op_param.epsilon_ = 0.001f;

  lite::Tensor input0(kNumberTypeFloat32, {1, 2, 2, 3});
  lite::Tensor input1(kNumberTypeFloat32, {3});
  lite::Tensor input2(kNumberTypeFloat32, {3});
  lite::Tensor input3(kNumberTypeFloat32, {3});
  lite::Tensor input4(kNumberTypeFloat32, {3});
  input0.set_data(in_data.data());
  input1.set_data(scale.data());
  input2.set_data(offset.data());
  input3.set_data(mean.data());
  input4.set_data(var.data());
  std::vector<lite::Tensor *> inputs_tensor = {&input0, &input1, &input2, &input3, &input4};

  std::vector<float> output(12);
  std::vector<float> corr_out = {-195.5765, 67.03745, -4.243883,  -42.028015, 74.37044, 9.075897,
                                 5.1857452, 56.60399, -77.215096, -181.18402, 49.81066, -59.204563};

  lite::Tensor output0(kNumberTypeFloat32, {1, 2, 2, 3});
  output0.set_data(output.data());
  std::vector<lite::Tensor *> outputs_tensor = {&output0};

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_FusedBatchNorm};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  lite::InnerContext ctx;
  ctx.thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx.Init());
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), &ctx, desc);
  ASSERT_NE(kernel, nullptr);
  kernel->Run();

  printf("==================output data=================\n");
  for (int i = 0; i < output0.ElementsNum(); i++) {
    std::cout << output[i] << " ,";
  }
  std::cout << std::endl;
  ASSERT_EQ(0, CompareOutputData(output.data(), corr_out.data(), output0.ElementsNum(), 0.001));

  input0.set_data(nullptr);
  input1.set_data(nullptr);
  input2.set_data(nullptr);
  input3.set_data(nullptr);
  input4.set_data(nullptr);
  output0.set_data(nullptr);
}

TEST_F(TestBatchnormFp32, easyTest) {
  std::vector<float> in_data = {1, 4, 2, 5, 3, 6, -1, -4, -2, -5, -3, -6};
  std::vector<float> in_data1 = {0.1, 0.6};
  std::vector<float> in_data2 = {3, 4};

  BatchNormParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_BatchNorm;
  op_param.epsilon_ = 0.001f;

  lite::Tensor input0(kNumberTypeFloat32, {1, 1, 6, 2});
  lite::Tensor input1(kNumberTypeFloat32, {2});
  lite::Tensor input2(kNumberTypeFloat32, {2});
  input0.set_data(in_data.data());
  input1.set_data(in_data1.data());
  input2.set_data(in_data2.data());
  std::vector<lite::Tensor *> inputs_tensor = {&input0, &input1, &input2};

  std::vector<float> output(12);
  std::vector<float> corr_out = {0.519529, 1.69979,  1.09678,  2.19973,  1.67404,  2.69966,
                                 -0.63498, -2.29971, -1.21223, -2.79965, -1.78949, -3.29959};

  lite::Tensor output0(kNumberTypeFloat32, {1, 1, 6, 2});
  output0.set_data(output.data());
  std::vector<lite::Tensor *> outputs_tensor = {&output0};

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_BatchNorm};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  lite::InnerContext ctx;
  ctx.thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx.Init());
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), &ctx, desc);
  ASSERT_NE(kernel, nullptr);
  kernel->Run();

  printf("==================output data=================\n");
  for (int i = 0; i < output0.ElementsNum(); i++) {
    std::cout << output[i] << " ,";
  }
  std::cout << std::endl;
  ASSERT_EQ(0, CompareOutputData(output.data(), corr_out.data(), output0.ElementsNum(), 0.001));

  input0.set_data(nullptr);
  input1.set_data(nullptr);
  input2.set_data(nullptr);
  output0.set_data(nullptr);
}

}  // namespace mindspore
