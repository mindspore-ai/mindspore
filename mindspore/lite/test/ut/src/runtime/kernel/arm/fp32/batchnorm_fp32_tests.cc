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
#include "nnacl/batchnorm_parameter.h"
#include "nnacl/nnacl_manager.h"

namespace mindspore {
class TestBatchnormFp32 : public mindspore::CommonTest {
 public:
  TestBatchnormFp32() {}
};

TEST_F(TestBatchnormFp32, BNTest) {
  float in_data0[] = {-11.18675,  11.433986,  11.386012, 11.245945,   -2.7614849, 14.692399,
                      -1.1983503, -6.6790967, 6.383416,  -13.3213005, -8.693595,  9.476344};
  float in_data1[] = {12.352293, 5.122387, 14.249514};
  float in_data2[] = {14.632595, 0.70900035, 11.179003};

  lite::Tensor input0_tensor(kNumberTypeFloat32, {1, 2, 2, 3});
  lite::Tensor input1_tensor(kNumberTypeFloat32, {3});
  lite::Tensor input2_tensor(kNumberTypeFloat32, {3});
  memcpy(input0_tensor.MutableData(), in_data0, input0_tensor.Size());
  memcpy(input1_tensor.MutableData(), in_data1, input1_tensor.Size());
  memcpy(input2_tensor.MutableData(), in_data2, input2_tensor.Size());
  std::vector<lite::Tensor *> inputs_tensor = {&input0_tensor, &input1_tensor, &input2_tensor};

  lite::Tensor output0(kNumberTypeFloat32, {1, 2, 2, 3});
  output0.MallocData();
  std::vector<lite::Tensor *> outputs_tensor = {&output0};

  BatchNormParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_BatchNorm;
  op_param.epsilon_ = 0.001f;
  op_param.op_parameter_.thread_num_ = 2;

  lite::InnerContext ctx;
  ASSERT_EQ(lite::RET_OK, ctx.Init());

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, NHWC, op_param.op_parameter_.type_};
  auto kernel = nnacl::NNACLKernelRegistry(&op_param.op_parameter_, inputs_tensor, outputs_tensor, &ctx, desc);
  ASSERT_NE(kernel, nullptr);

  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  float correct[] = {-6.1533737, 7.4904885,  -0.8563998, -0.289212,  -9.356432,  0.13245535,
                     -3.5422924, -14.005781, -2.3525476, -6.7113695, -16.396551, -1.4275324};
  float *output_data = reinterpret_cast<float *>(output0.data());
  ASSERT_EQ(0, CompareOutputData(output_data, correct, output0.ElementsNum(), 0.001));

  kernel->set_parameter(nullptr);
  delete kernel;
}

TEST_F(TestBatchnormFp32, FusedBNTest) {
  float in_data[] = {-7.400094, 11.37495, 2.0271842,  5.5954003,  13.255154, 4.6289115,
                     9.591311,  8.699771, -12.226144, -6.1819935, 6.957936,  -8.70818};
  float scale[] = {13.323708, 14.0656395, 12.634319};
  float offset[] = {27.888096, 24.533648, 15.335093};
  float mean[] = {11.5127125, 0.47681615, 5.851508};
  float var[] = {1.270583, 13.005714, 6.089223};

  BatchNormParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_FusedBatchNorm;
  op_param.epsilon_ = 0.001f;
  op_param.op_parameter_.thread_num_ = 2;

  lite::Tensor input0(kNumberTypeFloat32, {1, 2, 2, 3});
  lite::Tensor input1(kNumberTypeFloat32, {3});
  lite::Tensor input2(kNumberTypeFloat32, {3});
  lite::Tensor input3(kNumberTypeFloat32, {3});
  lite::Tensor input4(kNumberTypeFloat32, {3});
  memcpy(input0.MutableData(), in_data, input0.Size());
  memcpy(input1.MutableData(), scale, input1.Size());
  memcpy(input2.MutableData(), offset, input2.Size());
  memcpy(input3.MutableData(), mean, input3.Size());
  memcpy(input4.MutableData(), var, input4.Size());
  std::vector<lite::Tensor *> inputs_tensor = {&input0, &input1, &input2, &input3, &input4};

  lite::Tensor output0(kNumberTypeFloat32, {1, 2, 2, 3});
  output0.MallocData();
  std::vector<lite::Tensor *> outputs_tensor = {&output0};

  lite::InnerContext ctx;
  ASSERT_EQ(lite::RET_OK, ctx.Init());

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, NHWC, op_param.op_parameter_.type_};
  auto kernel = nnacl::NNACLKernelRegistry(&op_param.op_parameter_, inputs_tensor, outputs_tensor, &ctx, desc);
  ASSERT_NE(kernel, nullptr);

  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  float correct[] = {-195.5765, 67.03745, -4.243883,  -42.028015, 74.37044, 9.075897,
                     5.1857452, 56.60399, -77.215096, -181.18402, 49.81066, -59.204563};
  float *output_data = reinterpret_cast<float *>(output0.data());
  ASSERT_EQ(0, CompareOutputData(output_data, correct, output0.ElementsNum(), 0.001));

  kernel->set_parameter(nullptr);
  delete kernel;
}

TEST_F(TestBatchnormFp32, easyTest) {
  float in_data0[] = {1, 4, 2, 5, 3, 6, -1, -4, -2, -5, -3, -6};
  float in_data1[] = {0.1, 0.6};
  float in_data2[] = {3, 4};

  BatchNormParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_BatchNorm;
  op_param.epsilon_ = 0.001f;
  op_param.op_parameter_.thread_num_ = 2;

  lite::Tensor input0(kNumberTypeFloat32, {1, 1, 6, 2});
  lite::Tensor input1(kNumberTypeFloat32, {2});
  lite::Tensor input2(kNumberTypeFloat32, {2});
  memcpy(input0.MutableData(), in_data0, input0.Size());
  memcpy(input1.MutableData(), in_data1, input1.Size());
  memcpy(input2.MutableData(), in_data2, input2.Size());
  std::vector<lite::Tensor *> inputs_tensor = {&input0, &input1, &input2};

  lite::Tensor output0(kNumberTypeFloat32, {1, 1, 6, 2});
  output0.MallocData();
  std::vector<lite::Tensor *> outputs_tensor = {&output0};

  lite::InnerContext ctx;
  ASSERT_EQ(lite::RET_OK, ctx.Init());

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, NHWC, op_param.op_parameter_.type_};
  auto kernel = nnacl::NNACLKernelRegistry(&op_param.op_parameter_, inputs_tensor, outputs_tensor, &ctx, desc);
  ASSERT_NE(kernel, nullptr);

  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  float correct[] = {0.5195, 1.6998, 1.0968, 2.1997, 1.674, 2.7, -0.6349, -2.2997, -1.212, -2.7996, -1.789, -3.299};
  float *output_data = reinterpret_cast<float *>(output0.data());
  ASSERT_EQ(0, CompareOutputData(output_data, correct, output0.ElementsNum(), 0.001));

  kernel->set_parameter(nullptr);
  delete kernel;
}
}  // namespace mindspore
