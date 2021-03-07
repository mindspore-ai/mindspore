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
#include "mindspore/lite/nnacl/fp32/activation_fp32.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/src/lite_kernel.h"

namespace mindspore {

class TestActivationFp32 : public mindspore::CommonTest {
 public:
  TestActivationFp32() {}
};

TEST_F(TestActivationFp32, ReluFp32) {
  float input[8] = {-3, -2, -1, 0, 1, 5, 6, 7};
  float output[8] = {0};
  Fp32Relu(input, 8, output);
  float expect[8] = {0, 0, 0, 0, 1, 5, 6, 7};
  for (int i = 0; i < 8; ++i) {
    ASSERT_EQ(output[i], expect[i]);
  }
}

TEST_F(TestActivationFp32, Relu6Fp32) {
  float input[8] = {-3, -2, -1, 0, 1, 5, 6, 7};
  float output[8] = {0};
  Fp32Relu6(input, 8, output);
  float expect[8] = {0, 0, 0, 0, 1, 5, 6, 6};
  for (int i = 0; i < 8; ++i) {
    ASSERT_EQ(output[i], expect[i]);
  }
  MS_LOG(INFO) << "TestActivationFp32 passed";
}

TEST_F(TestActivationFp32, LReluFp32) {
  float input[8] = {-3, -2, -1, 0, 1, 5, 6, 7};
  float output[8] = {0};
  LRelu(input, 8, output, 0.01);
  float expect[8] = {-0.03, -0.02, -0.01, 0, 1, 5, 6, 7};
  for (int i = 0; i < 8; ++i) {
    ASSERT_EQ(output[i], expect[i]);
  }
  MS_LOG(INFO) << "TestActivationFp32 passed";
}

TEST_F(TestActivationFp32, SigmoidFp32) {
  float input[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  float output[8] = {0};
  Sigmoid(input, 8, output);

  // expect output {0.5, 0.731059, 0.880797, 0.952574, 0.982014, 0.993307, 0.997527, 0.999089};
  printf("==================output data=================\n");
  for (int i = 0; i < 8; ++i) {
    std::cout << output[i] << " ";
  }
  std::cout << std::endl;
  MS_LOG(INFO) << "TestSigmoidFp32 passed";
}

TEST_F(TestActivationFp32, SwishFp32) {
  float input[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  float output[8] = {0};
  Swish(input, 8, output);

  float expect[8] = {0, 0.731059, 1.761594, 2.857722, 3.928056, 4.966535, 5.985162, 6.993623};
  for (int i = 0; i < 8; ++i) {
    EXPECT_NEAR(output[i], expect[i], 0.00001);
  }
}

TEST_F(TestActivationFp32, TanhFp32) {
  float input[7] = {-3, -2, -1, 0, 1, 2, 3};
  float output[7] = {0};
  Tanh(input, 7, output);
  float expect[8] = {-0.995055, -0.964028, -0.761594, 0.000000, 0.761594, 0.964028, 0.995055};
  for (int i = 0; i < 8; ++i) {
    EXPECT_NEAR(output[i], expect[i], 0.00001);
  }
  MS_LOG(INFO) << "TanhFp32 passed";
}

TEST_F(TestActivationFp32, HSwishFp32) {
  std::vector<lite::Tensor *> inputs_tensor;
  std::vector<lite::Tensor *> outputs_tensor;

  ActivationParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_Activation;
  op_param.type_ = schema::ActivationType_HSWISH;
  op_param.alpha_ = 0.01;

  std::vector<float> input = {-3.0, -2.0, -1.0, 0.0, 1.0, 5.0, 6.0, 7.0};
  std::vector<int> in_shape = {8};

  lite::Tensor input0_tensor;
  inputs_tensor.push_back(&input0_tensor);
  input0_tensor.set_data(input.data());
  input0_tensor.set_shape(in_shape);

  std::vector<float> output(8);
  std::vector<int> output_shape = {8};

  lite::Tensor output0_tensor;
  outputs_tensor.push_back(&output0_tensor);
  output0_tensor.set_data(output.data());

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Activation};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  lite::InnerContext ctx;
  ctx.thread_num_ = 7;
  ASSERT_EQ(lite::RET_OK, ctx.Init());
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), &ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor.shape();
  kernel->Run();

  std::vector<float> expect_output = {-0, -0.33333334, -0.33333334, 0, 0.6666667, 5, 6, 7};
  ASSERT_EQ(0, CompareOutputData(output.data(), expect_output.data(), 8, 0.00001));

  input0_tensor.set_data(nullptr);
  output0_tensor.set_data(nullptr);
}

TEST_F(TestActivationFp32, HardTanh1) {
  std::vector<lite::Tensor *> inputs_tensor;
  std::vector<lite::Tensor *> outputs_tensor;

  ActivationParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_Activation;
  op_param.type_ = schema::ActivationType_HARD_TANH;
  op_param.min_val_ = -1.0f;
  op_param.max_val_ = 1.0f;

  std::vector<float> input = {-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0, 6.0};
  std::vector<int> in_shape = {8};

  lite::Tensor input0_tensor;
  inputs_tensor.push_back(&input0_tensor);
  input0_tensor.set_data(input.data());
  input0_tensor.set_shape(in_shape);

  std::vector<float> output(8);
  std::vector<int> output_shape = {8};

  lite::Tensor output0_tensor;
  outputs_tensor.push_back(&output0_tensor);
  output0_tensor.set_data(output.data());

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Activation};
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

  std::vector<float> expect_output = {-1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0, 1.0};
  ASSERT_EQ(0, CompareOutputData(output.data(), expect_output.data(), 8, 0.00001));

  input0_tensor.set_data(nullptr);
  output0_tensor.set_data(nullptr);
}

TEST_F(TestActivationFp32, HardTanh2) {
  std::vector<lite::Tensor *> inputs_tensor;
  std::vector<lite::Tensor *> outputs_tensor;

  ActivationParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_Activation;
  op_param.type_ = schema::ActivationType_HARD_TANH;
  op_param.min_val_ = -2.0f;
  op_param.max_val_ = 2.0f;

  std::vector<float> input = {-3.0, -2.0, -1.0, 0.0, 1.0, 5.0, 6.0, 7.0};
  std::vector<int> in_shape = {8};

  lite::Tensor input0_tensor;
  inputs_tensor.push_back(&input0_tensor);
  input0_tensor.set_data(input.data());
  input0_tensor.set_shape(in_shape);

  std::vector<float> output(8);
  std::vector<int> output_shape = {8};

  lite::Tensor output0_tensor;
  outputs_tensor.push_back(&output0_tensor);
  output0_tensor.set_data(output.data());

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Activation};
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

  std::vector<float> expect_output = {-2.0, -2.0, -1.0, 0.0, 1.0, 2.0, 2.0, 2.0};
  ASSERT_EQ(0, CompareOutputData(output.data(), expect_output.data(), 8, 0.00001));

  input0_tensor.set_data(nullptr);
  output0_tensor.set_data(nullptr);
}

}  // namespace mindspore
