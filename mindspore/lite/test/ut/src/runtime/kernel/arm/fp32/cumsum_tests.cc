/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "common/common_test.h"
#include "nnacl/cumsum_parameter.h"
#include "mindspore/lite/src/kernel_registry.h"

namespace mindspore {
class TestCumsum : public mindspore::CommonTest {
 public:
  TestCumsum() {}
};

TEST_F(TestCumsum, TestThread1) {
  lite::Tensor in_tensor0(kNumberTypeFloat32, {2, 3, 2});
  float input_data0[12] = {1, 1, 2, 2, 3, 3, 10, 10, 20, 20, 30, 30};
  in_tensor0.set_data(input_data0);
  lite::Tensor in_tensor1(kNumberTypeInt32, {1});
  int input_data1[1] = {1};  // axis 1
  in_tensor1.set_data(input_data1);
  std::vector<lite::Tensor *> inputs = {&in_tensor0, &in_tensor1};

  lite::Tensor out_tensor0(kNumberTypeFloat32, {2, 3, 2});
  float output_data0[12] = {0};
  out_tensor0.set_data(output_data0);
  std::vector<lite::Tensor *> outputs = {&out_tensor0};

  CumSumParameter *parameter = reinterpret_cast<CumSumParameter *>(malloc(sizeof(CumSumParameter)));
  parameter->op_parameter_.type_ = schema::PrimitiveType_CumSum;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->exclusive_ = false;
  parameter->reverse_ = false;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_CumSum};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  EXPECT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(parameter), ctx.get(), desc);
  EXPECT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);
  EXPECT_NEAR(1.0f, output_data0[0], 0.000001);
  EXPECT_NEAR(1.0f, output_data0[1], 0.000001);
  EXPECT_NEAR(3.0f, output_data0[2], 0.000001);
  EXPECT_NEAR(3.0f, output_data0[3], 0.000001);
  EXPECT_NEAR(6.0f, output_data0[4], 0.000001);
  EXPECT_NEAR(6.0f, output_data0[5], 0.000001);
  EXPECT_NEAR(10.0f, output_data0[6], 0.000001);
  EXPECT_NEAR(10.0f, output_data0[7], 0.000001);
  EXPECT_NEAR(30.0f, output_data0[8], 0.000001);
  EXPECT_NEAR(30.0f, output_data0[9], 0.000001);
  EXPECT_NEAR(60.0f, output_data0[10], 0.000001);
  EXPECT_NEAR(60.0f, output_data0[11], 0.000001);

  for (int i = 0; i < 12; ++i) {
    std::cout << output_data0[i] << " ";
  }
  std::cout << std::endl;
  out_tensor0.set_data(nullptr);
  in_tensor0.set_data(nullptr);
  in_tensor1.set_data(nullptr);
}

TEST_F(TestCumsum, TestExclusive) {
  lite::Tensor in_tensor0(kNumberTypeFloat32, {2, 3, 2});
  float input_data0[12] = {1, 1, 2, 2, 3, 3, 10, 10, 20, 20, 30, 30};
  in_tensor0.set_data(input_data0);
  lite::Tensor in_tensor1(kNumberTypeInt32, {1});
  int input_data1[1] = {1};  // axis 1
  in_tensor1.set_data(input_data1);
  std::vector<lite::Tensor *> inputs = {&in_tensor0, &in_tensor1};

  lite::Tensor out_tensor0(kNumberTypeFloat32, {2, 3, 2});
  float output_data0[12] = {0};
  out_tensor0.set_data(output_data0);
  std::vector<lite::Tensor *> outputs = {&out_tensor0};

  CumSumParameter *parameter = reinterpret_cast<CumSumParameter *>(malloc(sizeof(CumSumParameter)));
  parameter->op_parameter_.type_ = schema::PrimitiveType_CumSum;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->exclusive_ = true;
  parameter->reverse_ = false;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_CumSum};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  EXPECT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(parameter), ctx.get(), desc);
  EXPECT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);
  EXPECT_NEAR(0.0f, output_data0[0], 0.000001);
  EXPECT_NEAR(0.0f, output_data0[1], 0.000001);
  EXPECT_NEAR(1.0f, output_data0[2], 0.000001);
  EXPECT_NEAR(1.0f, output_data0[3], 0.000001);
  EXPECT_NEAR(3.0f, output_data0[4], 0.000001);
  EXPECT_NEAR(3.0f, output_data0[5], 0.000001);
  EXPECT_NEAR(0.0f, output_data0[6], 0.000001);
  EXPECT_NEAR(0.0f, output_data0[7], 0.000001);
  EXPECT_NEAR(10.0f, output_data0[8], 0.000001);
  EXPECT_NEAR(10.0f, output_data0[9], 0.000001);
  EXPECT_NEAR(30.0f, output_data0[10], 0.000001);
  EXPECT_NEAR(30.0f, output_data0[11], 0.000001);

  for (int i = 0; i < 12; ++i) {
    std::cout << output_data0[i] << " ";
  }
  out_tensor0.set_data(nullptr);
  in_tensor0.set_data(nullptr);
  in_tensor1.set_data(nullptr);
  delete kernel;
}

TEST_F(TestCumsum, TestReverse) {
  lite::Tensor in_tensor0(kNumberTypeFloat32, {2, 3, 2});
  float input_data0[12] = {1, 1, 2, 2, 3, 3, 10, 10, 20, 20, 30, 30};
  in_tensor0.set_data(input_data0);
  lite::Tensor in_tensor1(kNumberTypeInt32, {1});
  int input_data1[1] = {1};  // axis 1
  in_tensor1.set_data(input_data1);
  std::vector<lite::Tensor *> inputs = {&in_tensor0, &in_tensor1};

  lite::Tensor out_tensor0(kNumberTypeFloat32, {2, 3, 2});
  float output_data0[12] = {0};
  out_tensor0.set_data(output_data0);
  std::vector<lite::Tensor *> outputs = {&out_tensor0};

  CumSumParameter *parameter = reinterpret_cast<CumSumParameter *>(malloc(sizeof(CumSumParameter)));
  parameter->op_parameter_.type_ = schema::PrimitiveType_CumSum;
  parameter->op_parameter_.infer_flag_ = 1;
  parameter->exclusive_ = false;
  parameter->reverse_ = true;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_CumSum};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  EXPECT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(parameter), ctx.get(), desc);
  EXPECT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);
  EXPECT_NEAR(6.0f, output_data0[0], 0.000001);
  EXPECT_NEAR(6.0f, output_data0[1], 0.000001);
  EXPECT_NEAR(5.0f, output_data0[2], 0.000001);
  EXPECT_NEAR(5.0f, output_data0[3], 0.000001);
  EXPECT_NEAR(3.0f, output_data0[4], 0.000001);
  EXPECT_NEAR(3.0f, output_data0[5], 0.000001);
  EXPECT_NEAR(60.0f, output_data0[6], 0.000001);
  EXPECT_NEAR(60.0f, output_data0[7], 0.000001);
  EXPECT_NEAR(50.0f, output_data0[8], 0.000001);
  EXPECT_NEAR(50.0f, output_data0[9], 0.000001);
  EXPECT_NEAR(30.0f, output_data0[10], 0.000001);
  EXPECT_NEAR(30.0f, output_data0[11], 0.000001);

  for (int i = 0; i < 12; ++i) {
    std::cout << output_data0[i] << " ";
  }
  out_tensor0.set_data(nullptr);
  in_tensor0.set_data(nullptr);
  in_tensor1.set_data(nullptr);
  delete kernel;
}

TEST_F(TestCumsum, TestReverseExclusive) {
  lite::Tensor in_tensor0(kNumberTypeFloat32, {2, 3, 2});
  float input_data0[12] = {1, 1, 2, 2, 3, 3, 10, 10, 20, 20, 30, 30};
  in_tensor0.set_data(input_data0);
  lite::Tensor in_tensor1(kNumberTypeInt32, {1});
  int input_data1[1] = {1};  // axis 1
  in_tensor1.set_data(input_data1);
  std::vector<lite::Tensor *> inputs = {&in_tensor0, &in_tensor1};

  lite::Tensor out_tensor0(kNumberTypeFloat32, {2, 3, 2});
  float output_data0[12] = {0};
  out_tensor0.set_data(output_data0);
  std::vector<lite::Tensor *> outputs = {&out_tensor0};

  CumSumParameter *parameter = reinterpret_cast<CumSumParameter *>(malloc(sizeof(CumSumParameter)));
  parameter->op_parameter_.type_ = schema::PrimitiveType_CumSum;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->exclusive_ = true;
  parameter->reverse_ = true;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_CumSum};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  EXPECT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(parameter), ctx.get(), desc);
  EXPECT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);
  EXPECT_NEAR(5.0f, output_data0[0], 0.000001);
  EXPECT_NEAR(5.0f, output_data0[1], 0.000001);
  EXPECT_NEAR(3.0f, output_data0[2], 0.000001);
  EXPECT_NEAR(3.0f, output_data0[3], 0.000001);
  EXPECT_NEAR(0.0f, output_data0[4], 0.000001);
  EXPECT_NEAR(0.0f, output_data0[5], 0.000001);
  EXPECT_NEAR(50.0f, output_data0[6], 0.000001);
  EXPECT_NEAR(50.0f, output_data0[7], 0.000001);
  EXPECT_NEAR(30.0f, output_data0[8], 0.000001);
  EXPECT_NEAR(30.0f, output_data0[9], 0.000001);
  EXPECT_NEAR(0.0f, output_data0[10], 0.000001);
  EXPECT_NEAR(0.0f, output_data0[11], 0.000001);

  for (int i = 0; i < 12; ++i) {
    std::cout << output_data0[i] << " ";
  }
  out_tensor0.set_data(nullptr);
  in_tensor0.set_data(nullptr);
  in_tensor1.set_data(nullptr);
  delete kernel;
}

TEST_F(TestCumsum, TestIntRank2) {
  lite::Tensor in_tensor0(kNumberTypeInt32, {1, 6});
  int input_data0[6] = {1, 2, 3, 4, 5, 6};
  in_tensor0.set_data(input_data0);
  lite::Tensor in_tensor1(kNumberTypeInt32, {1});
  int input_data1[1] = {1};  // axis 1
  in_tensor1.set_data(input_data1);
  std::vector<lite::Tensor *> inputs = {&in_tensor0, &in_tensor1};

  lite::Tensor out_tensor0(kNumberTypeInt32, {1, 6});
  int output_data0[6] = {0};
  out_tensor0.set_data(output_data0);
  std::vector<lite::Tensor *> outputs = {&out_tensor0};

  CumSumParameter *parameter = reinterpret_cast<CumSumParameter *>(malloc(sizeof(CumSumParameter)));
  parameter->op_parameter_.type_ = schema::PrimitiveType_CumSum;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->exclusive_ = false;
  parameter->reverse_ = false;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_CumSum};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  EXPECT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(parameter), ctx.get(), desc);
  EXPECT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);
  EXPECT_EQ(1, output_data0[0]);
  EXPECT_EQ(3, output_data0[1]);
  EXPECT_EQ(6, output_data0[2]);
  EXPECT_EQ(10, output_data0[3]);
  EXPECT_EQ(15, output_data0[4]);
  EXPECT_EQ(21, output_data0[5]);

  for (int i = 0; i < 6; ++i) {
    std::cout << output_data0[i] << " ";
  }
  out_tensor0.set_data(nullptr);
  in_tensor0.set_data(nullptr);
  in_tensor1.set_data(nullptr);
  delete kernel;
}

TEST_F(TestCumsum, TestIntRank2Thread2) {
  lite::Tensor in_tensor0(kNumberTypeInt32, {1, 6});
  int input_data0[6] = {1, 2, 3, 4, 5, 6};
  in_tensor0.set_data(input_data0);
  lite::Tensor in_tensor1(kNumberTypeInt32, {1});
  int input_data1[1] = {1};  // axis 1
  in_tensor1.set_data(input_data1);
  std::vector<lite::Tensor *> inputs = {&in_tensor0, &in_tensor1};

  lite::Tensor out_tensor0(kNumberTypeInt32, {1, 6});
  int output_data0[6] = {0};
  out_tensor0.set_data(output_data0);
  std::vector<lite::Tensor *> outputs = {&out_tensor0};

  CumSumParameter *parameter = reinterpret_cast<CumSumParameter *>(malloc(sizeof(CumSumParameter)));
  parameter->op_parameter_.type_ = schema::PrimitiveType_CumSum;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->exclusive_ = false;
  parameter->reverse_ = false;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_CumSum};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  EXPECT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(parameter), ctx.get(), desc);
  EXPECT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);
  EXPECT_EQ(1, output_data0[0]);
  EXPECT_EQ(3, output_data0[1]);
  EXPECT_EQ(6, output_data0[2]);
  EXPECT_EQ(10, output_data0[3]);
  EXPECT_EQ(15, output_data0[4]);
  EXPECT_EQ(21, output_data0[5]);

  for (int i = 0; i < 6; ++i) {
    std::cout << output_data0[i] << " ";
  }
  out_tensor0.set_data(nullptr);
  in_tensor0.set_data(nullptr);
  in_tensor1.set_data(nullptr);
  delete kernel;
}

TEST_F(TestCumsum, TestIntRank2Thread4) {
  lite::Tensor in_tensor0(kNumberTypeInt32, {1, 6});
  int input_data0[6] = {1, 2, 3, 4, 5, 6};
  in_tensor0.set_data(input_data0);
  lite::Tensor in_tensor1(kNumberTypeInt32, {1});
  int input_data1[1] = {1};  // axis 1
  in_tensor1.set_data(input_data1);
  std::vector<lite::Tensor *> inputs = {&in_tensor0, &in_tensor1};

  lite::Tensor out_tensor0(kNumberTypeInt32, {1, 6});
  int output_data0[6] = {0};
  out_tensor0.set_data(output_data0);
  std::vector<lite::Tensor *> outputs = {&out_tensor0};

  CumSumParameter *parameter = reinterpret_cast<CumSumParameter *>(malloc(sizeof(CumSumParameter)));
  parameter->op_parameter_.type_ = schema::PrimitiveType_CumSum;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->exclusive_ = false;
  parameter->reverse_ = false;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_CumSum};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  EXPECT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ctx->thread_num_ = 4;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(parameter), ctx.get(), desc);
  EXPECT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);
  EXPECT_EQ(1, output_data0[0]);
  EXPECT_EQ(3, output_data0[1]);
  EXPECT_EQ(6, output_data0[2]);
  EXPECT_EQ(10, output_data0[3]);
  EXPECT_EQ(15, output_data0[4]);
  EXPECT_EQ(21, output_data0[5]);

  for (int i = 0; i < 6; ++i) {
    std::cout << output_data0[i] << " ";
  }
  out_tensor0.set_data(nullptr);
  in_tensor0.set_data(nullptr);
  in_tensor1.set_data(nullptr);
  delete kernel;
}

}  // namespace mindspore
