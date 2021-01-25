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
#include <vector>
#include "common/common_test.h"
#include "src/common/file_utils.h"
#include "mindspore/lite/src/runtime/kernel/arm/fp32_grad/deconvolution_grad_filter.h"
#include "mindspore/lite/nnacl/conv_parameter.h"
#include "mindspore/lite/src/kernel_registry.h"

namespace mindspore {
class TestDeConvolutionGradFp32 : public mindspore::CommonTest {
 public:
  TestDeConvolutionGradFp32() {}
};

TEST_F(TestDeConvolutionGradFp32, DeConvFp32FilterGrad) {
  // prepare stage
  auto conv_param = static_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  ASSERT_NE(conv_param, nullptr);

  conv_param->input_batch_ = 2;
  conv_param->input_h_ = 32;
  conv_param->input_w_ = 32;
  conv_param->input_channel_ = 3;

  conv_param->output_batch_ = 2;
  conv_param->output_h_ = 63;
  conv_param->output_w_ = 63;
  conv_param->output_channel_ = 9;

  conv_param->kernel_h_ = 3;
  conv_param->kernel_w_ = 3;

  conv_param->stride_h_ = 2;
  conv_param->stride_w_ = 2;

  conv_param->dilation_h_ = 1;
  conv_param->dilation_w_ = 1;

  conv_param->pad_u_ = 1;
  conv_param->pad_l_ = 1;
  conv_param->pad_r_ = 1;
  conv_param->pad_d_ = 1;

  conv_param->group_ = 1;
  conv_param->act_type_ = ActType_No;
  conv_param->thread_num_ = 1;

  size_t dy_size;
  std::string dy_path = "./test_data/deconv/deconvfp32_dy_2_9_63_63.bin";
  auto dy_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(dy_path.c_str(), &dy_size));
  ASSERT_NE(dy_data, nullptr);
  std::vector<int> dim_dy({2, 63, 63, 9});
  lite::Tensor dy_tensor(TypeId::kNumberTypeFloat32, dim_dy);
  dy_tensor.set_data(dy_data);

  size_t output_data_size =
    conv_param->output_channel_ * conv_param->kernel_h_ * conv_param->kernel_w_ * conv_param->input_channel_;

  size_t input_size;
  std::string input_path = "./test_data/deconv/deconvfp32_input0_2_3_32_32.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  ASSERT_NE(input_data, nullptr);
  std::vector<int> dim_x({2, 32, 32, 3});
  lite::Tensor x_tensor(TypeId::kNumberTypeFloat32, dim_x);
  x_tensor.set_data(input_data);

  auto dw_data = new float[output_data_size];
  ASSERT_NE(dw_data, nullptr);
  std::vector<int> dim_dw({3, 3, 3, 9});
  lite::Tensor dw_tensor(TypeId::kNumberTypeFloat32, dim_dw);
  dw_tensor.set_data(dw_data);
  std::vector<lite::Tensor *> inputs = {&dy_tensor, &x_tensor};
  std::vector<lite::Tensor *> outputs = {&dw_tensor};

  lite::InnerContext context;
  context.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, context.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_DeConv2DGradFilter};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(conv_param), &context, desc);
  ASSERT_NE(kernel, nullptr);
  mindspore::kernel::LiteKernel::AllocWorkspace(kernel->workspace_size());

  // warm up loop
  for (int i = 0; i < 3; i++) {
    kernel->Run();
  }

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;
  int loop_count = 100;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    kernel->Run();
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  std::string output_path = "./test_data/deconv/deconvfp32_dw_9_3_3_3.bin";
  auto res = CompareRelativeOutput(dw_data, output_path);

  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] dy_data;
  delete[] dw_data;
  delete kernel;
  // delete conv_param;
  dw_tensor.set_data(nullptr);
  x_tensor.set_data(nullptr);
  dy_tensor.set_data(nullptr);
  mindspore::kernel::LiteKernel::FreeWorkspace();
  MS_LOG(INFO) << "TestDeConvolutionGradFp32 Filter Grad passed";
}

TEST_F(TestDeConvolutionGradFp32, DeConvFp32Dilation2FilterGrad) {
  // prepare stage
  auto conv_param = static_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  ASSERT_NE(conv_param, nullptr);

  conv_param->input_batch_ = 2;
  conv_param->input_h_ = 32;
  conv_param->input_w_ = 32;
  conv_param->input_channel_ = 3;

  conv_param->output_batch_ = 2;
  conv_param->output_h_ = 65;
  conv_param->output_w_ = 65;
  conv_param->output_channel_ = 9;

  conv_param->kernel_h_ = 3;
  conv_param->kernel_w_ = 3;

  conv_param->stride_h_ = 2;
  conv_param->stride_w_ = 2;

  conv_param->dilation_h_ = 2;
  conv_param->dilation_w_ = 2;

  conv_param->pad_u_ = 1;
  conv_param->pad_l_ = 1;
  conv_param->pad_r_ = 1;
  conv_param->pad_d_ = 1;

  conv_param->group_ = 1;
  conv_param->act_type_ = ActType_No;
  conv_param->thread_num_ = 1;

  size_t dy_size;
  std::string dy_path = "./test_data/deconv/deconvfp32_dy_d2_2_9_65_65.bin";
  auto dy_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(dy_path.c_str(), &dy_size));
  ASSERT_NE(dy_data, nullptr);
  std::vector<int> dim_dy({2, 65, 65, 9});
  lite::Tensor dy_tensor(TypeId::kNumberTypeFloat32, dim_dy);
  dy_tensor.set_data(dy_data);

  size_t output_data_size =
    conv_param->output_channel_ * conv_param->kernel_h_ * conv_param->kernel_w_ * conv_param->input_channel_;

  size_t input_size;
  std::string input_path = "./test_data/deconv/deconvfp32_input0_d2_2_3_32_32.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  ASSERT_NE(input_data, nullptr);
  std::vector<int> dim_x({2, 32, 32, 3});
  lite::Tensor x_tensor(TypeId::kNumberTypeFloat32, dim_x);
  x_tensor.set_data(input_data);

  auto dw_data = new float[output_data_size];
  ASSERT_NE(dw_data, nullptr);
  std::vector<int> dim_dw({9, 3, 3, 3});
  lite::Tensor dw_tensor(TypeId::kNumberTypeFloat32, dim_dw);
  dw_tensor.set_data(dw_data);
  std::vector<lite::Tensor *> inputs = {&dy_tensor, &x_tensor};
  std::vector<lite::Tensor *> outputs = {&dw_tensor};

  lite::InnerContext context;
  context.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, context.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_DeConv2DGradFilter};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(conv_param), &context, desc);
  ASSERT_NE(kernel, nullptr);
  mindspore::kernel::LiteKernel::AllocWorkspace(kernel->workspace_size());
  for (int i = 0; i < 3; i++) {
  }

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;
  int loop_count = 100;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    kernel->Run();
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  std::string output_path = "./test_data/deconv/deconvfp32_dw_d2_9_3_3_3.bin";
  auto res = CompareRelativeOutput(dw_data, output_path);

  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] dy_data;
  delete[] dw_data;
  delete kernel;
  // delete conv_param;
  dw_tensor.set_data(nullptr);
  x_tensor.set_data(nullptr);
  dy_tensor.set_data(nullptr);
  mindspore::kernel::LiteKernel::FreeWorkspace();
  MS_LOG(INFO) << "TestDeConvolutionGradFp32 Filter Grad passed";
}

TEST_F(TestDeConvolutionGradFp32, DeConvFp32Dilation2Group3FilterGrad) {
  // prepare stage
  auto conv_param = static_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  ASSERT_NE(conv_param, nullptr);

  conv_param->input_batch_ = 2;
  conv_param->input_h_ = 32;
  conv_param->input_w_ = 32;
  conv_param->input_channel_ = 3;

  conv_param->output_batch_ = 2;
  conv_param->output_h_ = 65;
  conv_param->output_w_ = 65;
  conv_param->output_channel_ = 9;

  conv_param->kernel_h_ = 3;
  conv_param->kernel_w_ = 3;

  conv_param->stride_h_ = 2;
  conv_param->stride_w_ = 2;

  conv_param->dilation_h_ = 2;
  conv_param->dilation_w_ = 2;

  conv_param->pad_u_ = 1;
  conv_param->pad_l_ = 1;
  conv_param->pad_r_ = 1;
  conv_param->pad_d_ = 1;

  conv_param->group_ = 3;
  conv_param->act_type_ = ActType_No;
  conv_param->thread_num_ = 1;

  size_t dy_size;
  std::string dy_path = "./test_data/deconv/deconvfp32_dy_d2_g3_2_9_65_65.bin";
  auto dy_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(dy_path.c_str(), &dy_size));
  ASSERT_NE(dy_data, nullptr);
  std::vector<int> dim_dy({2, 65, 65, 9});
  lite::Tensor dy_tensor(TypeId::kNumberTypeFloat32, dim_dy);
  dy_tensor.set_data(dy_data);

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;
  size_t output_data_size =
    conv_param->output_channel_ * conv_param->kernel_h_ * conv_param->kernel_w_ * conv_param->input_channel_;

  size_t input_size;
  std::string input_path = "./test_data/deconv/deconvfp32_input0_d2_g3_2_3_32_32.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  ASSERT_NE(input_data, nullptr);
  std::vector<int> dim_x({2, 32, 32, 3});
  lite::Tensor x_tensor(TypeId::kNumberTypeFloat32, dim_x);
  x_tensor.set_data(input_data);

  auto dw_data = new float[output_data_size];
  ASSERT_NE(dw_data, nullptr);
  std::vector<int> dim_dw({3, 3, 3, 3});
  lite::Tensor dw_tensor(TypeId::kNumberTypeFloat32, dim_dw);
  dw_tensor.set_data(dw_data);
  std::vector<lite::Tensor *> inputs = {&dy_tensor, &x_tensor};
  std::vector<lite::Tensor *> outputs = {&dw_tensor};

  lite::InnerContext context;
  context.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, context.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_DeConv2DGradFilter};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(conv_param), &context, desc);
  ASSERT_NE(kernel, nullptr);
  mindspore::kernel::LiteKernel::AllocWorkspace(kernel->workspace_size());

  // warm up loop
  for (int i = 0; i < 3; i++) {
    kernel->Run();
  }

  int loop_count = 100;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    kernel->Run();
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  std::string output_path = "./test_data/deconv/deconvfp32_dw_d2_g3_3_3_3_3.bin";
  auto res = CompareRelativeOutput(dw_data, output_path);

  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] dy_data;
  delete[] dw_data;
  delete kernel;
  // delete conv_param;
  dw_tensor.set_data(nullptr);
  x_tensor.set_data(nullptr);
  dy_tensor.set_data(nullptr);
  mindspore::kernel::LiteKernel::FreeWorkspace();
  MS_LOG(INFO) << "TestDeConvolutionGradFp32 Filter Grad passed";
}

TEST_F(TestDeConvolutionGradFp32, DeConvFp32Dilation2Group3Stride1FilterGrad) {
  // prepare stage
  auto conv_param = static_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  ASSERT_NE(conv_param, nullptr);

  conv_param->input_batch_ = 2;
  conv_param->input_h_ = 32;
  conv_param->input_w_ = 32;
  conv_param->input_channel_ = 3;

  conv_param->output_batch_ = 2;
  conv_param->output_h_ = 34;
  conv_param->output_w_ = 34;
  conv_param->output_channel_ = 9;

  conv_param->kernel_h_ = 3;
  conv_param->kernel_w_ = 3;

  conv_param->stride_h_ = 1;
  conv_param->stride_w_ = 1;

  conv_param->dilation_h_ = 2;
  conv_param->dilation_w_ = 2;

  conv_param->pad_u_ = 1;
  conv_param->pad_l_ = 1;
  conv_param->pad_r_ = 1;
  conv_param->pad_d_ = 1;

  conv_param->group_ = 3;
  conv_param->act_type_ = ActType_No;
  conv_param->thread_num_ = 1;

  size_t dy_size;
  std::string dy_path = "./test_data/deconv/deconvfp32_dy_d2_g3_s1_2_9_34_34.bin";
  auto dy_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(dy_path.c_str(), &dy_size));
  ASSERT_NE(dy_data, nullptr);
  std::vector<int> dim_dy({2, 34, 34, 9});
  lite::Tensor dy_tensor(TypeId::kNumberTypeFloat32, dim_dy);
  dy_tensor.set_data(dy_data);

  size_t output_data_size =
    conv_param->output_channel_ * conv_param->kernel_h_ * conv_param->kernel_w_ * conv_param->input_channel_;

  size_t input_size;
  std::string input_path = "./test_data/deconv/deconvfp32_input0_d2_g3_s1_2_3_32_32.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  ASSERT_NE(input_data, nullptr);
  std::vector<int> dim_x({2, 32, 32, 3});
  lite::Tensor x_tensor(TypeId::kNumberTypeFloat32, dim_x);
  x_tensor.set_data(input_data);

  auto dw_data = new float[output_data_size];
  ASSERT_NE(dw_data, nullptr);
  std::vector<int> dim_dw({3, 3, 3, 3});
  lite::Tensor dw_tensor(TypeId::kNumberTypeFloat32, dim_dw);
  dw_tensor.set_data(dw_data);
  std::vector<lite::Tensor *> inputs = {&dy_tensor, &x_tensor};
  std::vector<lite::Tensor *> outputs = {&dw_tensor};

  lite::InnerContext context;
  context.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, context.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_DeConv2DGradFilter};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(conv_param), &context, desc);
  ASSERT_NE(kernel, nullptr);
  mindspore::kernel::LiteKernel::AllocWorkspace(kernel->workspace_size());

  // warm up loop
  for (int i = 0; i < 3; i++) {
    kernel->Run();
  }

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;
  int loop_count = 100;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    kernel->Run();
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  std::string output_path = "./test_data/deconv/deconvfp32_dw_d2_g3_s1_3_3_3_3.bin";
  auto res = CompareRelativeOutput(dw_data, output_path);

  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] dy_data;
  delete[] dw_data;
  delete kernel;
  // delete conv_param;
  dw_tensor.set_data(nullptr);
  x_tensor.set_data(nullptr);
  dy_tensor.set_data(nullptr);
  mindspore::kernel::LiteKernel::FreeWorkspace();
  MS_LOG(INFO) << "TestDeConvolutionGradFp32 Filter Grad passed";
}

TEST_F(TestDeConvolutionGradFp32, DeConvFp32Dilation2Group2Stride2FilterGrad) {
  // prepare stage
  auto conv_param = static_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  ASSERT_NE(conv_param, nullptr);

  conv_param->input_batch_ = 2;
  conv_param->input_h_ = 32;
  conv_param->input_w_ = 32;
  conv_param->input_channel_ = 4;

  conv_param->output_batch_ = 2;
  conv_param->output_h_ = 65;
  conv_param->output_w_ = 65;
  conv_param->output_channel_ = 12;

  conv_param->kernel_h_ = 3;
  conv_param->kernel_w_ = 3;

  conv_param->stride_h_ = 2;
  conv_param->stride_w_ = 2;

  conv_param->dilation_h_ = 2;
  conv_param->dilation_w_ = 2;

  conv_param->pad_u_ = 1;
  conv_param->pad_l_ = 1;
  conv_param->pad_r_ = 1;
  conv_param->pad_d_ = 1;

  conv_param->group_ = 2;
  conv_param->act_type_ = ActType_No;
  conv_param->thread_num_ = 1;

  size_t dy_size;
  std::string dy_path = "./test_data/deconv/deconvfp32_dy_d2_g2_s2_2_12_65_65.bin";
  auto dy_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(dy_path.c_str(), &dy_size));
  ASSERT_NE(dy_data, nullptr);
  std::vector<int> dim_dy({2, 65, 65, 12});
  lite::Tensor dy_tensor(TypeId::kNumberTypeFloat32, dim_dy);
  dy_tensor.set_data(dy_data);

  size_t output_data_size =
    conv_param->output_channel_ * conv_param->kernel_h_ * conv_param->kernel_w_ * conv_param->input_channel_;

  size_t input_size;
  std::string input_path = "./test_data/deconv/deconvfp32_input0_d2_g2_s2_2_4_32_32.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  ASSERT_NE(input_data, nullptr);
  std::vector<int> dim_x({2, 32, 32, 4});
  lite::Tensor x_tensor(TypeId::kNumberTypeFloat32, dim_x);
  x_tensor.set_data(input_data);

  auto dw_data = new float[output_data_size];
  ASSERT_NE(dw_data, nullptr);
  std::vector<int> dim_dw({6, 3, 3, 4});
  lite::Tensor dw_tensor(TypeId::kNumberTypeFloat32, dim_dw);
  dw_tensor.set_data(dw_data);
  std::vector<lite::Tensor *> inputs = {&dy_tensor, &x_tensor};
  std::vector<lite::Tensor *> outputs = {&dw_tensor};

  lite::InnerContext context;
  context.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, context.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_DeConv2DGradFilter};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(conv_param), &context, desc);
  ASSERT_NE(kernel, nullptr);
  mindspore::kernel::LiteKernel::AllocWorkspace(kernel->workspace_size());

  // warm up loop
  for (int i = 0; i < 3; i++) {
    kernel->Run();
  }

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;
  int loop_count = 100;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    kernel->Run();
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  std::string output_path = "./test_data/deconv/deconvfp32_dw_d2_g2_s2_6_4_3_3.bin";
  auto res = CompareRelativeOutput(dw_data, output_path);

  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] dy_data;
  delete[] dw_data;
  delete kernel;
  // delete conv_param;
  dw_tensor.set_data(nullptr);
  x_tensor.set_data(nullptr);
  dy_tensor.set_data(nullptr);
  mindspore::kernel::LiteKernel::FreeWorkspace();
  MS_LOG(INFO) << "TestDeConvolutionGradFp32 Filter Grad passed";
}

TEST_F(TestDeConvolutionGradFp32, DeConvFp32Dilation2Group12Stride2FilterGrad) {
  // prepare stage
  auto conv_param = static_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  ASSERT_NE(conv_param, nullptr);

  conv_param->input_batch_ = 2;
  conv_param->input_h_ = 32;
  conv_param->input_w_ = 32;
  conv_param->input_channel_ = 12;

  conv_param->output_batch_ = 2;
  conv_param->output_h_ = 65;
  conv_param->output_w_ = 65;
  conv_param->output_channel_ = 12;

  conv_param->kernel_h_ = 3;
  conv_param->kernel_w_ = 3;

  conv_param->stride_h_ = 2;
  conv_param->stride_w_ = 2;

  conv_param->dilation_h_ = 2;
  conv_param->dilation_w_ = 2;

  conv_param->pad_u_ = 1;
  conv_param->pad_l_ = 1;
  conv_param->pad_r_ = 1;
  conv_param->pad_d_ = 1;

  conv_param->group_ = 12;
  conv_param->act_type_ = ActType_No;
  conv_param->thread_num_ = 1;

  size_t dy_size;
  std::string dy_path = "./test_data/deconv/deconvfp32_dy_d2_g12_s2_2_12_65_65.bin";
  auto dy_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(dy_path.c_str(), &dy_size));
  ASSERT_NE(dy_data, nullptr);
  std::vector<int> dim_dy({2, 65, 65, 12});
  lite::Tensor dy_tensor(TypeId::kNumberTypeFloat32, dim_dy);
  dy_tensor.set_data(dy_data);

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;
  size_t output_data_size =
    conv_param->output_channel_ * conv_param->kernel_h_ * conv_param->kernel_w_ * conv_param->input_channel_;

  size_t input_size;
  std::string input_path = "./test_data/deconv/deconvfp32_input0_d2_g12_s2_2_12_32_32.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  ASSERT_NE(input_data, nullptr);
  std::vector<int> dim_x({2, 32, 32, 12});
  lite::Tensor x_tensor(TypeId::kNumberTypeFloat32, dim_x);
  x_tensor.set_data(input_data);

  auto dw_data = new float[output_data_size];
  ASSERT_NE(dw_data, nullptr);
  std::vector<int> dim_dw({1, 3, 3, 12});
  lite::Tensor dw_tensor(TypeId::kNumberTypeFloat32, dim_dw);
  dw_tensor.set_data(dw_data);
  std::vector<lite::Tensor *> inputs = {&dy_tensor, &x_tensor};
  std::vector<lite::Tensor *> outputs = {&dw_tensor};

  lite::InnerContext context;
  context.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, context.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_DeConv2DGradFilter};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(conv_param), &context, desc);
  ASSERT_NE(kernel, nullptr);

  mindspore::kernel::LiteKernel::AllocWorkspace(kernel->workspace_size());

  // warm up loop
  for (int i = 0; i < 3; i++) {
    kernel->Run();
  }

  int loop_count = 100;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    kernel->Run();
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  std::string output_path = "./test_data/deconv/deconvfp32_dw_d2_g12_s2_12_1_3_3.bin";
  auto res = CompareRelativeOutput(dw_data, output_path);

  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] dy_data;
  delete[] dw_data;
  delete kernel;
  // delete conv_param;
  dw_tensor.set_data(nullptr);
  x_tensor.set_data(nullptr);
  dy_tensor.set_data(nullptr);
  mindspore::kernel::LiteKernel::FreeWorkspace();
  MS_LOG(INFO) << "TestDeConvolutionGradFp32 Filter Grad passed";
}

}  // namespace mindspore
