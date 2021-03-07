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
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "src/common/file_utils.h"
#include "mindspore/lite/src/runtime/kernel/arm/fp32_grad/convolution.h"
#include "mindspore/lite/src/runtime/kernel/arm/fp32_grad/convolution_grad_filter.h"
#include "mindspore/lite/src/runtime/kernel/arm/fp32_grad/convolution_grad_input.h"
#include "mindspore/lite/nnacl/conv_parameter.h"
#include "mindspore/lite/src/kernel_registry.h"

namespace mindspore {
class TestConvolutionGradFp32 : public mindspore::CommonTest {
 public:
  TestConvolutionGradFp32() {}
};

void InitConvParamGroup1FP32(ConvParameter *conv_param) {
  conv_param->input_batch_ = 1;
  conv_param->input_h_ = 28;
  conv_param->input_w_ = 28;
  conv_param->input_channel_ = 3;

  conv_param->output_batch_ = 1;
  conv_param->output_h_ = 28;
  conv_param->output_w_ = 28;
  conv_param->output_channel_ = 32;

  conv_param->kernel_h_ = 3;
  conv_param->kernel_w_ = 3;

  conv_param->stride_h_ = 1;
  conv_param->stride_w_ = 1;

  conv_param->dilation_h_ = 1;
  conv_param->dilation_w_ = 1;

  conv_param->pad_u_ = 1;
  conv_param->pad_l_ = 1;

  conv_param->group_ = 1;
  conv_param->act_type_ = ActType_No;
  conv_param->thread_num_ = 1;
}

void InitConvParamGroup3FP32(ConvParameter *conv_param) {
  InitConvParamGroup1FP32(conv_param);
  conv_param->group_ = 3;
  conv_param->output_channel_ = 18;
}

void InitConvParamGroup3Dilation2FP32(ConvParameter *conv_param) {
  InitConvParamGroup3FP32(conv_param);
  conv_param->dilation_h_ = 2;
  conv_param->dilation_w_ = 2;
  conv_param->output_h_ = 26;
  conv_param->output_w_ = 26;
}

TEST_F(TestConvolutionGradFp32, ConvFp32FilterGrad) {
  // prepare stage
  auto conv_param = static_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  ASSERT_NE(conv_param, nullptr);

  InitConvParamGroup1FP32(conv_param);
  size_t dy_size;
  std::string dy_path = "./test_data/conv/convfp32_dy_1_28_28_32.bin";
  auto dy_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(dy_path.c_str(), &dy_size));
  ASSERT_NE(dy_data, nullptr);
  std::vector<int> dim_dy({1, 28, 28, 32});
  lite::Tensor dy_tensor(TypeId::kNumberTypeFloat32, dim_dy);
  dy_tensor.set_data(dy_data);

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;
  size_t output_data_size =
    conv_param->output_channel_ * conv_param->kernel_h_ * conv_param->kernel_w_ * conv_param->input_channel_;

  size_t input_size;
  std::string input_path = "./test_data/conv/convfp32_x_1_28_28_3.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  ASSERT_NE(input_data, nullptr);
  std::vector<int> dim_x({1, 28, 28, 3});
  lite::Tensor x_tensor(TypeId::kNumberTypeFloat32, dim_x);
  x_tensor.set_data(input_data);

  auto dw_data = new float[output_data_size];
  ASSERT_NE(dw_data, nullptr);
  std::vector<int> dim_dw({32, 3, 3, 3});
  lite::Tensor dw_tensor(TypeId::kNumberTypeFloat32, dim_dw);
  dw_tensor.set_data(dw_data);
  std::vector<lite::Tensor *> inputs = {&dy_tensor, &x_tensor};
  std::vector<lite::Tensor *> outputs = {&dw_tensor};

  lite::InnerContext context;
  context.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, context.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_Conv2DBackpropFilterFusion};
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

  std::string output_path = "./test_data/conv/convfp32_dw_32_3_3_3.bin";
  auto res = CompareRelativeOutput(dw_data, output_path);

  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] dy_data;
  delete[] dw_data;
  mindspore::kernel::LiteKernel::FreeWorkspace();
  delete kernel;
  // delete conv_param;
  dw_tensor.set_data(nullptr);
  x_tensor.set_data(nullptr);
  dy_tensor.set_data(nullptr);
  MS_LOG(INFO) << "TestConvolutionGradFp32 Filter Grad passed";
}

TEST_F(TestConvolutionGradFp32, ConvFp32InputGrad) {
  // prepare stage
  auto conv_param = static_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  ASSERT_NE(conv_param, nullptr);

  InitConvParamGroup1FP32(conv_param);
  size_t dy_size;
  std::string dy_path = "./test_data/conv/convfp32_dy_1_28_28_32.bin";
  auto dy_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(dy_path.c_str(), &dy_size));
  std::vector<int> dim_dy({1, 28, 28, 32});
  lite::Tensor dy_tensor(TypeId::kNumberTypeFloat32, dim_dy);
  dy_tensor.set_data(dy_data);

  size_t w_size;
  std::string w_path = "./test_data/conv/convfp32_w_32_3_3_3.bin";
  auto w_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(w_path.c_str(), &w_size));
  std::vector<int> dim_dw({32, 3, 3, 3});
  lite::Tensor w_tensor(TypeId::kNumberTypeFloat32, dim_dw);
  w_tensor.set_data(w_data);

  size_t output_data_size =
    conv_param->input_batch_ * conv_param->input_h_ * conv_param->input_w_ * conv_param->input_channel_;
  auto dx_data = new float[output_data_size];
  ASSERT_NE(dx_data, nullptr);
  std::vector<int> dim_dx({1, 28, 28, 3});
  lite::Tensor dx_tensor(TypeId::kNumberTypeFloat32, dim_dx);
  dx_tensor.set_data(dx_data);

  std::vector<lite::Tensor *> inputs = {&dy_tensor, &w_tensor};
  std::vector<lite::Tensor *> outputs = {&dx_tensor};
  // runtime part

  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;

  lite::InnerContext context;
  context.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, context.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_Conv2DBackpropInputFusion};
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

  std::string output_path = "./test_data/conv/convfp32_dx_1_28_28_3.bin";
  auto res = CompareRelativeOutput(dx_data, output_path);
  EXPECT_EQ(res, 0);
  delete[] dx_data;
  delete[] w_data;
  delete[] dy_data;
  w_tensor.set_data(nullptr);
  dy_tensor.set_data(nullptr);
  dx_tensor.set_data(nullptr);
  mindspore::kernel::LiteKernel::FreeWorkspace();
  delete kernel;
  // delete conv_param;

  MS_LOG(INFO) << "TestConvolutionGradFp32 Filter Grad passed";
}

TEST_F(TestConvolutionGradFp32, ConvFp32GroupFilterGrad) {
  // prepare stage
  auto conv_param = static_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  ASSERT_NE(conv_param, nullptr);

  InitConvParamGroup3FP32(conv_param);
  size_t dy_size;
  std::string dy_path = "./test_data/conv/convfp32_dy_g3_1_28_28_18.bin";
  auto dy_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(dy_path.c_str(), &dy_size));
  std::vector<int> dim_dy({1, 28, 28, 18});
  lite::Tensor dy_tensor(TypeId::kNumberTypeFloat32, dim_dy);
  dy_tensor.set_data(dy_data);

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;
  size_t output_data_size = conv_param->output_channel_ * conv_param->kernel_h_ * conv_param->kernel_w_ *
                            conv_param->input_channel_ / conv_param->group_;

  size_t input_size;
  std::string input_path = "./test_data/conv/convfp32_x_g3_1_28_28_3.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  std::vector<int> dim_x({1, 28, 28, 3});
  lite::Tensor x_tensor(TypeId::kNumberTypeFloat32, dim_x);
  x_tensor.set_data(input_data);

  auto dw_data = new float[output_data_size];
  ASSERT_NE(dw_data, nullptr);
  std::vector<int> dim_dw({18, 3, 3, 1});
  lite::Tensor dw_tensor(TypeId::kNumberTypeFloat32, dim_dw);
  dw_tensor.set_data(dw_data);
  std::vector<lite::Tensor *> inputs = {&dy_tensor, &x_tensor};
  std::vector<lite::Tensor *> outputs = {&dw_tensor};

  lite::InnerContext context;
  context.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, context.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_Conv2DBackpropFilterFusion};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(conv_param), &context, desc);
  ASSERT_NE(kernel, nullptr);
  mindspore::kernel::LiteKernel::AllocWorkspace(kernel->workspace_size());
  kernel->Run();

  int loop_count = 100;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    kernel->Run();
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  std::string output_path = "./test_data/conv/convfp32_dw_g3_18_3_3_3.bin";
  auto res = CompareRelativeOutput(dw_data, output_path);
  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] dy_data;
  delete[] dw_data;
  dw_tensor.set_data(nullptr);
  x_tensor.set_data(nullptr);
  dy_tensor.set_data(nullptr);
  mindspore::kernel::LiteKernel::FreeWorkspace();
  delete kernel;
  // delete conv_param;
  MS_LOG(INFO) << "TestConvolutionGradFp32 Filter Grad passed";
}

TEST_F(TestConvolutionGradFp32, ConvFp32GroupInputGrad) {
  // prepare stage
  auto conv_param = static_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  ASSERT_NE(conv_param, nullptr);

  InitConvParamGroup3FP32(conv_param);
  size_t dy_size;
  std::string dy_path = "./test_data/conv/convfp32_dy_g3_1_28_28_18.bin";
  auto dy_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(dy_path.c_str(), &dy_size));
  std::vector<int> dim_dy({1, 28, 28, 18});
  lite::Tensor dy_tensor(TypeId::kNumberTypeFloat32, dim_dy);
  dy_tensor.set_data(dy_data);

  size_t w_size;
  std::string w_path = "./test_data/conv/convfp32_w_g3_18_3_3_3.bin";
  auto w_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(w_path.c_str(), &w_size));
  std::vector<int> dim_dw({18, 3, 3, 1});
  lite::Tensor w_tensor(TypeId::kNumberTypeFloat32, dim_dw);
  w_tensor.set_data(w_data);

  size_t output_data_size =
    conv_param->input_batch_ * conv_param->input_h_ * conv_param->input_w_ * conv_param->input_channel_;
  auto dx_data = new float[output_data_size];
  ASSERT_NE(dx_data, nullptr);
  std::vector<int> dim_dx({1, 28, 28, 3});
  lite::Tensor dx_tensor(TypeId::kNumberTypeFloat32, dim_dx);
  dx_tensor.set_data(dx_data);

  std::vector<lite::Tensor *> inputs = {&dy_tensor, &w_tensor};
  std::vector<lite::Tensor *> outputs = {&dx_tensor};
  // runtime part

  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;

  lite::InnerContext context;
  context.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, context.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_Conv2DBackpropInputFusion};
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

  std::string output_path = "./test_data/conv/convfp32_dx_g3_1_28_28_3.bin";
  auto res = CompareRelativeOutput(dx_data, output_path);
  EXPECT_EQ(res, 0);
  delete[] dx_data;
  delete[] w_data;
  delete[] dy_data;
  dx_tensor.set_data(nullptr);
  w_tensor.set_data(nullptr);
  dy_tensor.set_data(nullptr);

  delete kernel;
  mindspore::kernel::LiteKernel::FreeWorkspace();
  // delete conv_param;
  MS_LOG(INFO) << "TestConvolutionGradFp32 Filter Grad passed";
}

TEST_F(TestConvolutionGradFp32, ConvFp32GroupDilationFilterGrad) {
  // prepare stage
  auto conv_param = static_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  ASSERT_NE(conv_param, nullptr);

  InitConvParamGroup3Dilation2FP32(conv_param);
  size_t dy_size;
  std::string dy_path = "./test_data/conv/convfp32_dy_g3_d2_1_26_26_18.bin";
  auto dy_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(dy_path.c_str(), &dy_size));
  std::vector<int> dim_dy({1, 26, 26, 18});
  lite::Tensor dy_tensor(TypeId::kNumberTypeFloat32, dim_dy);
  dy_tensor.set_data(dy_data);

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;
  size_t output_data_size = conv_param->output_channel_ * conv_param->kernel_h_ * conv_param->kernel_w_ *
                            conv_param->input_channel_ / conv_param->group_;

  size_t input_size;
  std::string input_path = "./test_data/conv/convfp32_x_g3_d2_1_28_28_3.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  std::vector<int> dim_x({1, 28, 28, 3});
  lite::Tensor x_tensor(TypeId::kNumberTypeFloat32, dim_x);
  x_tensor.set_data(input_data);

  auto dw_data = new float[output_data_size];
  ASSERT_NE(dw_data, nullptr);
  std::vector<int> dim_dw({18, 3, 3, 1});
  lite::Tensor dw_tensor(TypeId::kNumberTypeFloat32, dim_dw);
  dw_tensor.set_data(dw_data);
  std::vector<lite::Tensor *> inputs = {&dy_tensor, &x_tensor};
  std::vector<lite::Tensor *> outputs = {&dw_tensor};

  lite::InnerContext context;
  context.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, context.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_Conv2DBackpropFilterFusion};
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

  std::string output_path = "./test_data/conv/convfp32_dw_g3_d2_18_3_3_3.bin";
  auto res = CompareRelativeOutput(dw_data, output_path);
  EXPECT_EQ(res, 0);
  delete[] input_data;
  delete[] dy_data;
  delete[] dw_data;
  dw_tensor.set_data(nullptr);
  dy_tensor.set_data(nullptr);
  x_tensor.set_data(nullptr);
  mindspore::kernel::LiteKernel::FreeWorkspace();
  delete kernel;
  // delete conv_param;
  MS_LOG(INFO) << "TestConvolutionGradFp32 Filter Grad passed";
}

TEST_F(TestConvolutionGradFp32, ConvFp32GroupDilationInputGrad) {
  // prepare stage
  auto conv_param = static_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  ASSERT_NE(conv_param, nullptr);

  InitConvParamGroup3Dilation2FP32(conv_param);
  size_t dy_size;
  std::string dy_path = "./test_data/conv/convfp32_dy_g3_d2_1_26_26_18.bin";
  auto dy_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(dy_path.c_str(), &dy_size));
  std::vector<int> dim_dy({1, 26, 26, 18});
  lite::Tensor dy_tensor(TypeId::kNumberTypeFloat32, dim_dy);
  dy_tensor.set_data(dy_data);

  size_t w_size;
  std::string w_path = "./test_data/conv/convfp32_w_g3_d2_18_3_3_3.bin";
  auto w_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(w_path.c_str(), &w_size));
  std::vector<int> dim_w({18, 3, 3, 1});
  lite::Tensor w_tensor(TypeId::kNumberTypeFloat32, dim_w);
  w_tensor.set_data(w_data);

  size_t output_data_size =
    conv_param->input_batch_ * conv_param->input_h_ * conv_param->input_w_ * conv_param->input_channel_;
  auto dx_data = new float[output_data_size];
  ASSERT_NE(dx_data, nullptr);
  std::vector<int> dim_dx({1, 28, 28, 3});
  lite::Tensor dx_tensor(TypeId::kNumberTypeFloat32, dim_dx);
  dx_tensor.set_data(dx_data);

  std::vector<lite::Tensor *> inputs = {&dy_tensor, &w_tensor};
  std::vector<lite::Tensor *> outputs = {&dx_tensor};
  // runtime part

  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;

  lite::InnerContext context;
  context.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, context.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_Conv2DBackpropInputFusion};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(conv_param), &context, desc);
  ASSERT_NE(kernel, nullptr);
  mindspore::kernel::LiteKernel::AllocWorkspace(kernel->workspace_size());

  int loop_count = 100;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    kernel->Run();
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  std::string output_path = "./test_data/conv/convfp32_dx_g3_d2_1_28_28_3.bin";
  auto res = CompareRelativeOutput(dx_data, output_path);
  EXPECT_EQ(res, 0);
  delete[] dx_data;
  delete[] w_data;
  delete[] dy_data;
  dx_tensor.set_data(nullptr);
  dy_tensor.set_data(nullptr);
  w_tensor.set_data(nullptr);
  mindspore::kernel::LiteKernel::FreeWorkspace();
  delete kernel;
  // delete conv_param;
  MS_LOG(INFO) << "TestConvolutionGradFp32 Filter Grad passed";
}

TEST_F(TestConvolutionGradFp32, ConvGroupDilation) {
  // prepare stage
  auto conv_param = static_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  ASSERT_NE(conv_param, nullptr);

  InitConvParamGroup3Dilation2FP32(conv_param);
  size_t x_size;
  std::string x_path = "./test_data/conv/convfp32_x_g3_d2_1_28_28_3.bin";
  auto x_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(x_path.c_str(), &x_size));
  std::vector<int> dim_x({1, 28, 28, 3});
  lite::Tensor x_tensor(TypeId::kNumberTypeFloat32, dim_x);
  x_tensor.set_data(x_data);

  size_t w_size;
  std::string w_path = "./test_data/conv/convfp32_w_g3_d2_18_3_3_3.bin";
  auto w_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(w_path.c_str(), &w_size));
  std::vector<int> dim_w({18, 3, 3, 1});
  lite::Tensor w_tensor(TypeId::kNumberTypeFloat32, dim_w);
  w_tensor.set_data(w_data);

  size_t output_data_size =
    conv_param->output_batch_ * conv_param->output_h_ * conv_param->output_w_ * conv_param->output_channel_;
  auto y_data = new float[output_data_size];
  ASSERT_NE(y_data, nullptr);
  std::vector<int> dim_y({1, 26, 26, 18});
  lite::Tensor y_tensor(TypeId::kNumberTypeFloat32, dim_y);
  y_tensor.set_data(y_data);

  std::vector<lite::Tensor *> inputs = {&x_tensor, &w_tensor};
  std::vector<lite::Tensor *> outputs = {&y_tensor};
  // runtime part

  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;

  lite::InnerContext context;
  context.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, context.Init());

  auto *kernel = new mindspore::kernel::ConvolutionTrainCPUKernel(reinterpret_cast<OpParameter *>(conv_param), inputs,
                                                                  outputs, &context);
  ASSERT_NE(kernel, nullptr);
  kernel->Init();
  mindspore::kernel::LiteKernel::AllocWorkspace(kernel->workspace_size());

  kernel->Train();
  EXPECT_EQ(kernel->IsTrain(), 1);

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

  std::string output_path = "./test_data/conv/convfp32_y_g3_d2_1_26_26_18.bin";
  auto res = CompareRelativeOutput(y_data, output_path);
  EXPECT_EQ(res, 0);

  delete[] y_data;
  delete[] x_data;
  delete[] w_data;
  x_tensor.set_data(nullptr);
  y_tensor.set_data(nullptr);
  w_tensor.set_data(nullptr);
  mindspore::kernel::LiteKernel::FreeWorkspace();
  delete kernel;

  MS_LOG(INFO) << "TestConvolutionFp32 Filter Grad passed";
}

TEST_F(TestConvolutionGradFp32, ConvFp32Dilation2Group2Stride2FilterGrad) {
  // prepare stage
  auto conv_param = static_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  ASSERT_NE(conv_param, nullptr);

  conv_param->input_batch_ = 2;
  conv_param->input_h_ = 32;
  conv_param->input_w_ = 32;
  conv_param->input_channel_ = 4;

  conv_param->output_batch_ = 2;
  conv_param->output_h_ = 15;
  conv_param->output_w_ = 15;
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
  std::string dy_path = "./test_data/conv/convfp32_dy_d2_g2_s2_2_12_15_15.bin";
  auto dy_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(dy_path.c_str(), &dy_size));
  std::vector<int> dim_dy({2, 15, 15, 12});
  lite::Tensor dy_tensor(TypeId::kNumberTypeFloat32, dim_dy);
  dy_tensor.set_data(dy_data);

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;
  size_t output_data_size =
    conv_param->output_channel_ * conv_param->kernel_h_ * conv_param->kernel_w_ * conv_param->input_channel_;

  size_t input_size;
  std::string input_path = "./test_data/conv/convfp32_input0_d2_g2_s2_2_4_32_32.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  ASSERT_NE(input_data, nullptr);
  std::vector<int> dim_x({2, 32, 32, 4});
  lite::Tensor x_tensor(TypeId::kNumberTypeFloat32, dim_x);
  x_tensor.set_data(input_data);

  auto dw_data = new float[output_data_size];
  ASSERT_NE(dw_data, nullptr);
  std::vector<int> dim_dw({12, 3, 3, 2});
  lite::Tensor dw_tensor(TypeId::kNumberTypeFloat32, dim_dw);
  dw_tensor.set_data(dw_data);
  std::vector<lite::Tensor *> inputs = {&dy_tensor, &x_tensor};
  std::vector<lite::Tensor *> outputs = {&dw_tensor};

  lite::InnerContext context;
  context.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, context.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_Conv2DBackpropFilterFusion};
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

  std::string output_path = "./test_data/conv/convfp32_dw_d2_g2_s2_12_2_3_3.bin";
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
  MS_LOG(INFO) << "TestConvolutionGradFp32 Filter Grad passed";
}

TEST_F(TestConvolutionGradFp32, ConvGroup2Dilation2Stride2) {
  // prepare stage
  auto conv_param = static_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  ASSERT_NE(conv_param, nullptr);

  conv_param->input_batch_ = 2;
  conv_param->input_h_ = 32;
  conv_param->input_w_ = 32;
  conv_param->input_channel_ = 4;

  conv_param->output_batch_ = 2;
  conv_param->output_h_ = 15;
  conv_param->output_w_ = 15;
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
  std::string dy_path = "./test_data/conv/convfp32_dy_d2_g2_s2_2_12_15_15.bin";
  auto dy_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(dy_path.c_str(), &dy_size));
  ASSERT_NE(dy_data, nullptr);
  std::vector<int> dim_dy({2, 15, 15, 12});
  lite::Tensor dy_tensor(TypeId::kNumberTypeFloat32, dim_dy);
  dy_tensor.set_data(dy_data);

  size_t w_size;
  std::string w_path = "./test_data/conv/convfp32_w_d2_g2_s2_12_2_3_3.bin";
  auto w_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(w_path.c_str(), &w_size));
  ASSERT_NE(w_data, nullptr);
  std::vector<int> dim_w({12, 3, 3, 2});
  lite::Tensor w_tensor(TypeId::kNumberTypeFloat32, dim_w);
  w_tensor.set_data(w_data);

  size_t output_data_size =
    conv_param->input_batch_ * conv_param->input_h_ * conv_param->input_w_ * conv_param->input_channel_;
  auto dx_data = new float[output_data_size];
  ASSERT_NE(dx_data, nullptr);
  std::vector<int> dim_dx({2, 32, 32, 4});
  lite::Tensor dx_tensor(TypeId::kNumberTypeFloat32, dim_dx);
  dx_tensor.set_data(dx_data);

  std::vector<lite::Tensor *> inputs = {&dy_tensor, &w_tensor};
  std::vector<lite::Tensor *> outputs = {&dx_tensor};
  // runtime part

  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;

  lite::InnerContext context;
  context.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, context.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_Conv2DBackpropInputFusion};
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

  std::string output_path = "./test_data/conv/convfp32_inputdx_d2_g2_s2_2_4_32_32.bin";
  auto res = CompareRelativeOutput(dx_data, output_path);
  EXPECT_EQ(res, 0);
  delete[] dx_data;
  delete[] w_data;
  delete[] dy_data;
  dx_tensor.set_data(nullptr);
  dy_tensor.set_data(nullptr);
  w_tensor.set_data(nullptr);
  delete kernel;
  mindspore::kernel::LiteKernel::FreeWorkspace();
  MS_LOG(INFO) << "TestConvolutionGradFp32 Filter Grad passed";
}

}  // namespace mindspore
