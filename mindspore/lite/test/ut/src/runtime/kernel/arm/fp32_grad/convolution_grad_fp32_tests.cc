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
#include "utils/log_adapter.h"
#include "common/common_test.h"
#include "src/common/file_utils.h"
#include "src/common/file_utils_ext.h"
#include "mindspore/lite/src/runtime/kernel/arm/fp32_grad/convolution_grad_filter.h"
#include "mindspore/lite/src/runtime/kernel/arm/fp32_grad/convolution_grad_input.h"
#include "mindspore/lite/src/runtime/kernel/arm/nnacl/conv_parameter.h"
#include "mindspore/lite/src/kernel_registry.h"

namespace mindspore {
class TestConvolutionGradFp32 :  public mindspore::CommonTest {
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

  conv_param->pad_h_ = 1;
  conv_param->pad_w_ = 1;

  conv_param->group_ = 1;
  conv_param->is_relu_ = false;
  conv_param->is_relu6_ = false;
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
  auto conv_param = new ConvParameter();
  InitConvParamGroup1FP32(conv_param);

  size_t dy_size;
  std::string dy_path = "./test_data/conv/convfp32_dy_1_28_28_32.bin";
  auto dy_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(dy_path.c_str(), &dy_size));
  std::vector<int> dim_dy({1, 28, 28, 32});
  lite::tensor::Tensor dy_tensor(TypeId::kNumberTypeFloat32, dim_dy);
  dy_tensor.SetData(dy_data);

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;
  size_t output_data_size =
    conv_param->output_channel_ * conv_param->kernel_h_ * conv_param->kernel_w_ * conv_param->input_channel_;

  size_t input_size;
  std::string input_path = "./test_data/conv/convfp32_x_1_28_28_3.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  std::vector<int> dim_x({1, 28, 28, 3});
  lite::tensor::Tensor x_tensor(TypeId::kNumberTypeFloat32, dim_x);
  x_tensor.SetData(input_data);

  auto dw_data = new float[output_data_size];
  std::vector<int> dim_dw({32, 3, 3, 3});
  lite::tensor::Tensor dw_tensor(TypeId::kNumberTypeFloat32, dim_dw);
  dw_tensor.SetData(dw_data);
  std::vector<lite::tensor::Tensor *> inputs = {&dy_tensor, &x_tensor};
  std::vector<lite::tensor::Tensor *> outputs = {&dw_tensor};

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_Conv2DGradFilter};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(conv_param), NULL, desc, nullptr);

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
  auto res = lite::CompareRelativeOutput(dw_data, output_path);

  EXPECT_EQ(res, 0);

  // delete input_data;
  // delete dy_data;
  // delete [] dw_data;
  delete kernel;
  delete conv_param;
  MS_LOG(INFO) << "TestConvolutionGradFp32 Filter Grad passed";
}

TEST_F(TestConvolutionGradFp32, ConvFp32InputGrad) {
  // prepare stage
  auto conv_param = new ConvParameter();
  InitConvParamGroup1FP32(conv_param);

  size_t dy_size;
  std::string dy_path = "./test_data/conv/convfp32_dy_1_28_28_32.bin";
  auto dy_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(dy_path.c_str(), &dy_size));
  std::vector<int> dim_dy({1, 28, 28, 32});
  lite::tensor::Tensor dy_tensor(TypeId::kNumberTypeFloat32, dim_dy);
  dy_tensor.SetData(dy_data);

  size_t w_size;
  std::string w_path = "./test_data/conv/convfp32_w_32_3_3_3.bin";
  auto w_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(w_path.c_str(), &w_size));
  std::vector<int> dim_dw({32, 3, 3, 3});
  lite::tensor::Tensor w_tensor(TypeId::kNumberTypeFloat32, dim_dw);
  w_tensor.SetData(w_data);

  size_t output_data_size =
    conv_param->input_batch_ * conv_param->input_h_ * conv_param->input_w_ * conv_param->input_channel_;
  auto dx_data = new float[output_data_size];
  std::vector<int> dim_dx({1, 28, 28, 3});
  lite::tensor::Tensor dx_tensor(TypeId::kNumberTypeFloat32, dim_dx);
  dx_tensor.SetData(dx_data);

  std::vector<lite::tensor::Tensor *> inputs = {&dy_tensor, &w_tensor};
  std::vector<lite::tensor::Tensor *> outputs = {&dx_tensor};
  // runtime part

  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_Conv2DGradInput};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(conv_param), NULL, desc, nullptr);

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
  auto res = lite::CompareRelativeOutput(dx_data, output_path);
  EXPECT_EQ(res, 0);

  delete kernel;
  delete conv_param;
  MS_LOG(INFO) << "TestConvolutionGradFp32 Filter Grad passed";
}

TEST_F(TestConvolutionGradFp32, ConvFp32GroupFilterGrad) {
  // prepare stage
  auto conv_param = new ConvParameter();
  InitConvParamGroup3FP32(conv_param);

  size_t dy_size;
  std::string dy_path = "./test_data/conv/convfp32_dy_g3_1_28_28_18.bin";
  auto dy_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(dy_path.c_str(), &dy_size));
  std::vector<int> dim_dy({1, 28, 28, 18});
  lite::tensor::Tensor dy_tensor(TypeId::kNumberTypeFloat32, dim_dy);
  dy_tensor.SetData(dy_data);

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;
  size_t output_data_size = conv_param->output_channel_ * conv_param->kernel_h_ * conv_param->kernel_w_ *
                            conv_param->input_channel_ / conv_param->group_;

  size_t input_size;
  std::string input_path = "./test_data/conv/convfp32_x_g3_1_28_28_3.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  std::vector<int> dim_x({1, 28, 28, 3});
  lite::tensor::Tensor x_tensor(TypeId::kNumberTypeFloat32, dim_x);
  x_tensor.SetData(input_data);

  auto dw_data = new float[output_data_size];
  std::vector<int> dim_dw({18, 3, 3, 1});
  lite::tensor::Tensor dw_tensor(TypeId::kNumberTypeFloat32, dim_dw);
  dw_tensor.SetData(dw_data);
  std::vector<lite::tensor::Tensor *> inputs = {&dy_tensor, &x_tensor};
  std::vector<lite::tensor::Tensor *> outputs = {&dw_tensor};

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_Conv2DGradFilter};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(conv_param), NULL, desc, nullptr);

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

  std::string output_path = "./test_data/conv/convfp32_dw_g3_18_3_3_3.bin";
  auto res = lite::CompareRelativeOutput(dw_data, output_path);
  EXPECT_EQ(res, 0);

  // delete input_data;
  // delete dy_data;
  // delete [] dw_data;
  delete kernel;
  delete conv_param;
  MS_LOG(INFO) << "TestConvolutionGradFp32 Filter Grad passed";
}

TEST_F(TestConvolutionGradFp32, ConvFp32GroupInputGrad) {
  // prepare stage
  auto conv_param = new ConvParameter();
  InitConvParamGroup3FP32(conv_param);

  size_t dy_size;
  std::string dy_path = "./test_data/conv/convfp32_dy_g3_1_28_28_18.bin";
  auto dy_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(dy_path.c_str(), &dy_size));
  std::vector<int> dim_dy({1, 28, 28, 18});
  lite::tensor::Tensor dy_tensor(TypeId::kNumberTypeFloat32, dim_dy);
  dy_tensor.SetData(dy_data);

  size_t w_size;
  std::string w_path = "./test_data/conv/convfp32_w_g3_18_3_3_3.bin";
  auto w_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(w_path.c_str(), &w_size));
  std::vector<int> dim_dw({18, 3, 3, 1});
  lite::tensor::Tensor w_tensor(TypeId::kNumberTypeFloat32, dim_dw);
  w_tensor.SetData(w_data);

  size_t output_data_size =
    conv_param->input_batch_ * conv_param->input_h_ * conv_param->input_w_ * conv_param->input_channel_;
  auto dx_data = new float[output_data_size];
  std::vector<int> dim_dx({1, 28, 28, 3});
  lite::tensor::Tensor dx_tensor(TypeId::kNumberTypeFloat32, dim_dx);
  dx_tensor.SetData(dx_data);

  std::vector<lite::tensor::Tensor *> inputs = {&dy_tensor, &w_tensor};
  std::vector<lite::tensor::Tensor *> outputs = {&dx_tensor};
  // runtime part

  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_Conv2DGradInput};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(conv_param), NULL, desc, nullptr);

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
  auto res = lite::CompareRelativeOutput(dx_data, output_path);
  EXPECT_EQ(res, 0);

  delete kernel;
  delete conv_param;
  MS_LOG(INFO) << "TestConvolutionGradFp32 Filter Grad passed";
}

TEST_F(TestConvolutionGradFp32, ConvFp32GroupDilationFilterGrad) {
  // prepare stage
  auto conv_param = new ConvParameter();

  InitConvParamGroup3Dilation2FP32(conv_param);

  size_t dy_size;
  std::string dy_path = "./test_data/conv/convfp32_dy_g3_d2_1_26_26_18.bin";
  auto dy_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(dy_path.c_str(), &dy_size));
  std::vector<int> dim_dy({1, 26, 26, 18});
  lite::tensor::Tensor dy_tensor(TypeId::kNumberTypeFloat32, dim_dy);
  dy_tensor.SetData(dy_data);

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;
  size_t output_data_size = conv_param->output_channel_ * conv_param->kernel_h_ * conv_param->kernel_w_ *
                            conv_param->input_channel_ / conv_param->group_;

  size_t input_size;
  std::string input_path = "./test_data/conv/convfp32_x_g3_d2_1_28_28_3.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  std::vector<int> dim_x({1, 28, 28, 3});
  lite::tensor::Tensor x_tensor(TypeId::kNumberTypeFloat32, dim_x);
  x_tensor.SetData(input_data);

  auto dw_data = new float[output_data_size];
  std::vector<int> dim_dw({18, 3, 3, 1});
  lite::tensor::Tensor dw_tensor(TypeId::kNumberTypeFloat32, dim_dw);
  dw_tensor.SetData(dw_data);
  std::vector<lite::tensor::Tensor *> inputs = {&dy_tensor, &x_tensor};
  std::vector<lite::tensor::Tensor *> outputs = {&dw_tensor};

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_Conv2DGradFilter};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(conv_param), NULL, desc, nullptr);

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
  auto res = lite::CompareRelativeOutput(dw_data, output_path);
  EXPECT_EQ(res, 0);
  // delete input_data;
  // delete dy_data;
  // delete [] dw_data;
  delete kernel;
  delete conv_param;
  MS_LOG(INFO) << "TestConvolutionGradFp32 Filter Grad passed";
}

TEST_F(TestConvolutionGradFp32, ConvFp32GroupDilationInputGrad) {
  // prepare stage
  auto conv_param = new ConvParameter();
  InitConvParamGroup3Dilation2FP32(conv_param);

  size_t dy_size;
  std::string dy_path = "./test_data/conv/convfp32_dy_g3_d2_1_26_26_18.bin";
  auto dy_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(dy_path.c_str(), &dy_size));
  std::vector<int> dim_dy({1, 26, 26, 18});
  lite::tensor::Tensor dy_tensor(TypeId::kNumberTypeFloat32, dim_dy);
  dy_tensor.SetData(dy_data);

  size_t w_size;
  std::string w_path = "./test_data/conv/convfp32_w_g3_d2_18_3_3_3.bin";
  auto w_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(w_path.c_str(), &w_size));
  std::vector<int> dim_w({18, 3, 3, 1});
  lite::tensor::Tensor w_tensor(TypeId::kNumberTypeFloat32, dim_w);
  w_tensor.SetData(w_data);

  size_t output_data_size =
    conv_param->input_batch_ * conv_param->input_h_ * conv_param->input_w_ * conv_param->input_channel_;
  auto dx_data = new float[output_data_size];
  std::vector<int> dim_dx({1, 28, 28, 3});
  lite::tensor::Tensor dx_tensor(TypeId::kNumberTypeFloat32, dim_dx);
  dx_tensor.SetData(dx_data);

  std::vector<lite::tensor::Tensor *> inputs = {&dy_tensor, &w_tensor};
  std::vector<lite::tensor::Tensor *> outputs = {&dx_tensor};
  // runtime part

  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_Conv2DGradInput};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(conv_param), NULL, desc, nullptr);

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

  std::string output_path = "./test_data/conv/convfp32_dx_g3_d2_1_28_28_3.bin";
  auto res = lite::CompareRelativeOutput(dx_data, output_path);
  EXPECT_EQ(res, 0);

  delete kernel;
  delete conv_param;
  MS_LOG(INFO) << "TestConvolutionGradFp32 Filter Grad passed";
}

// TEST_F(TestConvolutionGradFp32, ConvGroupDilation) {
//   // prepare stage
//   auto conv_param = new ConvParameter();
//   InitConvParamGroup3Dilation2FP32(conv_param);

//   size_t x_size;
//   std::string x_path = "./test_data/conv/convfp32_x_g3_d2_1_28_28_3.bin";
//   auto x_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(x_path.c_str(), &x_size));
//   std::vector<int> dim_x({1, 28, 28, 3});
//   tensor::Tensor x_tensor(TypeId::kNumberTypeFloat32, dim_x);
//   x_tensor.SetData(x_data);

//   size_t w_size;
//   std::string w_path = "./test_data/conv/convfp32_w_g3_d2_18_3_3_3.bin";
//   auto w_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(w_path.c_str(), &w_size));
//   std::vector<int> dim_w({18, 3, 3, 1});
//   tensor::Tensor w_tensor(TypeId::kNumberTypeFloat32, dim_w);
//   w_tensor.SetData(w_data);

//   size_t output_data_size =
//     conv_param->output_batch_ * conv_param->output_h_ * conv_param->output_w_ * conv_param->output_channel_;
//   auto y_data = new float[output_data_size];
//   std::vector<int> dim_y({1, 26, 26, 18});
//   tensor::Tensor y_tensor(TypeId::kNumberTypeFloat32, dim_y);
//   y_tensor.SetData(y_data);

//   std::vector<tensor::Tensor *> inputs = {&x_tensor, &w_tensor};
//   std::vector<tensor::Tensor *> outputs = {&y_tensor};
//   // runtime part

//   printf("Calculating runtime cost...\n");
//   uint64_t time_avg = 0;

//   lite::Context context;
//   ;
//   context.deviceCtx.type = lite::DT_CPU;
//   context.threadNum = 1;

//   kernel::KernelKey desc = {kernel::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Conv2D};
//   auto creator = lite::KernelRegistry::GetInstance()->GetKernelCreator(desc);
//   auto kernel = creator(inputs, outputs, (OpParameter *)conv_param, &context, desc);

//   kernel->train();
//   EXPECT_EQ(kernel->is_train(), 1);

//   // warm up loop
//   for (int i = 0; i < 3; i++) {
//     kernel->Run();
//   }

//   int loop_count = 100;
//   auto time_start = mindspore::lite::GetTimeUs();
//   for (int i = 0; i < loop_count; i++) {
//     kernel->Run();
//   }
//   auto time_end = mindspore::lite::GetTimeUs();
//   auto cost = time_end - time_start;
//   time_avg = cost / loop_count;
//   printf("single thread running time : %f ms\n", time_avg / 1000.0f);

//   std::string output_path = "./test_data/conv/convfp32_y_g3_d2_1_26_26_18.bin";
//   auto res = lite::CompareRelativeOutput(y_data, output_path);
//   EXPECT_EQ(res, 0);

//   delete kernel;
//   delete conv_param;

//   MS_LOG(INFO) << "TestConvolutionFp32 Filter Grad passed";
// }

}  // namespace mindspore
