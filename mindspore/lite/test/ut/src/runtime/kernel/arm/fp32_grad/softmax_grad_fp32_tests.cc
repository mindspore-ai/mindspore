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

// #include "mindspore/lite/src/ir/tensor.h"
// #include "mindspore/lite/src/lite_kernel.h"

#include "mindspore/lite/include/context.h"
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "src/common/utils.h"
#include "src/common/file_utils.h"
#include "src/common/file_utils_ext.h"

#include "mindspore/lite/src/runtime/kernel/arm/fp32_grad/softmax_grad.h"
#include "mindspore/lite/nnacl/fp32_grad/softmax_grad.h"

namespace mindspore {
class TestSoftmaxGradFp32 : public mindspore::CommonTest {
 public:
  TestSoftmaxGradFp32() {}
};

void InitSoftMaxParam(SoftmaxParameter *softmax_param, int axis) {
  softmax_param->axis_ = axis;
  softmax_param->element_size_ = 1188;
  softmax_param->n_dim_ = 4;
  softmax_param->input_shape_[0] = 1;
  softmax_param->input_shape_[1] = 9;
  softmax_param->input_shape_[2] = 11;
  softmax_param->input_shape_[3] = 12;
}

void InitSoftMaxParam(SoftmaxParameter *softmax_param, int axis, int n, int c, int h, int w) {
  softmax_param->axis_ = axis;
  softmax_param->element_size_ = n * c * h * w;
  softmax_param->n_dim_ = 4;
  softmax_param->input_shape_[0] = n;
  softmax_param->input_shape_[1] = c;
  softmax_param->input_shape_[2] = h;
  softmax_param->input_shape_[3] = w;
}

#if 0   // kernel testing
TEST_F(TestSoftmaxGradFp32, SoftmaxGradKernelAxis0) {
  auto softmax_param = reinterpret_cast<SoftmaxParameter *>(malloc(sizeof(SoftmaxParameter)));
  // set parameters
  InitSoftMaxParam(softmax_param, 0);

  std::vector<int> shape = {1, 9, 11, 12};
  size_t input_size;
  std::string input_path = "./test_data/softmax/softmaxgrad_yinput.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  lite::tensor::Tensor input_tensor(TypeId::kNumberTypeFloat32, shape);
  input_tensor.SetData(input_data);

  std::string yt_path = "./test_data/softmax/softmaxgrad_yt_input.bin";
  auto yt_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(yt_path.c_str(), &input_size));
  lite::tensor::Tensor yt_tensor(TypeId::kNumberTypeFloat32, shape);
  yt_tensor.SetData(yt_data);

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;

  auto out_data = new float[softmax_param->element_size_];
  lite::tensor::Tensor out_tensor(TypeId::kNumberTypeFloat32, shape);
  out_tensor.SetData(out_data);

  std::vector<lite::tensor::Tensor *> inputs = {&input_tensor, &yt_tensor};
  std::vector<lite::tensor::Tensor *> outputs = {&out_tensor};

  // float sum_data[6];
  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_SoftMaxGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(softmax_param), NULL, desc, nullptr);

  kernel->Init();

  // warm up loop
  for (int i = 0; i < 3; i++) {
    kernel->Run();
  }

  int loop_count = 3;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    kernel->Run();
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  std::string output_path = "./test_data/softmax/softmaxgrad_out.bin";
  // auto output_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));

  auto res = lite::CompareRelativeOutput(out_data, output_path);
  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] yt_data;
  delete[] out_data;
  input_tensor.SetData(nullptr);
  yt_tensor.SetData(nullptr);
  out_tensor.SetData(nullptr);
  delete kernel;
  // delete softmax_param;

  MS_LOG(INFO) << "SoftmaxGradKernelAxis0 passed";
}

TEST_F(TestSoftmaxGradFp32, SoftmaxGradKernelAxis1) {
  auto softmax_param = reinterpret_cast<SoftmaxParameter *>(malloc(sizeof(SoftmaxParameter)));
  // set parameters
  InitSoftMaxParam(softmax_param, 1);

  std::vector<int> shape = {1, 9, 11, 12};
  size_t input_size;
  std::string input_path = "./test_data/softmax/softmaxgrad_1_yinput.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  lite::tensor::Tensor input_tensor(TypeId::kNumberTypeFloat32, shape);
  input_tensor.SetData(input_data);

  std::string yt_path = "./test_data/softmax/softmaxgrad_1_yt_input.bin";
  auto yt_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(yt_path.c_str(), &input_size));
  lite::tensor::Tensor yt_tensor(TypeId::kNumberTypeFloat32, shape);
  yt_tensor.SetData(yt_data);

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;

  auto out_data = new float[softmax_param->element_size_];
  lite::tensor::Tensor out_tensor(TypeId::kNumberTypeFloat32, shape);
  out_tensor.SetData(out_data);

  std::vector<lite::tensor::Tensor *> inputs = {&input_tensor, &yt_tensor};
  std::vector<lite::tensor::Tensor *> outputs = {&out_tensor};

  // float sum_data[6];
  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_SoftMaxGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(softmax_param), NULL, desc, nullptr);

  kernel->Init();

  // warm up loop
  for (int i = 0; i < 3; i++) {
    kernel->Run();
  }

  int loop_count = 3;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    kernel->Run();
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  std::string output_path = "./test_data/softmax/softmaxgrad_1_out.bin";
  // auto output_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));

  auto res = lite::CompareRelativeOutput(out_data, output_path);
  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] yt_data;
  delete[] out_data;
  input_tensor.SetData(nullptr);
  yt_tensor.SetData(nullptr);
  out_tensor.SetData(nullptr);
  delete kernel;
  // delete softmax_param;

  MS_LOG(INFO) << "SoftmaxGradKernelAxis1 passed";
}

TEST_F(TestSoftmaxGradFp32, SoftmaxGradKernelAxis2) {
  auto softmax_param = reinterpret_cast<SoftmaxParameter *>(malloc(sizeof(SoftmaxParameter)));
  // set parameters
  InitSoftMaxParam(softmax_param, 2);

  std::vector<int> shape = {1, 9, 11, 12};
  size_t input_size;
  std::string input_path = "./test_data/softmax/softmaxgrad_2_yinput.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  lite::tensor::Tensor input_tensor(TypeId::kNumberTypeFloat32, shape);
  input_tensor.SetData(input_data);

  std::string yt_path = "./test_data/softmax/softmaxgrad_2_yt_input.bin";
  auto yt_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(yt_path.c_str(), &input_size));
  lite::tensor::Tensor yt_tensor(TypeId::kNumberTypeFloat32, shape);
  yt_tensor.SetData(yt_data);

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;

  auto out_data = new float[softmax_param->element_size_];
  lite::tensor::Tensor out_tensor(TypeId::kNumberTypeFloat32, shape);
  out_tensor.SetData(out_data);

  std::vector<lite::tensor::Tensor *> inputs = {&input_tensor, &yt_tensor};
  std::vector<lite::tensor::Tensor *> outputs = {&out_tensor};

  // float sum_data[6];
  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_SoftMaxGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(softmax_param), NULL, desc, nullptr);

  kernel->Init();

  // warm up loop
  for (int i = 0; i < 3; i++) {
    kernel->Run();
  }

  int loop_count = 3;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    kernel->Run();
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  std::string output_path = "./test_data/softmax/softmaxgrad_2_out.bin";
  // auto output_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));

  auto res = lite::CompareRelativeOutput(out_data, output_path);
  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] yt_data;
  delete[] out_data;
  input_tensor.SetData(nullptr);
  yt_tensor.SetData(nullptr);
  out_tensor.SetData(nullptr);
  delete kernel;
  // delete softmax_param;

  MS_LOG(INFO) << "SoftmaxGradKernelAxis2 passed";
}

TEST_F(TestSoftmaxGradFp32, SoftmaxGradKernelAxis3) {
  auto softmax_param = reinterpret_cast<SoftmaxParameter *>(malloc(sizeof(SoftmaxParameter)));
  // set parameters
  InitSoftMaxParam(softmax_param, 3);

  std::vector<int> shape = {1, 9, 11, 12};
  size_t input_size;
  std::string input_path = "./test_data/softmax/softmaxgrad_3_yinput.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  lite::tensor::Tensor input_tensor(TypeId::kNumberTypeFloat32, shape);
  input_tensor.SetData(input_data);

  std::string yt_path = "./test_data/softmax/softmaxgrad_3_yt_input.bin";
  auto yt_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(yt_path.c_str(), &input_size));
  lite::tensor::Tensor yt_tensor(TypeId::kNumberTypeFloat32, shape);
  yt_tensor.SetData(yt_data);

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;

  auto out_data = new float[softmax_param->element_size_];
  lite::tensor::Tensor out_tensor(TypeId::kNumberTypeFloat32, shape);
  out_tensor.SetData(out_data);

  std::vector<lite::tensor::Tensor *> inputs = {&input_tensor, &yt_tensor};
  std::vector<lite::tensor::Tensor *> outputs = {&out_tensor};

  // float sum_data[6];
  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_SoftMaxGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(softmax_param), NULL, desc, nullptr);

  kernel->Init();

  // warm up loop
  for (int i = 0; i < 3; i++) {
    kernel->Run();
  }

  int loop_count = 3;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    kernel->Run();
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  std::string output_path = "./test_data/softmax/softmaxgrad_3_out.bin";
  // auto output_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));

  auto res = lite::CompareRelativeOutput(out_data, output_path);
  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] yt_data;
  delete[] out_data;
  input_tensor.SetData(nullptr);
  yt_tensor.SetData(nullptr);
  out_tensor.SetData(nullptr);
  delete kernel;
  // delete softmax_param;

  MS_LOG(INFO) << "SoftmaxGradKernelAxis3 passed";
}

TEST_F(TestSoftmaxGradFp32, SoftmaxGradKernelAxisMinus1) {
  auto softmax_param = reinterpret_cast<SoftmaxParameter *>(malloc(sizeof(SoftmaxParameter)));
  // set parameters
  InitSoftMaxParam(softmax_param, -1);

  std::vector<int> shape = {1, 9, 11, 12};
  size_t input_size;
  std::string input_path = "./test_data/softmax/softmaxgrad_-1_yinput.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  lite::tensor::Tensor input_tensor(TypeId::kNumberTypeFloat32, shape);
  input_tensor.SetData(input_data);

  std::string yt_path = "./test_data/softmax/softmaxgrad_-1_yt_input.bin";
  auto yt_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(yt_path.c_str(), &input_size));
  lite::tensor::Tensor yt_tensor(TypeId::kNumberTypeFloat32, shape);
  yt_tensor.SetData(yt_data);

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;

  auto out_data = new float[softmax_param->element_size_];
  lite::tensor::Tensor out_tensor(TypeId::kNumberTypeFloat32, shape);
  out_tensor.SetData(out_data);

  std::vector<lite::tensor::Tensor *> inputs = {&input_tensor, &yt_tensor};
  std::vector<lite::tensor::Tensor *> outputs = {&out_tensor};

  // float sum_data[6];
  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_SoftMaxGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(softmax_param), NULL, desc, nullptr);

  kernel->Init();

  // warm up loop
  for (int i = 0; i < 3; i++) {
    kernel->Run();
  }

  int loop_count = 3;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    kernel->Run();
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  std::string output_path = "./test_data/softmax/softmaxgrad_-1_out.bin";
  // auto output_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));

  auto res = lite::CompareRelativeOutput(out_data, output_path);
  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] yt_data;
  delete[] out_data;
  input_tensor.SetData(nullptr);
  yt_tensor.SetData(nullptr);
  out_tensor.SetData(nullptr);
  delete kernel;
  // delete softmax_param;

  MS_LOG(INFO) << "SoftmaxGradKernelAxisMinus1 passed";
}
#endif  // kernel testing

TEST_F(TestSoftmaxGradFp32, SoftmaxGradAxis0) {
  auto softmax_param = new SoftmaxParameter();
  // set parameters
  InitSoftMaxParam(softmax_param, 0);

  int inner_size = 1;
  if (softmax_param->axis_ == -1) softmax_param->axis_ = softmax_param->n_dim_ - 1;
  for (int i = softmax_param->axis_ + 1; i < softmax_param->n_dim_; i++) {
    inner_size *= softmax_param->input_shape_[i];
  }
  float *sum_data = new (std::nothrow) float[inner_size];
  float *sum_mul = new (std::nothrow) float[inner_size * softmax_param->input_shape_[softmax_param->axis_]];
  std::vector<int> shape = {1, 9, 11, 12};
  size_t input_size;
  std::string input_path = "./test_data/softmax/softmaxgrad_yinput.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));

  std::string yt_path = "./test_data/softmax/softmaxgrad_yt_input.bin";
  auto yt_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(yt_path.c_str(), &input_size));

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;

  auto out_data = new float[softmax_param->element_size_];

  // warm up loop
  for (int i = 0; i < 3; i++) {
    SoftmaxGrad(input_data, yt_data, out_data, sum_data, sum_mul, softmax_param);
  }

  int loop_count = 3;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    SoftmaxGrad(input_data, yt_data, out_data, sum_data, sum_mul, softmax_param);
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  std::string output_path = "./test_data/softmax/softmaxgrad_out.bin";

  auto res = lite::CompareRelativeOutput(out_data, output_path);
  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] yt_data;
  delete[] out_data;
  delete[] sum_data;
  delete[] sum_mul;

  delete softmax_param;

  MS_LOG(INFO) << "SoftmaxGradAxis0 passed";
}

TEST_F(TestSoftmaxGradFp32, SoftmaxGradAxis1) {
  auto softmax_param = new SoftmaxParameter();
  // set parameters
  InitSoftMaxParam(softmax_param, 1);

  int inner_size = 1;
  if (softmax_param->axis_ == -1) softmax_param->axis_ = softmax_param->n_dim_ - 1;
  for (int i = softmax_param->axis_ + 1; i < softmax_param->n_dim_; i++) {
    inner_size *= softmax_param->input_shape_[i];
  }
  float *sum_data = new (std::nothrow) float[inner_size];
  float *sum_mul = new (std::nothrow) float[inner_size * softmax_param->input_shape_[softmax_param->axis_]];

  std::vector<int> shape = {1, 9, 11, 12};
  size_t input_size;
  std::string input_path = "./test_data/softmax/softmaxgrad_1_yinput.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));

  std::string yt_path = "./test_data/softmax/softmaxgrad_1_yt_input.bin";
  auto yt_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(yt_path.c_str(), &input_size));

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;

  auto out_data = new float[softmax_param->element_size_];

  // warm up loop
  for (int i = 0; i < 3; i++) {
    SoftmaxGrad(input_data, yt_data, out_data, sum_data, sum_mul, softmax_param);
  }

  int loop_count = 3;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    SoftmaxGrad(input_data, yt_data, out_data, sum_data, sum_mul, softmax_param);
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  std::string output_path = "./test_data/softmax/softmaxgrad_1_out.bin";
  // auto output_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));

  auto res = lite::CompareRelativeOutput(out_data, output_path);
  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] yt_data;
  delete[] out_data;
  delete[] sum_data;
  delete[] sum_mul;

  delete softmax_param;

  MS_LOG(INFO) << "SoftmaxGradAxis1 passed";
}

TEST_F(TestSoftmaxGradFp32, SoftmaxGradAxis2) {
  auto softmax_param = new SoftmaxParameter();
  // set parameters
  InitSoftMaxParam(softmax_param, 2);

  int inner_size = 1;
  if (softmax_param->axis_ == -1) softmax_param->axis_ = softmax_param->n_dim_ - 1;
  for (int i = softmax_param->axis_ + 1; i < softmax_param->n_dim_; i++) {
    inner_size *= softmax_param->input_shape_[i];
  }
  float *sum_data = new (std::nothrow) float[inner_size];
  float *sum_mul = new (std::nothrow) float[inner_size * softmax_param->input_shape_[softmax_param->axis_]];

  std::vector<int> shape = {1, 9, 11, 12};
  size_t input_size;
  std::string input_path = "./test_data/softmax/softmaxgrad_2_yinput.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));

  std::string yt_path = "./test_data/softmax/softmaxgrad_2_yt_input.bin";
  auto yt_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(yt_path.c_str(), &input_size));

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;

  auto out_data = new float[softmax_param->element_size_];

  // warm up loop
  for (int i = 0; i < 3; i++) {
    SoftmaxGrad(input_data, yt_data, out_data, sum_data, sum_mul, softmax_param);
  }

  int loop_count = 3;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    SoftmaxGrad(input_data, yt_data, out_data, sum_data, sum_mul, softmax_param);
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  std::string output_path = "./test_data/softmax/softmaxgrad_2_out.bin";
  // auto output_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));

  auto res = lite::CompareRelativeOutput(out_data, output_path);
  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] yt_data;
  delete[] out_data;
  delete[] sum_data;
  delete[] sum_mul;

  delete softmax_param;

  MS_LOG(INFO) << "SoftmaxGradAxis2 passed";
}

TEST_F(TestSoftmaxGradFp32, SoftmaxGradAxis3) {
  auto softmax_param = new SoftmaxParameter();
  // set parameters
  InitSoftMaxParam(softmax_param, 3);

  int inner_size = 1;
  if (softmax_param->axis_ == -1) softmax_param->axis_ = softmax_param->n_dim_ - 1;
  for (int i = softmax_param->axis_ + 1; i < softmax_param->n_dim_; i++) {
    inner_size *= softmax_param->input_shape_[i];
  }
  float *sum_data = new (std::nothrow) float[inner_size];
  float *sum_mul = new (std::nothrow) float[inner_size * softmax_param->input_shape_[softmax_param->axis_]];

  std::vector<int> shape = {1, 9, 11, 12};
  size_t input_size;
  std::string input_path = "./test_data/softmax/softmaxgrad_3_yinput.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));

  std::string yt_path = "./test_data/softmax/softmaxgrad_3_yt_input.bin";
  auto yt_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(yt_path.c_str(), &input_size));

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;

  auto out_data = new float[softmax_param->element_size_];

  // warm up loop
  for (int i = 0; i < 3; i++) {
    SoftmaxGrad(input_data, yt_data, out_data, sum_data, sum_mul, softmax_param);
  }

  int loop_count = 3;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    SoftmaxGrad(input_data, yt_data, out_data, sum_data, sum_mul, softmax_param);
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  std::string output_path = "./test_data/softmax/softmaxgrad_3_out.bin";
  // auto output_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));

  auto res = lite::CompareRelativeOutput(out_data, output_path);
  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] yt_data;
  delete[] out_data;
  delete[] sum_data;
  delete[] sum_mul;

  delete softmax_param;

  MS_LOG(INFO) << "SoftmaxGradAxis3 passed";
}

TEST_F(TestSoftmaxGradFp32, SoftmaxGradAxisMinus1) {
  auto softmax_param = new SoftmaxParameter();
  // set parameters
  InitSoftMaxParam(softmax_param, -1);

  int inner_size = 1;
  if (softmax_param->axis_ == -1) softmax_param->axis_ = softmax_param->n_dim_ - 1;
  for (int i = softmax_param->axis_ + 1; i < softmax_param->n_dim_; i++) {
    inner_size *= softmax_param->input_shape_[i];
  }
  float *sum_data = new (std::nothrow) float[inner_size];
  float *sum_mul = new (std::nothrow) float[inner_size * softmax_param->input_shape_[softmax_param->axis_]];

  std::vector<int> shape = {1, 9, 11, 12};
  size_t input_size;
  std::string input_path = "./test_data/softmax/softmaxgrad_-1_yinput.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));

  std::string yt_path = "./test_data/softmax/softmaxgrad_-1_yt_input.bin";
  auto yt_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(yt_path.c_str(), &input_size));

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;

  auto out_data = new float[softmax_param->element_size_];

  // warm up loop
  for (int i = 0; i < 3; i++) {
    SoftmaxGrad(input_data, yt_data, out_data, sum_data, sum_mul, softmax_param);
  }

  int loop_count = 3;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    SoftmaxGrad(input_data, yt_data, out_data, sum_data, sum_mul, softmax_param);
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  std::string output_path = "./test_data/softmax/softmaxgrad_-1_out.bin";
  // auto output_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));

  auto res = lite::CompareRelativeOutput(out_data, output_path);
  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] yt_data;
  delete[] out_data;
  delete[] sum_data;
  delete[] sum_mul;

  delete softmax_param;

  MS_LOG(INFO) << "SoftmaxGradAxisMinus1 passed";
}

}  // namespace mindspore
