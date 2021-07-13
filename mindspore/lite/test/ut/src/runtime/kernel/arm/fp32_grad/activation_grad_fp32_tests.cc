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
#include <algorithm>

#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "src/common/file_utils.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/src/tensor.h"
#include "mindspore/lite/src/lite_kernel.h"
#include "mindspore/lite/src/runtime/kernel/arm/fp32_grad/activation_grad.h"
#include "nnacl/fp32_grad/activation_grad.h"

namespace mindspore {
class TestActGradFp32 : public mindspore::CommonTest {
 public:
  TestActGradFp32() {}
};

TEST_F(TestActGradFp32, ReluGradFp32) {
  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;
  size_t output_data_size = 50;

  size_t input_size;
  std::string input_path = "./activationGrad/relu_y_50.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  ASSERT_NE(input_data, nullptr);
  EXPECT_EQ(input_size, output_data_size * sizeof(float));

  std::string yt_path = "./activationGrad/relu_yt_50.bin";
  auto yt_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(yt_path.c_str(), &input_size));
  ASSERT_NE(yt_data, nullptr);
  EXPECT_EQ(input_size, output_data_size * sizeof(float));

  auto output_data = new float[output_data_size];
  ASSERT_NE(output_data, nullptr);

  // warm up loop
  for (int i = 0; i < 3; i++) {
    ReluGrad(yt_data, input_data, output_data_size, output_data);
  }

  int loop_count = 100;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    ReluGrad(yt_data, input_data, 50, output_data);
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  printf("==================output data=================\n");
  for (int i = 0; i < 20; i++) {
    std::cout << output_data[i] << " ,";
  }
  std::cout << std::endl;

  std::string output_path = "./activationGrad/relu_out_50.bin";

  int res = CompareRelativeOutput(output_data, output_path);

  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] output_data;
  delete[] yt_data;

  MS_LOG(INFO) << "ReluGradFp32 passed";
}

TEST_F(TestActGradFp32, Relu6GradFp32) {
  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;
  size_t output_data_size = 50;

  size_t input_size;
  std::string input_path = "./activationGrad/relu6_y_50.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  ASSERT_NE(input_data, nullptr);

  std::string yt_path = "./activationGrad/relu6_yt_50.bin";
  auto yt_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(yt_path.c_str(), &input_size));
  ASSERT_NE(yt_data, nullptr);

  auto output_data = new float[output_data_size];
  ASSERT_NE(output_data, nullptr);

  // warm up loop
  for (int i = 0; i < 3; i++) {
    Relu6Grad(yt_data, input_data, 50, output_data);
  }

  int loop_count = 100;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    Relu6Grad(yt_data, input_data, 50, output_data);
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  printf("==================output data=================\n");
  for (int i = 0; i < 20; i++) {
    std::cout << output_data[i] << " ,";
  }
  std::cout << std::endl;

  std::string output_path = "./activationGrad/relu6_out_50.bin";
  int res = CompareRelativeOutput(output_data, output_path);

  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] output_data;
  delete[] yt_data;

  MS_LOG(INFO) << "Relu6GradFp32 passed";
}

TEST_F(TestActGradFp32, LReluGradFp32) {
  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;
  size_t output_data_size = 50;

  size_t input_size;
  std::string input_path = "./activationGrad/lrelu_y_50.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  ASSERT_NE(input_data, nullptr);

  std::string yt_path = "./activationGrad/lrelu_yt_50.bin";
  auto yt_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(yt_path.c_str(), &input_size));
  ASSERT_NE(yt_data, nullptr);

  auto output_data = new float[output_data_size];
  ASSERT_NE(output_data, nullptr);

  // warm up loop
  for (int i = 0; i < 3; i++) {
    LReluGrad(yt_data, input_data, 50, output_data, 0.1);
  }

  int loop_count = 100;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    LReluGrad(yt_data, input_data, 50, output_data, 0.1);
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  printf("==================output data=================\n");
  for (int i = 0; i < 20; i++) {
    std::cout << output_data[i] << " ,";
  }
  std::cout << std::endl;

  std::string output_path = "./activationGrad/lrelu_out_50.bin";
  int res = CompareRelativeOutput(output_data, output_path);

  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] output_data;
  delete[] yt_data;

  MS_LOG(INFO) << "LReluGradFp32 passed";
}

TEST_F(TestActGradFp32, SigmoidGradFp32) {
  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;
  size_t output_data_size = 50;

  size_t input_size;
  std::string input_path = "./activationGrad/sigmoid_y_50.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  ASSERT_NE(input_data, nullptr);

  std::string yt_path = "./activationGrad/sigmoid_yt_50.bin";
  auto yt_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(yt_path.c_str(), &input_size));
  ASSERT_NE(yt_data, nullptr);

  auto output_data = new float[output_data_size];
  ASSERT_NE(output_data, nullptr);

  // warm up loop
  for (int i = 0; i < 3; i++) {
    SigmoidGrad(yt_data, input_data, 50, output_data);
  }

  int loop_count = 100;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    SigmoidGrad(yt_data, input_data, 50, output_data);
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  printf("==================output data=================\n");
  for (int i = 0; i < 20; i++) {
    std::cout << output_data[i] << " ,";
  }
  std::cout << std::endl;

  std::string output_path = "./activationGrad/sigmoid_out_50.bin";
  int res = CompareRelativeOutput(output_data, output_path);

  EXPECT_EQ(res, 0);
  // CompareOutput(output_data, output_data_size, output_path);

  delete[] input_data;
  delete[] output_data;
  delete[] yt_data;

  MS_LOG(INFO) << "SigmoidGradFp32 passed";
}

TEST_F(TestActGradFp32, tanhGradFp32) {
  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;
  size_t output_data_size = 50;

  size_t input_size;
  std::string input_path = "./activationGrad/tanh_y_50.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  ASSERT_NE(input_data, nullptr);

  std::string yt_path = "./activationGrad/tanh_yt_50.bin";
  auto yt_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(yt_path.c_str(), &input_size));
  ASSERT_NE(yt_data, nullptr);

  auto output_data = new float[output_data_size];
  ASSERT_NE(output_data, nullptr);

  // warm up loop
  for (int i = 0; i < 3; i++) {
    TanhGrad(yt_data, input_data, 50, output_data);
  }

  int loop_count = 100;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    TanhGrad(yt_data, input_data, 50, output_data);
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  printf("==================output data=================\n");
  for (int i = 0; i < 20; i++) {
    std::cout << output_data[i] << " ,";
  }
  std::cout << std::endl;

  std::string output_path = "./activationGrad/tanh_out_50.bin";
  int res = CompareRelativeOutput(output_data, output_path);

  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] output_data;
  delete[] yt_data;
  MS_LOG(INFO) << "TanhGradFp32 passed";
}

TEST_F(TestActGradFp32, hswishGradFp32) {
  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;
  const size_t output_data_size = 10;

  size_t input_size;
  std::string input_path = "./activationGrad/hswish_x_50.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  ASSERT_NE(input_data, nullptr);
  EXPECT_EQ(input_size, output_data_size * sizeof(float));

  std::string yt_path = "./activationGrad/hswish_yt_50.bin";
  auto yt_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(yt_path.c_str(), &input_size));
  ASSERT_NE(yt_data, nullptr);
  EXPECT_EQ(input_size, output_data_size * sizeof(float));

  auto output_data = new float[output_data_size];
  ASSERT_NE(output_data, nullptr);

  // warm up loop
  for (int i = 0; i < 3; i++) {
    HSwishGrad(yt_data, input_data, static_cast<int>(output_data_size), output_data);
  }

  int loop_count = 100;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    HSwishGrad(yt_data, input_data, output_data_size, output_data);
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  printf("==================output data=================\n");
  size_t min = (output_data_size < 20UL) ? output_data_size : 20UL;
  for (size_t i = 0; i < min; i++) {
    std::cout << output_data[i] << " ,";
  }
  std::cout << std::endl;

  std::string output_path = "./activationGrad/hswish_out_50.bin";
  int res = CompareRelativeOutput(output_data, output_path);

  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] output_data;
  delete[] yt_data;
  MS_LOG(INFO) << "hswishGradFp32 passed";
}

TEST_F(TestActGradFp32, hsigmoidGradFp32) {
  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;
  const size_t output_data_size = 10;

  size_t input_size;
  std::string input_path = "./activationGrad/hsig_x_50.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  ASSERT_NE(input_data, nullptr);
  EXPECT_EQ(input_size, output_data_size * sizeof(float));

  std::string yt_path = "./activationGrad/hsig_yt_50.bin";
  auto yt_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(yt_path.c_str(), &input_size));
  ASSERT_NE(yt_data, nullptr);
  EXPECT_EQ(input_size, output_data_size * sizeof(float));

  auto output_data = new float[output_data_size];
  ASSERT_NE(output_data, nullptr);

  // warm up loop
  for (int i = 0; i < 3; i++) {
    HSigmoidGrad(yt_data, input_data, static_cast<int>(output_data_size), output_data);
  }

  int loop_count = 100;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    HSigmoidGrad(yt_data, input_data, output_data_size, output_data);
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  printf("==================output data=================\n");
  size_t min = (output_data_size < 20UL) ? output_data_size : 20UL;
  for (size_t i = 0; i < min; i++) {
    std::cout << output_data[i] << " ,";
  }
  std::cout << std::endl;

  std::string output_path = "./activationGrad/hsig_out_50.bin";
  int res = CompareRelativeOutput(output_data, output_path);

  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] output_data;
  delete[] yt_data;
  MS_LOG(INFO) << "hsigmoidGradFp32 passed";
}

}  // namespace mindspore
