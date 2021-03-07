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
#include <vector>
#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "src/common/file_utils.h"
#include "nnacl/fp16_grad/arithmetic_self_grad.h"

namespace mindspore {
class TestArithmeticSelfGradFp16 : public mindspore::CommonTest {
 public:
  TestArithmeticSelfGradFp16() {}
  float error_bound = 1e-3;
};

TEST_F(TestArithmeticSelfGradFp16, LogGradFp16) {
  size_t output_data_size = 50;
  size_t input_size;
  std::string input_path = "./test_data/activationGrad/log_x_50.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  ASSERT_NE(input_data, nullptr);

  std::string yt_path = "./test_data/activationGrad/log_yt_50.bin";
  auto yt_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(yt_path.c_str(), &input_size));
  ASSERT_NE(yt_data, nullptr);

  std::string output_path = "./test_data/activationGrad/log_out_50.bin";
  auto ref_data = reinterpret_cast<const float *>(mindspore::lite::ReadFile(output_path.c_str(), &input_size));
  ASSERT_NE(ref_data, nullptr);
  EXPECT_EQ(input_size, output_data_size * sizeof(float));

  auto yt_buf = new float16_t[output_data_size];
  auto input_buf = new float16_t[output_data_size];
  auto output_buf = new float16_t[output_data_size];

  for (int i = 0; i < output_data_size; i++) {
    yt_buf[i] = (float16_t)yt_data[i];
    input_buf[i] = (float16_t)input_data[i];
  }

  Fp16LogGrad(yt_buf, input_buf, 50, output_buf);

  int res = 0;
  float error = 0;
  std::cout << "======Compare with reference data======" << std::endl;
  for (int i = 0; i < output_data_size; i++) {
    float diff = std::fabs(static_cast<float>(output_buf[i]) - ref_data[i]);
    if (diff > 0.00001) {
      error += diff;
    }
  }
  error /= static_cast<float>(output_data_size);
  if (error > error_bound) {
    printf("error%f while error_bound=%f\n", error, error_bound);
    res = 1;
  }

  EXPECT_EQ(res, 0);

  delete[] output_buf;
  delete[] yt_buf;
  delete[] input_buf;
  delete[] ref_data;
  delete[] yt_data;
  delete[] input_data;

  MS_LOG(INFO) << "LogGradFp16 passed";
}

}  // namespace mindspore
