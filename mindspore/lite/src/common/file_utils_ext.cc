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
#include <cmath>
#include <cstddef>
#include <iostream>
#include "src/common/file_utils.h"
#include "src/common/file_utils_ext.h"

namespace mindspore {
namespace lite {
static int CompareOutputRelativeData(float *output_data, float *correct_data, int data_size) {
  float error = 0;

  // relative error
  float diffSum = 0.0f;
  float sum = 0.0f;
  for (int i = 0; i < data_size; i++) {
    sum += std::abs(correct_data[i]);
  }
  for (int i = 0; i < data_size; i++) {
    float diff = std::abs(output_data[i] - correct_data[i]);
    diffSum += diff;
  }
  error = diffSum / sum;
  if (error > 1e-4) {
    std::cout << "has accuracy error!\n" << error << "\n";
    return 1;
  }
  return 0;
}

int CompareRelativeOutput(float *output_data, std::string file_path) {
  size_t output_size;
  auto ground_truth = reinterpret_cast<float *>(mindspore::lite::ReadFile(file_path.c_str(), &output_size));
  size_t output_num = output_size / sizeof(float);
  std::cout << "output num : " << output_num << "\n";
  return CompareOutputRelativeData(output_data, ground_truth, output_num);
}
}  // namespace lite
}  // namespace mindspore
