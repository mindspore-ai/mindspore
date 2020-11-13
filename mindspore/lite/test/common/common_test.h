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
#ifndef MINDSPORE_LITE_TEST_COMMON_COMMON_TEST_H_
#define MINDSPORE_LITE_TEST_COMMON_COMMON_TEST_H_

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <algorithm>
#include "gtest/gtest.h"
#include "src/common/file_utils.h"

namespace mindspore {
class CommonTest : public testing::Test {
 public:
  // TestCase only enter once
  static void SetUpTestCase();
  static void TearDownTestCase();

  // every TEST_F macro will enter one
  virtual void SetUp();
  virtual void TearDown();

  template <typename T>
  void PrintData(const std::string &name, T *output_data, int size) {
    std::cout << "The " << name << " is as follows:" << std::endl;
    if (typeid(output_data[0]) == typeid(uint8_t) || typeid(output_data[0]) == typeid(int8_t)) {
      for (int i = 0; i < std::min(size, 100); i++) {
        std::cout << static_cast<int>(output_data[i]) << " ";
      }
    } else {
      for (int i = 0; i < std::min(size, 100); i++) {
        std::cout << output_data[i] << " ";
      }
    }
    std::cout << std::endl;
  }

  template <typename T>
  static int CompareOutputData(const T *output_data, const T *correct_data, int size, float err_bound = 1e-4) {
    float error = 0;
    for (int i = 0; i < size; i++) {
      T diff = std::fabs(output_data[i] - correct_data[i]);
      if (diff > 0.00001) {
        error += diff;
      }
    }
    error /= static_cast<float>(size);
    if (error > err_bound) {
      return 1;
    }
    return 0;
  }

  static void CompareOutputInt8(int8_t *output_data, int8_t *correct_data, int size, float err_percent) {
    int bias_count = 0;
    for (int i = 0; i < size; i++) {
      int8_t diff = abs(output_data[i] - correct_data[i]);
      ASSERT_LE(diff, 1);
      if (diff == 1) {
        bias_count++;
      }
    }
    float bias_percent = static_cast<float>(bias_count) / static_cast<float>(size);
    ASSERT_LE(bias_percent, err_percent);
  }

  static int CompareOutput(const float *output_data, size_t output_num, const std::string &file_path) {
    size_t ground_truth_size = 0;
    auto ground_truth = reinterpret_cast<float *>(lite::ReadFile(file_path.c_str(), &ground_truth_size));
    size_t ground_truth_num = ground_truth_size / sizeof(float);
    printf("ground truth num : %zu\n", ground_truth_num);
    int res = CompareOutputData(output_data, ground_truth, ground_truth_num);
    delete[] ground_truth;
    return res;
  }

  static float CompareOutputRelativeData(const float *output_data, const float *correct_data, int data_size) {
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
    return error;
  }

  static int CompareRelativeOutput(const float *output_data, const std::string &file_path) {
    size_t output_size;
    auto ground_truth = reinterpret_cast<float *>(mindspore::lite::ReadFile(file_path.c_str(), &output_size));
    if (ground_truth == nullptr) {
      return 1;
    }
    size_t output_num = output_size / sizeof(float);
    float error = CompareOutputRelativeData(output_data, ground_truth, output_num);
    delete[] ground_truth;
    if (error > 1e-4) {
      return 1;
    }
    return 0;
  }

  static float RelativeOutputError(const float *output_data, const std::string &file_path) {
    size_t output_size = 0;
    auto ground_truth = reinterpret_cast<float *>(mindspore::lite::ReadFile(file_path.c_str(), &output_size));
    size_t output_num = output_size / sizeof(float);
    float error = CompareOutputRelativeData(output_data, ground_truth, output_num);
    delete[] ground_truth;
    return error;
  }

  static void ReadFile(const char *file, size_t *size, char **buf) {
    ASSERT_NE(nullptr, file);
    ASSERT_NE(nullptr, size);
    ASSERT_NE(nullptr, buf);
    std::string path = std::string(file);
    std::ifstream ifs(path);
    ASSERT_EQ(true, ifs.good());
    ASSERT_EQ(true, ifs.is_open());

    ifs.seekg(0, std::ios::end);
    *size = ifs.tellg();
    *buf = new char[*size];

    ifs.seekg(0, std::ios::beg);
    ifs.read(*buf, *size);
    ifs.close();
  }
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TEST_COMMON_COMMON_TEST_H_
