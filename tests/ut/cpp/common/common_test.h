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
#ifndef TESTS_UT_COMMON_UT_COMMON_H_
#define TESTS_UT_COMMON_UT_COMMON_H_

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include "gtest/gtest.h"

namespace UT {
class Common : public testing::Test {
 public:
  // TestCase only enter once
  static void SetUpTestCase();
  static void TearDownTestCase();

  // every TEST_F macro will enter one
  virtual void SetUp();
  virtual void TearDown();

  template <typename T>
  void PrintData(std::string name, T *output_data, int size) {
    std::cout << "The " << name << " is as follows:" << std::endl;
    if (typeid(output_data[0]) == typeid(uint8_t) || typeid(output_data[0]) == typeid(int8_t)) {
      for (size_t i = 0; i < std::min(size, 100); i++) {
        std::cout << (int)output_data[i] << " ";
      }
    } else {
      for (size_t i = 0; i < std::min(size, 100); i++) {
        std::cout << output_data[i] << " ";
      }
    }
    std::cout << std::endl;
  }

  template <typename T>
  static void CompareOutputData(T *output_data, T *correct_data, int size, float err_bound) {
    for (size_t i = 0; i < size; i++) {
      T abs = fabs(output_data[i] - correct_data[i]);
      ASSERT_LE(abs, err_bound);
    }
  }

  void ReadFile(const char *file, size_t *size, char **buf) {
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

#define UT_CHECK_NULL(pointer) ASSERT_NE(pointer, nullptr)
}  // namespace UT
#endif  // TESTS_UT_COMMON_UT_COMMON_H_
