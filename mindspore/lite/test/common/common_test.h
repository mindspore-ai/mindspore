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
#include <vector>
#include "gtest/gtest.h"
#include "include/api/format.h"
#include "src/litert/tensor_category.h"

namespace mindspore {
namespace lite {
class Tensor;
}
class CommonTest : public testing::Test {
 public:
  // TestCase only enter once
  static void SetUpTestCase();
  static void TearDownTestCase();

  // every TEST_F macro will enter one
  virtual void SetUp();
  virtual void TearDown();

  void SetShape(int *dst, std::vector<int> src, size_t num) {
    for (size_t i = 0; i < num; i++) {
      dst[i] = src[i];
    }
  }

  template <typename T>
  lite::Tensor *CreateTensor(TypeId dtype, std::vector<int> shape, std::vector<T> data, const Format &format = NHWC,
                             lite::Category category = lite::Category::VAR) {
    auto tensor = CreateTensor(dtype, shape, format, category);
    if (tensor != nullptr) {
      if (!data.empty()) {
        memcpy(TensorMutabData(tensor), data.data(), TensorSize(tensor));
      } else {
        (void)TensorMallocData(tensor);
      }
    }
    return tensor;
  }

  void DestroyTensors(std::vector<lite::Tensor *> tensors);

  template <typename T>
  void PrintData(const std::string &name, T *output_data, int size) {
    std::cout << "The " << name << " is as follows:" << std::endl;
    for (int i = 0; i < std::min(size, 100); i++) {
      std::cout << output_data[i] << " ";
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

  void CompareOutputInt8(int8_t *output_data, int8_t *correct_data, int size, float err_percent);

  int CompareOutput(const float *output_data, size_t output_num, const std::string &file_path);

  float CompareOutputRelativeData(const float *output_data, const float *correct_data, int data_size);

  int CompareRelativeOutput(const float *output_data, const std::string &file_path);

 private:
  lite::Tensor *CreateTensor(TypeId dtype, std::vector<int> shape, const Format &format = NHWC,
                             lite::Category category = lite::Category::VAR);
  void *TensorMutabData(lite::Tensor *tensor);
  size_t TensorSize(lite::Tensor *tensor);
  int TensorMallocData(lite::Tensor *tensor);
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TEST_COMMON_COMMON_TEST_H_
