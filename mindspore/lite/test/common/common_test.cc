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
#include "common/common_test.h"
#include "src/common/file_utils.h"
#include "src/common/log_adapter.h"
#include "src/tensor.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

namespace mindspore {
void CommonTest::SetUpTestCase() {}
void CommonTest::TearDownTestCase() {}
void CommonTest::SetUp() {}
void CommonTest::TearDown() {}

lite::Tensor *CommonTest::CreateTensor(TypeId dtype, std::vector<int> shape, const Format &format,
                                       lite::Category category) {
  return new (std::nothrow) lite::Tensor(dtype, shape, format, category);
}

void CommonTest::DestroyTensors(std::vector<lite::Tensor *> tensors) {
  for (auto &tensor : tensors) {
    delete tensor;
  }
}

void CommonTest::CompareOutputInt8(int8_t *output_data, int8_t *correct_data, int size, float err_percent) {
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

int CommonTest::CompareOutput(const float *output_data, size_t output_num, const std::string &file_path) {
  size_t ground_truth_size = 0;
  auto ground_truth = reinterpret_cast<float *>(lite::ReadFile(file_path.c_str(), &ground_truth_size));
  size_t ground_truth_num = ground_truth_size / sizeof(float);
  printf("ground truth num : %zu\n", ground_truth_num);
  int res = CompareOutputData(output_data, ground_truth, ground_truth_num);
  delete[] ground_truth;
  return res;
}

float CommonTest::CompareOutputRelativeData(const float *output_data, const float *correct_data, int data_size) {
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

int CommonTest::CompareRelativeOutput(const float *output_data, const std::string &file_path) {
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

void *CommonTest::TensorMutabData(lite::Tensor *tensor) { return tensor->MutableData(); }

size_t CommonTest::TensorSize(lite::Tensor *tensor) { return tensor->Size(); }

int CommonTest::TensorMallocData(lite::Tensor *tensor) { return tensor->MallocData(); }
}  // namespace mindspore

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
