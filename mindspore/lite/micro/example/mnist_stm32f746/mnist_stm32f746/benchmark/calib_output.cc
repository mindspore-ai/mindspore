
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

#include "calib_output.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdio.h>
#include <cmath>

namespace mindspore {
namespace lite {
constexpr float kToleranceVal = 0.0001;

#define MS_ERROR_IF_NULL(ptr)            \
  do {                                   \
    if ((ptr) == nullptr) {              \
      return mindspore::lite::RET_ERROR; \
    }                                    \
  } while (0)

int Calibrator::ReadCalibData(const char *calib_data_path) {
  std::ifstream in_file(calib_data_path);
  if (!in_file.good()) {
    printf("file is not exist, %s\n", calib_data_path);
    return RET_ERROR;
  }
  if (!in_file.is_open()) {
    printf("open file failed, %s\n", calib_data_path);
    in_file.close();
    return RET_ERROR;
  }
  while (!in_file.eof()) {
    std::string line;
    getline(in_file, line);
    if (line.empty()) {
      continue;
    }
    std::stringstream name_line(line);
    std::string tensor_name;
    size_t dim = 0;
    name_line >> tensor_name >> dim;
    size_t elements = 1;
    for (size_t i = 0; i < dim; i++) {
      size_t tmp_dim;
      name_line >> tmp_dim;
      elements *= tmp_dim;
    }
    getline(in_file, line);
    std::stringstream data_line(line);
    String name(tensor_name.c_str());
    CalibTensor *output = new (std::nothrow) CalibTensor(name, elements);
    MS_ERROR_IF_NULL(output);
    float *data = output->MutableData();
    MS_ERROR_IF_NULL(data);
    for (size_t i = 0; i < elements; i++) {
      data_line >> data[i];
    }
    calib_outputs_.push_back(output);
  }
  in_file.close();
  return RET_OK;
}

template <typename T>
float CompareData(const T *output, const float *calib, size_t elements_num) {
  float error = 0.;
  if (output == nullptr || calib == nullptr) {
    printf("output or calib is nullptr\n");
    return error;
  }
  for (size_t i = 0; i < elements_num; ++i) {
    if (std::isnan(output[i]) || std::isinf(output[i]) || std::isnan(calib[i]) || std::isinf(calib[i])) {
      printf("error, output data is nan or inf\n");
      return error;
    }
    error += std::abs(output[i] - calib[i]);
  }
  return error;
}

int Calibrator::CompareOutputs(const Vector<tensor::MSTensor *> &outputs) const {
  if (outputs.size() != calib_outputs_.size()) {
    printf("error, outputs and calibs size is mismatch\n");
    return RET_ERROR;
  }
  float total_error = 0;
  size_t outputs_num = outputs.size();
  for (size_t i = 0; i < outputs_num; ++i) {
    tensor::MSTensor *output = outputs[i];
    MS_ERROR_IF_NULL(output);
    CalibTensor *calib = calib_outputs_[i];
    MS_ERROR_IF_NULL(calib);
    if (output->tensor_name() != calib->tensor_name()) {
      printf("error, output tensor name is not equal to calib\n");
      return RET_ERROR;
    }
    if (output->ElementsNum() != calib->ElementsNum()) {
      printf("error, output elements num is not equal to calib\n");
      return RET_ERROR;
    }
    switch (output->data_type()) {
      case TypeId::kNumberTypeFloat:
      case TypeId::kNumberTypeFloat32: {
        total_error += CompareData(static_cast<float *>(output->data()), calib->MutableData(), output->ElementsNum());
        break;
      }
      case TypeId::kNumberTypeInt8: {
        total_error += CompareData(static_cast<int8_t *>(output->data()), calib->MutableData(), output->ElementsNum());
        break;
      }
      case TypeId::kNumberTypeUInt8: {
        total_error += CompareData(static_cast<uint8_t *>(output->data()), calib->MutableData(), output->ElementsNum());
        break;
      }
      case TypeId::kNumberTypeUInt:
      case TypeId::kNumberTypeUInt32: {
        total_error += CompareData(static_cast<int32_t *>(output->data()), calib->MutableData(), output->ElementsNum());
        break;
      }
      default: {
        printf("unsupported tensor data type\n");
      }
    }
  }
  if (total_error > kToleranceVal) {
    printf("compare outputs failed, total error: %f\n", total_error);
    return RET_ERROR;
  }
  printf("compare outputs success, total error: %f\n", total_error);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
