
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

#ifndef MINDSPORE_LITE_MICRO_CALIB_OUTPUT_H_
#define MINDSPORE_LITE_MICRO_CALIB_OUTPUT_H_

#include "include/lite_utils.h"
#include "include/ms_tensor.h"
#include "include/errorcode.h"

namespace mindspore {
namespace lite {

class CalibTensor {
 public:
  CalibTensor(String name, size_t elements_num) : tensor_name_(name), elements_num_(elements_num) {}
  ~CalibTensor() {
    free(data_);
    data_ = nullptr;
  }
  String tensor_name() const { return tensor_name_; }
  int ElementsNum() const { return elements_num_; }
  float *MutableData() {
    if (data_ == nullptr) {
      if (elements_num_ == 0 || elements_num_ > INT16_MAX) {
        return nullptr;
      }
      data_ = static_cast<float *>(malloc(elements_num_ * sizeof(float)));
    }
    return data_;
  }

 private:
  String tensor_name_;
  int elements_num_{0};
  float *data_{nullptr};
};

class Calibrator {
 public:
  Calibrator() = default;
  ~Calibrator() {
    for (auto &calib : calib_outputs_) {
      delete calib;
      calib = nullptr;
    }
    calib_outputs_.clear();
  }
  int ReadCalibData(const char *calib_data_path);
  int CompareOutputs(const Vector<tensor::MSTensor *> &outputs) const;

 private:
  Vector<CalibTensor *> calib_outputs_;
};

}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_MICRO_CALIB_OUTPUT_H_
