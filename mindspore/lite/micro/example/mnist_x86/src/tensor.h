
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

#ifndef MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_TENSOR_H_
#define MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_TENSOR_H_

#include "include/ms_tensor.h"

namespace mindspore {
namespace lite {
struct QuantArg {
  double scale;
  int32_t zeroPoint;
  float var_corr{1};
  float mean_corr{0};
  bool inited;
  Vector<float> clusters{};
  int bitNum;
  int roundType;
  int multiplier;
  int dstDtype;
};

class MTensor : public mindspore::tensor::MSTensor {
 public:
  MTensor() = default;
  MTensor(String name, TypeId type, Vector<int> shape) : tensor_name_(name), data_type_(type), shape_(shape) {}
  ~MTensor() override;

  TypeId data_type() const override { return data_type_; }
  Vector<int> shape() const override { return shape_; }
  void set_shape(const Vector<int> &shape) override { shape_ = shape; }
  int ElementsNum() const override;
  size_t Size() const override;
  String tensor_name() const override { return tensor_name_; }
  void set_tensor_name(const String &name) override { tensor_name_ = name; }
  void *MutableData() override;
  void *data() override { return data_; }
  void set_data(void *data) override { data_ = data; }

 private:
  String tensor_name_;
  TypeId data_type_;
  Vector<int> shape_;
  void *data_ = nullptr;
  Vector<QuantArg> quant_params_;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_TENSOR_H_
