

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
#include <utility>
#include <vector>

namespace mindspore {
namespace lite {
struct QuantArg {
  double scale;
  int32_t zeroPoint;
  float var_corr{1};
  float mean_corr{0};
  bool inited;
  std::vector<float> clusters{};
  int bitNum;
  int roundType;
  int multiplier;
  int dstDtype;
};

class MTensor : public mindspore::tensor::MSTensor {
 public:
  MTensor() = default;
  MTensor(std::string name, enum TypeId type, std::vector<int32_t> shape)
      : tensor_name_(std::move(name)), data_type_(type), shape_(std::move(shape)) {}
  ~MTensor() override;

  TypeId data_type() const override { return data_type_; }
  std::vector<int> shape() const override { return shape_; }
  int DimensionSize(size_t index) const override;
  int ElementsNum() const override;
  size_t Size() const override;
  void *MutableData() override;
  std::string tensor_name() const override { return tensor_name_; }
  void set_tensor_name(const std::string name) override { tensor_name_ = name; }
  void set_data(void *data) override { data_ = data; }

 private:
  std::string tensor_name_;
  TypeId data_type_;
  std::vector<int> shape_;
  void *data_ = nullptr;
  std::vector<QuantArg> quant_params_;
};

}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_TENSOR_H_


