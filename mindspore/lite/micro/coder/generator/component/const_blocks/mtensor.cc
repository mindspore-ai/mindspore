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

#include "coder/generator/component/const_blocks/mtensor.h"

namespace mindspore::lite::micro {

const char *tensor_header = R"RAW(
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
)RAW";

const char *tensor_source = R"RAW(
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

#include "tensor.h"

namespace mindspore {
namespace lite {
size_t DataTypeSize(const TypeId type) {
  switch (type) {
    case kNumberTypeFloat64:
      return sizeof(double);
    case kNumberTypeFloat:
    case kNumberTypeFloat32:
      return sizeof(float);
    case kNumberTypeInt8:
      return sizeof(int8_t);
    case kNumberTypeUInt8:
      return sizeof(uint8_t);
    case kNumberTypeFloat16:
    case kNumberTypeInt16:
      return sizeof(int16_t);
    case kNumberTypeInt32:
      return sizeof(int32_t);
    case kNumberTypeInt64:
      return sizeof(int64_t);
    case kNumberTypeUInt16:
      return sizeof(uint16_t);
    case kNumberTypeUInt32:
      return sizeof(uint32_t);
    case kNumberTypeUInt64:
      return sizeof(uint64_t);
    case kNumberTypeBool:
      return sizeof(bool);
    case kObjectTypeString:
      return sizeof(char);
    case kObjectTypeTensorType:
    default:
      return 0;
  }
}

MTensor::~MTensor() {
  if (data_ != nullptr) {
    free(data_);
    data_ = nullptr;
  }
}

int MTensor::ElementsNum() const {
  int elements = 1;
  for (int i : shape_) {
    elements *= i;
  }
  return elements;
}

size_t MTensor::Size() const {
  size_t element_size = DataTypeSize(data_type_);
  return element_size * ElementsNum();
}

void *MTensor::MutableData() {
  if (data_ == nullptr) {
    data_ = malloc(this->Size());
  }
  return data_;
}
}  // namespace lite
}  // namespace mindspore
)RAW";

}  // namespace mindspore::lite::micro
