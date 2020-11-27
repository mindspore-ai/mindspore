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

#ifndef MINDSPORE_LITE_SRC_PARAM_VALUE_LITE_H_
#define MINDSPORE_LITE_SRC_PARAM_VALUE_LITE_H_

#include <memory>
#include <algorithm>
#include <vector>
#include <utility>
#include "src/tensor.h"

namespace mindspore {
class ParamValueLite : public Value {
 public:
  ParamValueLite() : tensor_addr_(nullptr), tensor_size_(0) {}
  ~ParamValueLite() override {
    if (tensor_addr_ != nullptr) {
      auto tensor_mem = reinterpret_cast<char *>(tensor_addr_);
      delete[](tensor_mem);
      tensor_addr_ = nullptr;
      tensor_size_ = 0;
    }
  }
  MS_DECLARE_PARENT(ParamValueLite, Value)
  size_t tensor_size() const { return tensor_size_; }
  void set_tensor_size(const size_t size) { tensor_size_ = size; }
  void *tensor_addr() const { return tensor_addr_; }
  void set_tensor_addr(void *addr) { tensor_addr_ = addr; }

  std::vector<int> tensor_shape() const { return tensor_shape_; }
  void set_tensor_shape(const std::vector<int> &tensor_shape) { tensor_shape_ = tensor_shape; }

  TypeId tensor_type() const { return type_id_; }
  void set_tensor_type(const TypeId type_id) { type_id_ = type_id; }

  void SetTensorData(void *addr, const size_t size) {
    this->tensor_addr_ = addr;
    this->tensor_size_ = size;
  }

  int tensor_shape_size() const {
    int size = 1;
    for (auto val : tensor_shape_) {
      size *= val;
    }
    return size;
  }

  bool operator==(const Value &other) const override { return this == &other; }

  int format() const { return this->format_; }

  void set_format(int format) { this->format_ = format; }

 private:
  void *tensor_addr_ = nullptr;
  size_t tensor_size_ = 0;
  int format_ = schema::Format::Format_KCHW;
  std::vector<int> tensor_shape_{};
  TypeId type_id_ = TypeId::kNumberTypeFloat32;
};

using ParamValueLitePtr = std::shared_ptr<ParamValueLite>;
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_PARAM_VALUE_LITE_H_
