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

#include "ir/param_value.h"
#include "ir/dtype/type_id.h"

namespace mindspore {
struct AnfQuantParam {
  double scale;
  int32_t zeroPoint;
  double min;
  double max;
  bool narrowRange;
  bool inited;
  int32_t numBits;
  AnfQuantParam() : scale(1.0), zeroPoint(0), min(0.0), max(0.0), narrowRange(false), numBits(8), inited(false) {}
};
class ParamValueLite : public Value {
 public:
  ParamValueLite() : tensor_addr_(nullptr), tensor_size_(0) {}
  virtual ~ParamValueLite() = default;

  size_t tensor_size() const { return tensor_size_; }
  void set_tensor_size(size_t size) { tensor_size_ = size; }
  // todo
  void *tensor_addr() const { return tensor_addr_; }
  void set_tensor_addr(void *addr) { tensor_addr_ = addr; }

  std::vector<int> tensor_shape() const { return tensor_shape_; }
  void set_tensor_shape(std::vector<int> tensor_shape) { tensor_shape_ = std::move(tensor_shape); }

  TypeId tensor_type() const { return type_id_; }
  void set_tensor_type(TypeId type_id) { type_id_ = type_id; }

  int tensor_shape_size() const {
    int size = 1;
    for (auto val : tensor_shape_) {
      size *= val;
    }
    return size;
  }
  std::vector<std::unique_ptr<AnfQuantParam>> &quant_param() { return quant_params_; }
  void set_quant_param(std::unique_ptr<AnfQuantParam> &quant_param) {
    quant_params_.emplace_back(std::move(quant_param));
  }

  bool operator==(const Value &other) const override {
    this == &other;
  }

 private:
  void *tensor_addr_;
  size_t tensor_size_;
  std::vector<int> tensor_shape_;
  TypeId type_id_;
  std::vector<std::unique_ptr<AnfQuantParam>> quant_params_;
};

using ParamValueLitePtr = std::shared_ptr<ParamValueLite>;
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_PARAM_VALUE_LITE_H_

