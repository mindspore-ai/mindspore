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

#ifndef MINDSPORE_CCSRC_MINNIE_PARAM_VALUE_LITE_H_
#define MINDSPORE_CCSRC_MINNIE_PARAM_VALUE_LITE_H_

#include <memory>

#include "ir/anf.h"

namespace mindspore {
class ParamValueLite : public ParamValue {
 public:
  ParamValueLite() : tensor_addr_(nullptr), tensor_size_(0) {}
  virtual ~ParamValueLite() = default;

  size_t tensor_size() const { return tensor_size_; }
  void set_tensor_size(size_t size) { tensor_size_ = size; }

  void *tensor_addr() const { return tensor_addr_; }
  void set_tensor_addr(void *addr) { tensor_addr_ = addr; }

 private:
  void *tensor_addr_;
  size_t tensor_size_;
};

using ParamValueLitePtr = std::shared_ptr<ParamValueLite>;
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINNIE_PARAM_VALUE_LITE_H_
