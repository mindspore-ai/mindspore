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

#ifndef MINDSPORE_CCSRC_MINNIE_TENSOR_MINNIE_H_
#define MINDSPORE_CCSRC_MINNIE_TENSOR_MINNIE_H_

#include <memory>

#include "ir/meta_tensor.h"

namespace mindspore {
namespace tensor {
// definition of Tensor Minnie
class TensorMinnie : public MetaTensor {
 public:
  TensorMinnie() : MetaTensor() {}
  ~TensorMinnie() override = default;
  MS_DECLARE_PARENT(TensorMinnie, MetaTensor)

  // brief Overloads operator = for TensorMinnie.
  //
  // The constructed TensorMinnie object has the same type and shape with tensor_base.
  //
  // param meta_tensor An existing TensorMinnie object.
  virtual TensorMinnie &operator=(const TensorMinnie &tensor);

  // brief Compares two TensorMinnie objects.
  //
  // The constructed TensorMinnie object has the same type and shape with tensor_base.
  //
  // param meta_tensor The TensorMinnie object to be compared.
  // return true: If having same type and shape, return true, or return false.
  virtual bool operator==(const TensorMinnie &tensor);

  // brief Get the tensor's size for C++
  //
  // return size_t
  size_t tensor_size() const { return tensor_size_; }

  // brief Set Tensor data size for c++ type
  void set_tensor_size(size_t size) { tensor_size_ = size; }

  // brief Get Tensor data pointer for c++ type
  //
  // return The pointer to the object
  void *tensor_addr() const { return tensor_addr_; }

  // brief Set Tensor data pointer for c++ type
  void set_tensor_addr(void *addr) { tensor_addr_ = addr; }

 protected:
  // brief Data addr of the tensor.
  void *tensor_addr_;

  // brief Data size of the tensor.
  size_t tensor_size_;
};

using TensorMinniePtr = std::shared_ptr<TensorMinnie>;
}  // namespace tensor
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINNIE_TENSOR_MINNIE_H_
