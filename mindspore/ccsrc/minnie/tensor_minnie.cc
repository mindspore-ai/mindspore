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

#include "minnie/tensor_minnie.h"

namespace mindspore {
namespace tensor {
TensorMinnie &TensorMinnie::operator=(const TensorMinnie &tensor) {
  if (&tensor == this) {
    return *this;
  }
  this->tensor_addr_ = tensor.tensor_addr();
  this->tensor_size_ = tensor.tensor_size();
  return *this;
}

bool TensorMinnie::operator==(const TensorMinnie &tensor) {
  return tensor_addr_ == tensor.tensor_addr() && tensor_size_ == tensor.tensor_size();
}
}  // namespace tensor
}  // namespace mindspore
