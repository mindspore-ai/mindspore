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

#include "include/ms_tensor.h"
#include "src/tensor.h"

namespace mindspore {
namespace tensor {
tensor::MSTensor *tensor::MSTensor::CreateTensor(const std::string &name, TypeId type, const std::vector<int> &shape,
                                                 const void *data, size_t data_len) {
  auto tensor = new (std::nothrow) lite::Tensor();
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "Failed to allocate tensor.";
    return nullptr;
  }
  tensor->set_data(const_cast<void *>(data));
  tensor->set_shape(shape);
  tensor->set_tensor_name(name);
  tensor->set_data_type(type);
  return tensor;
}
}  // namespace tensor
}  // namespace mindspore
