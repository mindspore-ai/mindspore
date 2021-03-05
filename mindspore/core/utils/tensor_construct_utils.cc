/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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
#include "utils/tensor_construct_utils.h"
#include <vector>
#include <memory>
namespace mindspore {
tensor::TensorPtr TensorConstructUtils::CreateZerosTensor(TypeId type, const std::vector<int64_t> &shape) {
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(type, shape);
  size_t mem_size = IntToSize(tensor->ElementsNum());
  auto tensor_data = tensor->data_c();
  char *data = reinterpret_cast<char *>(tensor_data);
  MS_EXCEPTION_IF_NULL(data);
  (void)memset_s(data, mem_size, 0, mem_size);

  return tensor;
}

tensor::TensorPtr TensorConstructUtils::CreateOnesTensor(TypeId type, const std::vector<int64_t> &shape) {
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(type, shape);
  size_t mem_size = IntToSize(tensor->ElementsNum());
  if (tensor->data_type() == kNumberTypeFloat32) {
    SetTensorData<float>(tensor->data_c(), 1.0, mem_size);
  } else if (tensor->data_type() == kNumberTypeInt) {
    SetTensorData<int>(tensor->data_c(), 1, mem_size);
  }
  return tensor;
}

tensor::TensorPtr TensorConstructUtils::CreateTensor(TypeId type, const std::vector<int64_t> &shape, void *data) {
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(type, shape, data, type);
  return tensor;
}
}  // namespace mindspore
