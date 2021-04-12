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
tensor::TensorPtr TensorConstructUtils::CreateZerosTensor(const TypePtr type_ptr, const std::vector<int64_t> &shape) {
  MS_EXCEPTION_IF_NULL(type_ptr);
  auto type_id = ExtractTypeId(type_ptr);
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(type_id, shape);
  size_t mem_size = IntToSize(tensor->ElementsNum());
  auto tensor_data = tensor->data_c();
  char *data = reinterpret_cast<char *>(tensor_data);
  MS_EXCEPTION_IF_NULL(data);
  (void)memset_s(data, mem_size, 0, mem_size);

  return tensor;
}

tensor::TensorPtr TensorConstructUtils::CreateOnesTensor(const TypePtr type_ptr, const std::vector<int64_t> &shape) {
  MS_EXCEPTION_IF_NULL(type_ptr);
  auto type_id = ExtractTypeId(type_ptr);
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(type_id, shape);
  size_t mem_size = IntToSize(tensor->ElementsNum());
  if (tensor->data_type() == kNumberTypeFloat32) {
    SetTensorData<float>(tensor->data_c(), 1.0, mem_size);
  } else if (tensor->data_type() == kNumberTypeInt) {
    SetTensorData<int>(tensor->data_c(), 1, mem_size);
  }
  return tensor;
}

tensor::TensorPtr TensorConstructUtils::CreateTensor(const TypePtr type_ptr, const std::vector<int64_t> &shape,
                                                     void *data) {
  MS_EXCEPTION_IF_NULL(type_ptr);
  auto type_id = ExtractTypeId(type_ptr);
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(type_id, shape, data, type_id);
  return tensor;
}

TypeId TensorConstructUtils::ExtractTypeId(const TypePtr type_ptr) {
  MS_EXCEPTION_IF_NULL(type_ptr);
  TypeId type_id;
  if (type_ptr->isa<TensorType>()) {
    auto tensor_type = type_ptr->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_type);
    type_id = tensor_type->element()->type_id();
  } else {
    type_id = type_ptr->type_id();
  }
  return type_id;
}
}  // namespace mindspore
