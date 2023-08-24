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
#include <memory>
#include <vector>
#include <map>
#include <functional>
namespace mindspore {
tensor::TensorPtr TensorConstructUtils::CreateZerosTensor(const TypePtr &type, const std::vector<int64_t> &shape) {
  MS_EXCEPTION_IF_NULL(type);
  auto type_id = ExtractTypeId(type);
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(type_id, shape);
  size_t mem_size = LongToSize(tensor->ElementsNum());
  auto tensor_data = tensor->data_c();
  char *data = reinterpret_cast<char *>(tensor_data);
  MS_EXCEPTION_IF_NULL(data);
  if (memset_s(data, mem_size, 0, mem_size) != EOK) {
    MS_LOG(ERROR) << "Cannot create zeros tensor.";
    return nullptr;
  }

  return tensor;
}

tensor::TensorPtr TensorConstructUtils::CreateOnesTensor(const TypePtr &type, const std::vector<int64_t> &shape,
                                                         bool skip_exception) {
  MS_EXCEPTION_IF_NULL(type);
  auto type_id = ExtractTypeId(type);
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(type_id, shape);
  const size_t &mem_size = LongToSize(tensor->ElementsNum());
  auto tensor_data = tensor->data_c();
  std::map<TypeId, std::function<void()>> type_dict{
    {kNumberTypeBool, [&tensor_data, mem_size]() { SetTensorData<bool>(tensor_data, true, mem_size); }},
    {kNumberTypeInt8,
     [&tensor_data, mem_size]() { SetTensorData<int8_t>(tensor_data, static_cast<int8_t>(1), mem_size); }},
    {kNumberTypeInt16,
     [&tensor_data, mem_size]() { SetTensorData<int16_t>(tensor_data, static_cast<int16_t>(1), mem_size); }},
    {kNumberTypeInt32,
     [&tensor_data, mem_size]() { SetTensorData<int32_t>(tensor_data, static_cast<int32_t>(1), mem_size); }},
    {kNumberTypeInt64,
     [&tensor_data, mem_size]() { SetTensorData<int64_t>(tensor_data, static_cast<int64_t>(1), mem_size); }},
    {kNumberTypeUInt8,
     [&tensor_data, mem_size]() { SetTensorData<uint8_t>(tensor_data, static_cast<uint8_t>(1), mem_size); }},
    {kNumberTypeUInt16,
     [&tensor_data, mem_size]() { SetTensorData<uint16_t>(tensor_data, static_cast<uint16_t>(1), mem_size); }},
    {kNumberTypeUInt32,
     [&tensor_data, mem_size]() { SetTensorData<uint32_t>(tensor_data, static_cast<uint32_t>(1), mem_size); }},
    {kNumberTypeUInt64,
     [&tensor_data, mem_size]() { SetTensorData<uint64_t>(tensor_data, static_cast<uint64_t>(1), mem_size); }},
    {kNumberTypeFloat16,
     [&tensor_data, mem_size]() { SetTensorData<float16>(tensor_data, static_cast<float16>(1.0), mem_size); }},
    {kNumberTypeFloat32,
     [&tensor_data, mem_size]() { SetTensorData<float>(tensor_data, static_cast<float>(1.0), mem_size); }},
    {kNumberTypeFloat64,
     [&tensor_data, mem_size]() { SetTensorData<double>(tensor_data, static_cast<double>(1.0), mem_size); }},
    {kNumberTypeBFloat16,
     [&tensor_data, mem_size]() { SetTensorData<bfloat16>(tensor_data, static_cast<bfloat16>(1.0), mem_size); }},
  };

  const auto &tensor_type = tensor->data_type();
  auto iter = type_dict.find(tensor_type);
  if (iter == type_dict.end()) {
    if (skip_exception) {
      return nullptr;
    }
    MS_LOG(EXCEPTION) << "unsupported data type: " << tensor_type;
  }
  iter->second();
  return tensor;
}

tensor::TensorPtr TensorConstructUtils::CreateTensor(const TypePtr &type, const std::vector<int64_t> &shape,
                                                     void *data) {
  MS_EXCEPTION_IF_NULL(type);
  auto type_id = ExtractTypeId(type);
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(type_id, shape, data, type_id);
  return tensor;
}

TypeId TensorConstructUtils::ExtractTypeId(const TypePtr &type) {
  MS_EXCEPTION_IF_NULL(type);
  TypeId type_id;
  if (type->isa<TensorType>()) {
    auto tensor_type = type->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_type);
    type_id = tensor_type->element()->type_id();
  } else {
    type_id = type->type_id();
  }
  return type_id;
}
}  // namespace mindspore
