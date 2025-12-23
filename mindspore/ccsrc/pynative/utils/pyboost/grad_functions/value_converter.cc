/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "include/pynative/utils/pyboost/grad_functions/value_converter.h"

#include <algorithm>
#include <iterator>
#include <vector>
#include <memory>
#include "include/pynative/utils/pyboost/auto_generate/contiguous.h"
#include "ops_utils/op_utils.h"

namespace mindspore::runtime {
namespace {
tensor::TensorPtr GetContiguousTensor(device::DeviceType device_target, const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  auto device_address = tensor->device_address();
  if (device_address == nullptr || device_address->GetTensorStorageInfo() == nullptr) {
    return tensor;
  }

  auto op = CREATE_PYBOOST_OP(Contiguous, device_target);
  return op->Call(tensor);
}
}  // namespace

Int64ImmPtr ValueConverter::ToInt(const ValuePtr &input) { return Convert<Int64ImmPtr>(input); }

FP32ImmPtr ValueConverter::ToFloat(const ValuePtr &input) { return Convert<FP32ImmPtr>(input); }

BoolImmPtr ValueConverter::ToBool(const ValuePtr &input) { return Convert<BoolImmPtr>(input); }

ScalarPtr ValueConverter::ToScalar(const ValuePtr &input) { return Convert<ScalarPtr>(input); }

tensor::TensorPtr ValueConverter::ToTensor(const ValuePtr &input) { return Convert<tensor::TensorPtr>(input); }

StringImmPtr ValueConverter::ToString(const ValuePtr &input) { return Convert<StringImmPtr>(input); }

TypePtr ValueConverter::ToDtype(const ValuePtr &input) { return Convert<TypePtr>(input); }

ValueTuplePtr ValueConverter::ToValueTuple(const ValuePtr &input) { return Convert<ValueTuplePtr>(input); }

ValuePtr ValueConverter::ToValue(const tensor::TensorPtr &tensor) { return tensor; }

ValuePtr ValueConverter::ToValue(const std::vector<tensor::TensorPtr> &tensors) {
  ValuePtrList values;
  (void)std::transform(tensors.begin(), tensors.end(), std::back_inserter(values),
                       [](const tensor::TensorPtr &tensor) { return tensor; });
  return std::make_shared<ValueTuple>(values);
}

std::vector<int64_t> ValueConverter::ToBasicIntVector(const ValuePtr &input) {
  return ConvertBasic<ValueTuplePtr, std::vector<int64_t>>(input);
}

int64_t ValueConverter::ToBasicInt(const ValuePtr &input) { return ConvertBasic<Int64ImmPtr, int64_t>(input); }

std::optional<Int64ImmPtr> ValueConverter::ToIntOptional(const ValuePtr &input) {
  return ConvertOptional<Int64ImmPtr>(input);
}

std::optional<FP32ImmPtr> ValueConverter::ToFloatOptional(const ValuePtr &input) {
  return ConvertOptional<FP32ImmPtr>(input);
}

std::optional<BoolImmPtr> ValueConverter::ToBoolOptional(const ValuePtr &input) {
  return ConvertOptional<BoolImmPtr>(input);
}

std::optional<ScalarPtr> ValueConverter::ToScalarOptional(const ValuePtr &input) {
  return ConvertOptional<ScalarPtr>(input);
}

std::optional<tensor::TensorPtr> ValueConverter::ToTensorOptional(const ValuePtr &input) {
  return ConvertOptional<tensor::TensorPtr>(input);
}

std::optional<StringImmPtr> ValueConverter::ToStringOptional(const ValuePtr &input) {
  return ConvertOptional<StringImmPtr>(input);
}

std::optional<TypePtr> ValueConverter::ToDtypeOptional(const ValuePtr &input) {
  return ConvertOptional<TypePtr>(input);
}

std::optional<ValueTuplePtr> ValueConverter::ToValueTupleOptional(const ValuePtr &input) {
  return ConvertOptional<ValueTuplePtr>(input);
}

std::optional<std::vector<int64_t>> ValueConverter::ToBasicIntVectorOptional(const ValuePtr &input) {
  return ConvertBasicOptional<ValueTuplePtr, std::vector<int64_t>>(input);
}

std::optional<int64_t> ValueConverter::ToBasicIntOptional(const ValuePtr &input) {
  return ConvertBasicOptional<Int64ImmPtr, int64_t>(input);
}

tensor::TensorPtr ValueConverter::ContiguousTensorValue(device::DeviceType device_target,
                                                        const tensor::TensorPtr &tensor) {
  static bool op_plugin_loaded = ops::GetOpPluginHandle() != nullptr;
  if (device_target == device::DeviceType::kAscend || (op_plugin_loaded && device_target == device::DeviceType::kCPU)) {
    return tensor;
  }
  return GetContiguousTensor(device_target, tensor);
}
ValueTuplePtr ValueConverter::ContiguousTensorValue(device::DeviceType device_target, const ValueTuplePtr &tuple) {
  MS_EXCEPTION_IF_NULL(tuple);
  static bool op_plugin_loaded = ops::GetOpPluginHandle() != nullptr;
  if (device_target == device::DeviceType::kAscend || (op_plugin_loaded && device_target == device::DeviceType::kCPU)) {
    return tuple;
  }

  const auto &value_list = tuple->value();
  if (value_list.empty()) {
    return tuple;
  }

  std::vector<ValuePtr> new_value_list(value_list);
  bool need_rebuild_tuple = false;
  for (size_t i = 0; i < value_list.size(); i++) {
    auto val = value_list[i];
    MS_EXCEPTION_IF_NULL(val);
    if (!val->isa<tensor::Tensor>()) {
      // No need to contiguous, when tuple is not tensor tuple.
      break;
    }

    const auto &tensor = val->cast<tensor::TensorPtr>();
    auto contiguous_tensor = GetContiguousTensor(device_target, tensor);
    if (contiguous_tensor != tensor) {
      need_rebuild_tuple = true;
      new_value_list[i] = contiguous_tensor;
    }
  }

  if (need_rebuild_tuple) {
    return std::make_shared<ValueTuple>(new_value_list);
  }
  return tuple;
}
}  // namespace mindspore::runtime
