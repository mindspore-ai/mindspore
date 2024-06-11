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

#include "runtime/pynative/op_function/value_converter.h"

#include <vector>
#include <memory>
#include "kernel/pyboost/auto_generate/contiguous.h"
#include "runtime/device/ms_device_shape_transfer.h"

namespace mindspore::runtime {
namespace {
tensor::BaseTensorPtr GetContiguousTensor(OpRunnerInfo *op_runner_info, const tensor::BaseTensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  const auto &storage_info = tensor->storage_info();
  if (storage_info == nullptr) {
    return tensor;
  }

  if (storage_info->is_contiguous && storage_info->storage_offset == 0) {
    // Tensor is not contiguous, or offset is not zero. Need to contiguous or copy.
    auto new_device_address = std::static_pointer_cast<device::DeviceAddress>(tensor->device_address());
    if (trans::FormatHelper::GetInstance().IsBaseFormatType(new_device_address->GetFormatEnum())) {
      // Special format need to contiguous
      return tensor;
    }
  }

  auto op = CREATE_PYBOOST_OP(Contiguous, op_runner_info->device_target);
  return op->Call(tensor);
}
}  // namespace

Int64ImmPtr ValueConverter::ToInt(const ValuePtrList &inputs, size_t i) { return Convert<Int64ImmPtr>(inputs, i); }

FP32ImmPtr ValueConverter::ToFloat(const ValuePtrList &inputs, size_t i) { return Convert<FP32ImmPtr>(inputs, i); }

BoolImmPtr ValueConverter::ToBool(const ValuePtrList &inputs, size_t i) { return Convert<BoolImmPtr>(inputs, i); }

ScalarPtr ValueConverter::ToScalar(const ValuePtrList &inputs, size_t i) { return Convert<ScalarPtr>(inputs, i); }

tensor::BaseTensorPtr ValueConverter::ToTensor(const ValuePtrList &inputs, size_t i) {
  return Convert<tensor::BaseTensorPtr>(inputs, i);
}

StringImmPtr ValueConverter::ToString(const ValuePtrList &inputs, size_t i) { return Convert<StringImmPtr>(inputs, i); }

TypePtr ValueConverter::ToDtype(const ValuePtrList &inputs, size_t i) { return Convert<TypePtr>(inputs, i); }

ValueTuplePtr ValueConverter::ToValueTuple(const ValuePtrList &inputs, size_t i) {
  return Convert<ValueTuplePtr>(inputs, i);
}

std::optional<Int64ImmPtr> ValueConverter::ToIntOptional(const ValuePtrList &inputs, size_t i) {
  return ConvertOptional<Int64ImmPtr>(inputs, i);
}

std::optional<FP32ImmPtr> ValueConverter::ToFloatOptional(const ValuePtrList &inputs, size_t i) {
  return ConvertOptional<FP32ImmPtr>(inputs, i);
}

std::optional<BoolImmPtr> ValueConverter::ToBoolOptional(const ValuePtrList &inputs, size_t i) {
  return ConvertOptional<BoolImmPtr>(inputs, i);
}

std::optional<ScalarPtr> ValueConverter::ToScalarOptional(const ValuePtrList &inputs, size_t i) {
  return ConvertOptional<ScalarPtr>(inputs, i);
}

std::optional<tensor::BaseTensorPtr> ValueConverter::ToTensorOptional(const ValuePtrList &inputs, size_t i) {
  return ConvertOptional<tensor::BaseTensorPtr>(inputs, i);
}

std::optional<StringImmPtr> ValueConverter::ToStringOptional(const ValuePtrList &inputs, size_t i) {
  return ConvertOptional<StringImmPtr>(inputs, i);
}

std::optional<TypePtr> ValueConverter::ToDtypeOptional(const ValuePtrList &inputs, size_t i) {
  return ConvertOptional<TypePtr>(inputs, i);
}

std::optional<ValueTuplePtr> ValueConverter::ToValueTupleOptional(const ValuePtrList &inputs, size_t i) {
  return ConvertOptional<ValueTuplePtr>(inputs, i);
}

tensor::BaseTensorPtr ValueConverter::ContiguousTensorValue(OpRunnerInfo *op_runner_info,
                                                            const tensor::BaseTensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(op_runner_info);
  if (op_runner_info->device_target == kAscendDevice) {
    return tensor;
  }

  return GetContiguousTensor(op_runner_info, tensor);
}

ValueTuplePtr ValueConverter::ContiguousTensorValue(OpRunnerInfo *op_runner_info, const ValueTuplePtr &tuple) {
  MS_EXCEPTION_IF_NULL(op_runner_info);
  MS_EXCEPTION_IF_NULL(tuple);
  if (op_runner_info->device_target == kAscendDevice) {
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
    if (!val->isa<tensor::BaseTensor>()) {
      // No need to contiguous, when tuple is not tensor tuple.
      break;
    }

    const auto &tensor = val->cast<tensor::BaseTensorPtr>();
    auto contiguous_tensor = GetContiguousTensor(op_runner_info, tensor);
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
