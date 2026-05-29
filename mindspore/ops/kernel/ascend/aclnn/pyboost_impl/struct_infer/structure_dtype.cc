/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/pyboost_impl/struct_infer/structure_dtype.h"
#include "ir/dtype/number.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/convert_utils.h"

namespace mindspore::kernel::pyboost::op_infer {
namespace {
struct ResultTypeState {
  TypeId dim_result{kTypeUnknown};
  TypeId zero_result{kTypeUnknown};
};

bool IsFloatingTypeId(TypeId t) {
  switch (t) {
    case kNumberTypeFloat16:
    case kNumberTypeBFloat16:
    case kNumberTypeFloat32:
    case kNumberTypeFloat64:
    case kNumberTypeFloat8E4M3FN:
    case kNumberTypeFloat8E5M2:
    case kNumberTypeHiFloat8:
      return true;
    default:
      return false;
  }
}

bool IsComplexType(TypeId t) { return t == kNumberTypeComplex64 || t == kNumberTypeComplex128; }

TypeId ToComplexTypeId(TypeId t) {
  switch (t) {
    case kNumberTypeComplex64:
    case kNumberTypeComplex128:
      return t;
    case kNumberTypeBFloat16:
    case kNumberTypeFloat16:
    case kNumberTypeFloat32:
      return kNumberTypeComplex64;
    case kNumberTypeFloat64:
      return kNumberTypeComplex128;
    default:
      MS_LOG(EXCEPTION) << "Cannot convert type " << TypeIdToString(t) << " to complex type.";
  }
  return kTypeUnknown;
}

TypeId PromoteSkipUndefined(TypeId a, TypeId b) {
  if (a == kTypeUnknown) {
    return b;
  }
  if (b == kTypeUnknown) {
    return a;
  }
  return ConvertTypeForTensorsOrScalars(a, b);
}

TypeId CombineCategories(TypeId higher, TypeId lower) {
  if (IsComplexType(higher)) {
    return higher;
  }
  if (IsComplexType(lower)) {
    if (IsFloatingTypeId(higher)) {
      return ToComplexTypeId(higher);
    }
    return lower;
  }
  if (IsFloatingTypeId(higher)) {
    return higher;
  }
  if (higher == kNumberTypeBool || IsFloatingTypeId(lower)) {
    return PromoteSkipUndefined(higher, lower);
  }
  if (higher != kTypeUnknown) {
    return higher;
  }
  return lower;
}

ResultTypeState UpdateResultTypeState(const tensor::TensorPtr &tensor, const ResultTypeState &in_state) {
  MS_EXCEPTION_IF_NULL(tensor);
  ResultTypeState new_state = in_state;
  const auto current = tensor->data_type();
  if (tensor->DataDim() > 0) {
    new_state.dim_result = PromoteSkipUndefined(in_state.dim_result, current);
  } else {
    new_state.zero_result = PromoteSkipUndefined(in_state.zero_result, current);
  }
  return new_state;
}

template <typename TensorRange>
TypeId ResultTypeFromTensors(const TensorRange &tensors) {
  ResultTypeState state;
  for (const auto &tensor : tensors) {
    // cppcheck-suppress useStlAlgorithm
    state = UpdateResultTypeState(tensor, state);
  }
  return CombineCategories(state.dim_result, state.zero_result);
}
}  // namespace

bool IsIntegralType(TypeId t, bool include_bool) {
  switch (t) {
    case kNumberTypeInt8:
    case kNumberTypeInt16:
    case kNumberTypeInt32:
    case kNumberTypeInt64:
    case kNumberTypeUInt8:
    case kNumberTypeUInt16:
    case kNumberTypeUInt32:
    case kNumberTypeUInt64:
      return true;
    case kNumberTypeBool:
      return include_bool;
    default:
      return false;
  }
}

bool IsFloatingType(TypeId t) { return IsFloatingTypeId(t); }

TypeId PromoteTypes(TypeId a, TypeId b) {
  if (a == kTypeUnknown) {
    return b;
  }
  if (b == kTypeUnknown) {
    return a;
  }
  return ops::PromoteType(a, b, "StructuredDtypeInfer");
}

TypeId ResultType(const tensor::TensorPtr &lhs, const tensor::TensorPtr &rhs) {
  return ResultTypeFromTensors(std::initializer_list<tensor::TensorPtr>{lhs, rhs});
}

TypeId ResultType(std::initializer_list<tensor::TensorPtr> tensors) { return ResultTypeFromTensors(tensors); }

TypeId ResultType(const std::vector<tensor::TensorPtr> &tensors) { return ResultTypeFromTensors(tensors); }
}  // namespace mindspore::kernel::pyboost::op_infer
