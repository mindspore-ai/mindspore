/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "ops/ops_func_impl/repeat_interleave_tensor.h"
#include <utility>
#include <memory>
#include <functional>
#include "ops/op_utils.h"
#include "mindspore/ccsrc/include/common/utils/convert_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
namespace {
bool check_repeats_dim(const PrimitivePtr &primitive, const ShapeVector &input_shape, const ShapeVector &repeats,
                       const ValuePtr &dim) {
  auto rank = SizeToLong(input_shape.size());
  auto repeats_size = SizeToLong(repeats.size());
  auto numel =
    std::accumulate(input_shape.begin(), input_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  if (!dim->isa<None>()) {
    auto dim_opt = GetScalarValue<int64_t>(dim);
    if (dim_opt.has_value()) {
      int64_t real_dim = dim_opt.value();
      MS_CHECK_VALUE(
        real_dim >= -rank && real_dim <= rank - 1,
        CheckAndConvertUtils::FormatCheckInRangeMsg("dim", real_dim, kIncludeBoth, {-rank, rank - 1}, primitive));
      real_dim = (real_dim < 0) ? (real_dim + rank) : real_dim;
      if (repeats_size == input_shape[real_dim] || repeats_size == 1) {
        return true;
      }
    } else {
      return true;
    }
  } else {
    if (repeats_size == numel || repeats_size == 1) {
      return true;
    }
  }
  if (repeats.empty()) {
    return true;
  }
  return false;
}

inline ShapeVector GetInferredShape(const ShapeVector &input_shape, const ShapeVector &repeats, const ValuePtr &dim) {
  ShapeVector result_shape;
  auto rank = SizeToLong(input_shape.size());
  auto numel =
    std::accumulate(input_shape.begin(), input_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  int64_t repeats_numel = SizeToLong(repeats.size());
  int64_t repeats_sum = 0;
  if (repeats.empty()) {
    repeats_sum = -1;
  }
  for (size_t i = 0; i < repeats.size(); ++i) {
    if (repeats[i] == -1) {
      repeats_sum = -1;
      break;
    }
    repeats_sum += repeats[i];
  }
  if (!dim->isa<None>()) {
    auto dim_opt = GetScalarValue<int64_t>(dim);
    if (dim_opt.has_value()) {
      int64_t real_dim = dim_opt.value();
      real_dim = (real_dim < 0) ? (real_dim + rank) : real_dim;
      for (int64_t i = 0; i < rank; i++) {
        if (i == real_dim) {
          int64_t arg = 1;
          if (repeats_numel == 1) {
            arg = input_shape[real_dim];
          }
          int64_t item = repeats_sum == -1 ? -1 : arg * repeats_sum;
          result_shape.emplace_back(item);
        } else {
          result_shape.emplace_back(input_shape[i]);
        }
      }
    } else {
      ShapeVector res_shape(rank, abstract::TensorShape::kShapeDimAny);
      return res_shape;
    }
  } else {
    int64_t base = repeats_sum;
    if (repeats_numel == 1 && base != -1) {
      base *= numel;
    }
    result_shape.emplace_back(base);
  }
  return result_shape;
}

template <typename T>
ShapeVector GetNewRepeats(const PrimitivePtr &primitive, const ArrayValue<T> repeats_values) {
  ShapeVector repeats;
  for (size_t i = 0; i < repeats_values.size(); i++) {
    if (repeats_values.IsValueUnknown(i)) {
      repeats.push_back(abstract::Shape::kShapeDimAny);
    } else {
      if (repeats_values[i] < 0) {
        MS_EXCEPTION(RuntimeError) << "For '" << primitive->name() << "', 'repeats' can not be negative.";
      }
      repeats.push_back(repeats_values[i]);
    }
  }
  return repeats;
}
}  // namespace

BaseShapePtr RepeatInterleaveTensorFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                        const std::vector<AbstractBasePtr> &input_args) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->ascend_soc_version() == kAscendVersion910) {
    MS_EXCEPTION(RuntimeError) << primitive->name() << " is only supported on Atlas A2 training series.";
  }

  auto x_base_shape = input_args[kInputIndex0]->GetShape();
  auto x_shape = x_base_shape->GetShapeVector();
  auto repeats_base_shape = input_args[kInputIndex1]->GetShape();
  auto repeats_shape = repeats_base_shape->GetShapeVector();
  if (MS_UNLIKELY(IsDynamicRank(x_shape)) || MS_UNLIKELY(IsDynamicRank(repeats_shape))) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeRankAny});
  }

  if (repeats_shape.size() > 1) {
    MS_EXCEPTION(RuntimeError) << "For '" << primitive->name() << "', 'repeats' must be 0-dim or 1-dim tensor.";
  }

  std::vector<TypeId> valid_types = {kNumberTypeInt32, kNumberTypeInt64};
  auto input1_tensor = input_args[kInputIndex1]->GetType()->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(input1_tensor);
  auto repeats_type = input1_tensor->element()->type_id();
  if (std::find(valid_types.begin(), valid_types.end(), repeats_type) == valid_types.end()) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name() << "', 'repeats' must be int32 or int64. but got "
                            << TypeIdToType(repeats_type)->ToString();
  }

  ShapeVector repeats;
  if (repeats_type == kNumberTypeInt32) {
    auto repeats_opt = GetArrayValue<int32_t>(input_args[kInputIndex1]);
    if (repeats_opt.has_value()) {
      auto repeats_values = repeats_opt.value();
      repeats = GetNewRepeats<int32_t>(primitive, repeats_values);
    }

  } else {
    auto repeats_opt = GetArrayValue<int64_t>(input_args[kInputIndex1]);
    if (repeats_opt.has_value()) {
      auto repeats_values = repeats_opt.value();
      repeats = GetNewRepeats<int64_t>(primitive, repeats_values);
    }
  }

  auto dim = input_args[kInputIndex2]->GetValue();
  if (!check_repeats_dim(primitive, x_shape, repeats, dim) && !IsDynamicShape(x_shape)) {
    MS_EXCEPTION(RuntimeError) << "For '" << primitive->name()
                               << "', 'repeats' must have the same size as input along dim.";
  }

  auto inferred_shape = GetInferredShape(x_shape, repeats, dim);
  return std::make_shared<abstract::TensorShape>(inferred_shape);
}

TypePtr RepeatInterleaveTensorFuncImpl::InferType(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetType();
}

ShapeArray RepeatInterleaveTensorFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                      const ValuePtrList &input_values) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->ascend_soc_version() == kAscendVersion910) {
    MS_EXCEPTION(RuntimeError) << primitive->name() << " is only supported on Atlas A2 training series.";
  }

  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto x_shape = x_tensor->shape();
  const auto &repeats_tensor = input_values[kInputIndex1]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(repeats_tensor);
  const auto repeats_shape = repeats_tensor->shape();
  if (repeats_shape.size() > 1) {
    MS_EXCEPTION(RuntimeError) << "For '" << primitive->name() << "', 'repeats' must be 0-dim or 1-dim tensor.";
  }

  std::vector<TypeId> valid_types = {kNumberTypeInt32, kNumberTypeInt64};
  auto repeats_type = repeats_tensor->data_type();
  if (std::find(valid_types.begin(), valid_types.end(), repeats_type) == valid_types.end()) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name() << "', 'repeats' must be int32 or int64. but got "
                            << TypeIdToType(repeats_type)->ToString();
  }

  ShapeVector repeats;
  if (repeats_type == kNumberTypeInt32) {
    auto repeats_opt = GetArrayValue<int32_t>(input_values[kInputIndex1]);
    auto repeats_values = repeats_opt.value();
    repeats = GetNewRepeats<int32_t>(primitive, repeats_values);
  } else {
    auto repeats_opt = GetArrayValue<int64_t>(input_values[kInputIndex1]);
    auto repeats_values = repeats_opt.value();
    repeats = GetNewRepeats<int64_t>(primitive, repeats_values);
  }

  auto dim = input_values[kInputIndex2];
  if (!check_repeats_dim(primitive, x_shape, repeats, dim)) {
    MS_EXCEPTION(RuntimeError) << "For '" << primitive->name()
                               << "', 'repeats' must have the same size as input along dim.";
  }

  auto inferred_shape = GetInferredShape(x_shape, repeats, dim);
  return ShapeArray{
    inferred_shape,
  };
}

TypePtrList RepeatInterleaveTensorFuncImpl::InferType(const PrimitivePtr &primitive,
                                                      const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &input_x_type = x_tensor->Dtype();
  TypePtrList type_ptr_list{input_x_type};
  return type_ptr_list;
}
REGISTER_SIMPLE_INFER(kNameRepeatInterleaveTensor, RepeatInterleaveTensorFuncImpl)
}  // namespace ops
}  // namespace mindspore
