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

#include <set>
#include <memory>
#include <unordered_map>
#include "ops/op_utils.h"
#include "ops/ops_func_impl/op_func_impl.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/fft_arithmetic.h"

namespace mindspore {
namespace ops {
BaseShapePtr FFTInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto input_shape_ptr = input_args[kIndex0]->GetShape();
  auto input_shape = input_shape_ptr->GetShapeVector();

  // When input is a dynamic rank, it needs to be processed in the kernel
  if (IsDynamicRank(input_shape)) {
    ShapeVector dyn_output{abstract::TensorShape::kShapeRankAny};
    return std::make_shared<abstract::TensorShape>(dyn_output);
  }

  auto y_shape = input_shape;
  if (!input_args[kInputIndex1]->GetType()->isa<TypeNone>()) {
    auto n_opt = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());
    if (n_opt.has_value()) {
      auto n = n_opt.value();
      auto dim_opt = GetScalarValue<int64_t>(input_args[kInputIndex2]->GetValue());
      auto x_rank = input_shape.size();

      int64_t tmp_pos;
      if (dim_opt.has_value()) {
        int64_t dim = dim_opt.value();
        tmp_pos = dim < 0 ? x_rank + dim : dim;
      } else {
        tmp_pos = x_rank - 1;
      }
      y_shape[tmp_pos] = n;
    }
  }

  return std::make_shared<abstract::TensorShape>(y_shape);
}

BaseShapePtr FFTNInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto input_shape_ptr = input_args[kIndex0]->GetShape();
  auto input_shape = input_shape_ptr->GetShapeVector();

  if (IsDynamicRank(input_shape)) {
    ShapeVector dyn_output{abstract::TensorShape::kShapeRankAny};
    return std::make_shared<abstract::TensorShape>(dyn_output);
  }

  auto y_shape = input_shape;
  auto x_rank = input_shape.size();
  if (input_args[kInputIndex1]->GetType()->isa<TypeNone>()) {
    return std::make_shared<abstract::TensorShape>(y_shape);
  }

  auto s_opt = GetArrayValue<int64_t>(input_args[kInputIndex1]->GetValue());
  if (!s_opt.has_value()) {
    return std::make_shared<abstract::TensorShape>(y_shape);
  }
  std::vector<int64_t> s = s_opt.value().ToVector();

  std::vector<int64_t> dim;
  if (!input_args[kInputIndex2]->GetType()->isa<TypeNone>()) {
    auto dim_opt = GetArrayValue<int64_t>(input_args[kInputIndex2]->GetValue());
    if (dim_opt.has_value()) {
      dim = dim_opt.value().ToVector();
      for (size_t i = 0; i < s.size(); i++) {
        dim[i] = dim[i] < 0 ? x_rank + dim[i] : dim[i];
      }
    }
  }

  if (dim.size() == 0) {
    for (size_t i = 0; i < s.size(); i++) {
      (void)dim.emplace_back(x_rank - s.size() + i);
    }
  }

  for (size_t i = 0; i < s.size(); i++) {
    y_shape[dim[i]] = s[i];
  }
  return std::make_shared<abstract::TensorShape>(y_shape);
}

TypePtr FFTInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto input_type = input_args[kIndex0]->GetType();
  auto input_type_id = input_type->cast<TensorTypePtr>()->element()->type_id();

  static const std::vector<TypeId> double_type = {kNumberTypeFloat64, kNumberTypeComplex128};
  bool is_double_type = std::any_of(double_type.begin(), double_type.end(),
                                    [&input_type_id](const TypeId &type_id) { return input_type_id == type_id; });
  if (is_double_type) {
    return std::make_shared<TensorType>(kComplex128);
  } else {
    return std::make_shared<TensorType>(kComplex64);
  }
}

/*
  Error list:
  1) `input.ndim` is not in the range of "[1, 8]".
  2) The value in `dim` is not in the range of "[-`input.ndim`, `input.ndim`)"
  3) The value in `n` is less than or equal to 0.
*/
int32_t FFTCheckValidation(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto check_status = OP_CHECK_SUCCESS;
  const auto &input_x_shape = input_args[kIndex0]->GetShape();
  auto x_shape_vec = input_x_shape->GetShapeVector();

  if (MS_UNLIKELY(IsDynamicRank(x_shape_vec))) {
    check_status = OP_CHECK_RETRY;
  }

  const int64_t kMinRank = 1;
  const int64_t kMaxRank = 8;
  int64_t x_rank = SizeToLong(x_shape_vec.size());

  if (x_shape_vec.size() < kMinRank || x_shape_vec.size() > kMaxRank) {
    MS_EXCEPTION(ValueError) << CheckAndConvertUtils::FormatCheckInRangeMsg("rank of input", x_rank, kIncludeBoth,
                                                                            {kMinRank, kMaxRank}, primitive);
  }

  if (std::accumulate(x_shape_vec.begin(), x_shape_vec.end(), 0) == 0) {
    MS_EXCEPTION(ValueError) << "Unsupported input shape dimension. The shape should not be empty.";
  }

  if (!input_args[kInputIndex1]->GetType()->isa<TypeNone>()) {
    auto n_opt = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());
    if (n_opt.has_value()) {
      int64_t n = n_opt.value();
      (void)CheckAndConvertUtils::CheckInteger("n", n, kGreaterThan, 0);
    }
  }

  auto dim_opt = GetScalarValue<int64_t>(input_args[kInputIndex2]->GetValue());
  if (dim_opt.has_value()) {
    int64_t dim = dim_opt.value();
    if (dim < -x_rank || dim >= x_rank) {
      MS_EXCEPTION(ValueError) << CheckAndConvertUtils::FormatCheckInRangeMsg("dim", dim, kIncludeLeft,
                                                                              {-x_rank, x_rank - 1}, primitive);
    }
  }

  return check_status;
}

/*
  Error list:
  1) `input.ndim` is not in the range of "[1, 8]".
  2) The value in `dim` is not in the range of "[-`input.ndim`, `input.ndim`)"
  3) The value in `s` is less than or equal to 0.
  4) If `dim` has duplicate values.
  5）If `s` and `dim` are given but have different shapes.
*/
int32_t FFTNCheckValidation(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto check_status = OP_CHECK_SUCCESS;
  const auto &input_x_shape = input_args[kIndex0]->GetShape();
  auto x_shape_vec = input_x_shape->GetShapeVector();

  if (MS_UNLIKELY(IsDynamicRank(x_shape_vec))) {
    check_status = OP_CHECK_RETRY;
  }
  const int64_t kMinRank = 1;
  const int64_t kMaxRank = 8;
  int64_t x_rank = SizeToLong(x_shape_vec.size());

  if (x_shape_vec.size() < kMinRank || x_shape_vec.size() > kMaxRank) {
    MS_EXCEPTION(ValueError) << CheckAndConvertUtils::FormatCheckInRangeMsg("rank of input", x_rank, kIncludeBoth,
                                                                            {kMinRank, kMaxRank}, primitive);
  }

  if (std::accumulate(x_shape_vec.begin(), x_shape_vec.end(), 0) == 0) {
    MS_EXCEPTION(ValueError) << "Unsupported input shape dimension. The shape should not be empty.";
  }

  std::vector<int64_t> s;
  if (!input_args[kInputIndex1]->GetType()->isa<TypeNone>()) {
    auto s_opt = GetArrayValue<int64_t>(input_args[kInputIndex1]->GetValue());
    if (s_opt.has_value()) {
      s = s_opt.value().ToVector();
      for (size_t i = 0; i < s.size(); i++) {
        (void)CheckAndConvertUtils::CheckInteger("s", s[i], kGreaterThan, 0);
      }
    }
  }

  std::vector<int64_t> dim;
  if (!input_args[kInputIndex2]->GetType()->isa<TypeNone>()) {
    auto dim_opt = GetArrayValue<int64_t>(input_args[kInputIndex2]->GetValue());
    if (dim_opt.has_value()) {
      dim = dim_opt.value().ToVector();
      for (size_t i = 0; i < dim.size(); i++) {
        MS_CHECK_VALUE(
          dim[i] >= -x_rank && dim[i] < x_rank,
          CheckAndConvertUtils::FormatCheckInRangeMsg("dim", dim[i], kIncludeLeft, {-x_rank, x_rank}, primitive));
      }
    }
  }

  if (!s.empty() && !dim.empty() && s.size() != dim.size()) {
    MS_EXCEPTION(ValueError) << "Unsupported input shape dimension. The shape should not be empty.";
  }

  return check_status;
}
}  // namespace ops
}  // namespace mindspore
