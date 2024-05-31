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
  auto dim_opt = GetScalarValue<int64_t>(input_args[kInputIndex2]->GetValue());
  auto x_rank = SizeToLong(input_shape.size());

  int64_t tmp_pos;
  if (dim_opt.has_value()) {
    int64_t dim = dim_opt.value();
    tmp_pos = dim < 0 ? x_rank + dim : dim;
  } else {
    tmp_pos = x_rank - 1;
  }

  auto y_shape = input_shape;
  if (!input_args[kInputIndex1]->GetType()->isa<TypeNone>()) {
    auto n_opt = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());
    if (n_opt.has_value()) {
      auto n = n_opt.value();

      y_shape[tmp_pos] = n;
      if (primitive->name() == prim::kPrimIHFFT->name() || primitive->name() == prim::kPrimRFFT->name()) {
        y_shape[tmp_pos] = n / 2 + 1;
      }
    }
  } else {
    if (primitive->name() == prim::kPrimHFFT->name() || primitive->name() == prim::kPrimIRFFT->name()) {
      y_shape[tmp_pos] = (y_shape[tmp_pos] - 1) * 2;
    } else if (primitive->name() == prim::kPrimIHFFT->name() || primitive->name() == prim::kPrimRFFT->name()) {
      y_shape[tmp_pos] = y_shape[tmp_pos] / 2 + 1;
    }
  }

  return std::make_shared<abstract::TensorShape>(y_shape);
}

void FFTNGetAttr(const std::vector<AbstractBasePtr> &input_args, std::vector<int64_t> *s, std::vector<int64_t> *dim) {
  auto input_shape_ptr = input_args[kIndex0]->GetShape();
  auto input_shape = input_shape_ptr->GetShapeVector();
  auto x_rank = input_shape.size();

  bool s_is_none = input_args[kInputIndex1]->GetType()->isa<TypeNone>();
  bool dim_is_none = input_args[kInputIndex2]->GetType()->isa<TypeNone>();

  if (!s_is_none) {
    auto s_opt = GetArrayValue<int64_t>(input_args[kInputIndex1]->GetValue());
    if (s_opt.has_value()) {
      s->clear();
      *s = s_opt.value().ToVector();
    }
  }

  if (!dim_is_none) {
    auto dim_opt = GetArrayValue<int64_t>(input_args[kInputIndex2]->GetValue());
    if (dim_opt.has_value()) {
      dim->clear();
      *dim = dim_opt.value().ToVector();
      for (size_t i = 0; i < dim->size(); i++) {
        (*dim)[i] = (*dim)[i] < 0 ? x_rank + (*dim)[i] : (*dim)[i];
      }
    }
  }

  if (dim->empty() && !s->empty()) {
    for (size_t i = 0; i < s->size(); i++) {
      (void)dim->emplace_back(x_rank - s->size() + i);
    }
  }
  if (s->empty() && !dim->empty()) {
    for (size_t i = 0; i < dim->size(); i++) {
      (void)s->emplace_back(input_shape[(*dim)[i]]);
    }
  }
  if (s->empty() && dim->empty()) {
    for (size_t i = 0; i < x_rank; i++) {
      (void)dim->emplace_back(i);
      (void)s->emplace_back(input_shape[i]);
    }
  }
}

BaseShapePtr FFTNInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto input_shape_ptr = input_args[kIndex0]->GetShape();
  auto input_shape = input_shape_ptr->GetShapeVector();

  if (IsDynamicRank(input_shape)) {
    ShapeVector dyn_output{abstract::TensorShape::kShapeRankAny};
    return std::make_shared<abstract::TensorShape>(dyn_output);
  }

  std::string op_name = primitive->name();
  static const std::vector<std::string> same_shape_prim = {prim::kPrimFFT2->name(), prim::kPrimFFTN->name(),
                                                           prim::kPrimIFFT2->name(), prim::kPrimIFFTN->name()};
  static const std::vector<std::string> half_shape_prim = {prim::kPrimIHFFT2->name(), prim::kPrimIHFFTN->name(),
                                                           prim::kPrimRFFT2->name(), prim::kPrimRFFTN->name()};
  static const std::vector<std::string> double_shape_prim = {prim::kPrimHFFT2->name(), prim::kPrimHFFTN->name(),
                                                             prim::kPrimIRFFT2->name(), prim::kPrimIRFFTN->name()};

  bool is_same_shape_prim = std::find(same_shape_prim.begin(), same_shape_prim.end(), op_name) != same_shape_prim.end();
  bool is_half_shape_prim = std::find(half_shape_prim.begin(), half_shape_prim.end(), op_name) != half_shape_prim.end();
  bool is_double_shape_prim =
    std::find(double_shape_prim.begin(), double_shape_prim.end(), op_name) != double_shape_prim.end();
  bool s_is_none = input_args[kInputIndex1]->GetType()->isa<TypeNone>();
  bool dim_is_none = input_args[kInputIndex2]->GetType()->isa<TypeNone>();

  auto y_shape = input_shape;

  if (s_is_none && is_same_shape_prim) {
    return std::make_shared<abstract::TensorShape>(y_shape);
  }

  std::vector<int64_t> s;
  std::vector<int64_t> dim;
  if (!s_is_none) {
    auto s_opt = GetArrayValue<int64_t>(input_args[kInputIndex1]->GetValue());
    if (!s_opt.has_value()) {
      ShapeVector dyn_output{abstract::TensorShape::kShapeRankAny};
      return std::make_shared<abstract::TensorShape>(dyn_output);
    }
  }
  if (!dim_is_none) {
    auto dim_opt = GetArrayValue<int64_t>(input_args[kInputIndex2]->GetValue());
    if (!dim_opt.has_value()) {
      ShapeVector dyn_output{abstract::TensorShape::kShapeRankAny};
      return std::make_shared<abstract::TensorShape>(dyn_output);
    }
  }

  FFTNGetAttr(input_args, &s, &dim);

  for (size_t i = 0; i < s.size(); i++) {
    y_shape[dim[i]] = s[i];
  }

  if (is_double_shape_prim && s_is_none) {
    y_shape[dim.back()] = (y_shape[dim.back()] - 1) * 2;
  }

  if (is_half_shape_prim && s_is_none) {
    y_shape[dim.back()] = y_shape[dim.back()] / 2 + 1;
  }
  if (is_half_shape_prim && !s_is_none) {
    y_shape[dim.back()] = s.back() / 2 + 1;
  }
  return std::make_shared<abstract::TensorShape>(y_shape);
}

TypePtr FFTInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto input_type = input_args[kIndex0]->GetType();
  auto input_type_id = input_type->cast<TensorTypePtr>()->element()->type_id();
  std::string op_name = primitive->name();

  static const std::vector<TypeId> double_type = {kNumberTypeFloat64, kNumberTypeComplex128};
  bool is_double_type = std::any_of(double_type.begin(), double_type.end(),
                                    [&input_type_id](const TypeId &type_id) { return input_type_id == type_id; });

  static const std::vector<std::string> float_prim = {prim::kPrimHFFT->name(),   prim::kPrimHFFT2->name(),
                                                      prim::kPrimHFFTN->name(),  prim::kPrimIRFFT->name(),
                                                      prim::kPrimIRFFT2->name(), prim::kPrimIRFFTN->name()};
  bool is_float_prim = std::find(float_prim.begin(), float_prim.end(), op_name) != float_prim.end();

  TypePtr output_type;
  if (is_double_type && is_float_prim) {
    output_type = kFloat64;
  }
  if (is_double_type && !is_float_prim) {
    output_type = kComplex128;
  }
  if (!is_double_type && is_float_prim) {
    output_type = kFloat32;
  }
  if (!is_double_type && !is_float_prim) {
    output_type = kComplex64;
  }

  return std::make_shared<TensorType>(output_type);
}

void FFTCheckInputShape(const PrimitivePtr &primitive, std::vector<int64_t> x_shape_vec, int64_t x_rank) {
  const int64_t kMinRank = 1;
  const int64_t kMaxRank = 8;

  if (x_shape_vec.size() < kMinRank || x_shape_vec.size() > kMaxRank) {
    MS_EXCEPTION(ValueError) << CheckAndConvertUtils::FormatCheckInRangeMsg("rank of input", x_rank, kIncludeBoth,
                                                                            {kMinRank, kMaxRank}, primitive);
  }
  if (std::find(x_shape_vec.begin(), x_shape_vec.end(), 0) != x_shape_vec.end()) {
    MS_EXCEPTION(ValueError) << "Unsupported input shape dimension. The shape should not be empty.";
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
  int64_t x_rank = SizeToLong(x_shape_vec.size());
  FFTCheckInputShape(primitive, x_shape_vec, x_rank);

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
  int64_t x_rank = SizeToLong(x_shape_vec.size());
  FFTCheckInputShape(primitive, x_shape_vec, x_rank);

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
  if (std::set<int64_t>(dim.begin(), dim.end()).size() != dim.size()) {
    MS_EXCEPTION(ValueError) << "The dims must be unique.";
  }
  if (!s.empty() && !dim.empty() && s.size() != dim.size()) {
    MS_EXCEPTION(ValueError) << "When givec, dim and s arguuments must have the same length.";
  }
  return check_status;
}
}  // namespace ops
}  // namespace mindspore
