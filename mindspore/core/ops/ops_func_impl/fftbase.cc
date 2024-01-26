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

#include "ops/ops_func_impl/fftbase.h"
#include <set>
#include <memory>
#include <unordered_map>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr FFTBaseFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape_ptr = input_args[kIndex0]->GetShape();
  auto input_shape = input_shape_ptr->GetShapeVector();

  // When input is a dynamic rank, it needs to be processed in the kernel
  if (IsDynamicRank(input_shape)) {
    ShapeVector dyn_output{abstract::TensorShape::kShapeRankAny};
    return std::make_shared<abstract::TensorShape>(dyn_output);
  }

  auto s_vec = GetArrayValue<int64_t>(input_args[kInputIndex1]->GetValue()).value();
  auto dims_vec = GetArrayValue<int64_t>(input_args[kInputIndex2]->GetValue()).value();
  auto x_rank = input_shape.size();

  auto y_shape = input_shape;
  // If s is given, the input will be zero-padded or trimmed to this length.
  int64_t tmp_pos;
  for (size_t i = 0; i < s_vec.size(); i++) {
    if (dims_vec.size() == 0) {
      tmp_pos = x_rank - s_vec.size() + i;
      y_shape[tmp_pos] = s_vec[i];
    } else {
      tmp_pos = dims_vec[i] < 0 ? x_rank + dims_vec[i] : dims_vec[i];
      y_shape[tmp_pos] = s_vec[i];
    }
  }
  return std::make_shared<abstract::TensorShape>(y_shape);
}

TypePtr FFTBaseFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
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
  3) Duplicate values exist in `dim`.
  4) `dim` and `s` have values at the same time, but they have different shapes.
  5) The value in `s` is less than or equal to 0.
*/
int32_t FFTBaseFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto check_status = OP_CHECK_SUCCESS;
  const auto &input_x_shape = input_args[kIndex0]->GetShape();
  auto x_shape_vec = input_x_shape->GetShapeVector();

  if (MS_UNLIKELY(IsDynamicRank(x_shape_vec))) {
    check_status = OP_CHECK_RETRY;
  }
  const int64_t x_min_rank = 1;
  const int64_t x_max_rank = 8;
  int64_t x_rank = SizeToLong(x_shape_vec.size());

  MS_CHECK_VALUE(x_shape_vec.size() >= x_min_rank && x_shape_vec.size() <= x_max_rank,
                 CheckAndConvertUtils::FormatCheckInRangeMsg("rank of input", x_rank, kIncludeBoth,
                                                             {x_min_rank, x_max_rank}, primitive));

  if (x_rank == 1 && x_shape_vec[0] == 0) {
    MS_EXCEPTION(ValueError) << "Unsupported input shape dimension. The shape should not be empty.";
  }

  auto s = input_args[kIndex1]->GetValue();
  auto dims = input_args[kIndex2]->GetValue();
  auto s_vec = GetArrayValue<int64_t>(s).value();
  auto dims_vec = GetArrayValue<int64_t>(dims).value();
  auto fft_mode = input_args[kIndex4]->GetValue();
  auto fft_mode_id = GetScalarValue<int64_t>(fft_mode).value();
  std::map<int64_t, string> fft_mode_map = {{0, "fft"}, {1, "ifft"}};
  string s_str = "s";
  if (fft_mode_id <= 1) {
    s_str = "n";
    (void)CheckAndConvertUtils::CheckInteger(s_str, s_vec.size(), kLessEqual, 1, fft_mode_map.at(fft_mode_id));
    (void)CheckAndConvertUtils::CheckInteger("dims", dims_vec.size(), kLessEqual, 1, fft_mode_map.at(fft_mode_id));
  }

  std::set<size_t> seen;
  for (size_t i = 0; i < dims_vec.size(); ++i) {
    MS_CHECK_VALUE(
      dims_vec[i] >= -x_rank && dims_vec[i] < x_rank,
      CheckAndConvertUtils::FormatCheckInRangeMsg("dim", dims_vec[i], kIncludeRight, {-x_rank, x_rank}, nullptr));

    MS_CHECK_VALUE(
      seen.count(dims_vec[i]) == 0,
      CheckAndConvertUtils::FormatCommMsg(" 'dim' should all be unique dim, but", dims_vec[i], " is not unique!"));
    seen.insert(dims_vec[i]);
  }

  for (size_t i = 0; i < s_vec.size(); ++i) {
    (void)CheckAndConvertUtils::CheckInteger(s_str, s_vec[i], kGreaterThan, 0);
  }

  if (s_vec.size() > 0 && dims_vec.size() > 0 && s_vec.size() != dims_vec.size()) {
    MS_EXCEPTION(ValueError) << "When given, " << s_str << " and dim arguments must have the same length";
  }
  return check_status;
}
}  // namespace ops
}  // namespace mindspore
